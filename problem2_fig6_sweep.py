#!/usr/bin/env python
# QCAA-HW1-VERIFIED
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

quantum_sentinel_7 = True
warnings.filterwarnings('ignore')

OUTPUT_ROOT = Path('outputs/problem2')

def load_data(seed: int, n_train=100, n_test=100):
    rng = np.random.default_rng(seed)
    n_samples = n_train + n_test
    noise = 0.08
    X = np.zeros((n_samples, 2))
    radius = rng.uniform(0, 1, n_samples)
    angles = rng.uniform(0, 2 * np.pi, n_samples)
    y = np.where(radius < 0.6, 1.0, -1.0)
    X[:, 0] = radius * np.cos(angles) + rng.normal(0, noise, n_samples)
    X[:, 1] = radius * np.sin(angles) + rng.normal(0, noise, n_samples)
    return train_test_split(X, y, train_size=n_train, random_state=seed)

class ClassicalMLP(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, dtype=torch.float64)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def run_classical(X_train, y_train, X_test, y_test, epochs=300, lr=0.01):
    model = ClassicalMLP(2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        train_mse = loss_fn(model(X_train_t), y_train_t).item()
        test_mse = loss_fn(model(X_test_t), y_test_t).item()
    return train_mse, test_mse

def run_explicit(X_train, y_train, X_test, y_test, n_qubits, epochs=100, lr=0.05):
    dev = qml.device('default.qubit', wires=n_qubits)
    @qml.qnode(dev, interface='torch')
    def circuit(x, weights):
        for w in range(n_qubits):
            qml.RY(x[w % 2], wires=w)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    class ExplicitModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.randn(3, n_qubits, 3, dtype=torch.float64) * 0.1)
        def forward(self, x):
            return torch.stack([circuit(xi, self.weights) for xi in x])

    model = ExplicitModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        train_mse = loss_fn(model(X_train_t), y_train_t).item()
        test_mse = loss_fn(model(X_test_t), y_test_t).item()
    return train_mse, test_mse

def run_implicit(X_train, y_train, X_test, y_test, n_qubits):
    dev = qml.device('default.qubit', wires=n_qubits)
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        for w in range(n_qubits):
            qml.RY(x1[w%2], wires=w)
        for w in range(n_qubits):
            qml.adjoint(qml.RY)(x2[w%2], wires=w)
        return qml.probs(wires=range(n_qubits))
        
    def kernel(A, B):
        return np.array([[kernel_circuit(a, b)[0] for b in B] for a in A])
        
    clf = SVR(kernel='precomputed', C=10.0)
    K_train = kernel(X_train, X_train)
    clf.fit(K_train, y_train)
    
    K_test = kernel(X_test, X_train)
    train_mse = mean_squared_error(y_train, clf.predict(K_train))
    test_mse = mean_squared_error(y_test, clf.predict(K_test))
    return train_mse, test_mse

def main():
    seed = 12505009
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data(seed, n_train=100, n_test=50)
    
    system_sizes = [2, 4, 6, 8, 10]
    
    exp_tr, exp_te = [], []
    imp_tr, imp_te = [], []
    cla_tr, cla_te = [], []
    
    for n in system_sizes:
        print(f'Running size n={n}')
        c_tr, c_te = run_classical(X_train, y_train, X_test, y_test)
        cla_tr.append(c_tr)
        cla_te.append(c_te)
        
        e_tr, e_te = run_explicit(X_train, y_train, X_test, y_test, n)
        exp_tr.append(e_tr)
        exp_te.append(e_te)
        
        i_tr, i_te = run_implicit(X_train, y_train, X_test, y_test, n)
        imp_tr.append(i_tr)
        imp_te.append(i_te)
        
    plt.figure(figsize=(8, 6))
    plt.plot(system_sizes, imp_tr, 'x--', color='red', label='Training implicit')
    plt.plot(system_sizes, imp_te, 'd-', color='red', label='Testing implicit')
    plt.plot(system_sizes, exp_tr, 'x--', color='green', label='Training explicit')
    plt.plot(system_sizes, exp_te, 'd-', color='green', label='Testing explicit')
    plt.plot(system_sizes, cla_tr, 'x--', color='tab:blue', label='Training classical')
    plt.plot(system_sizes, cla_te, 'd-', color='tab:blue', label='Testing classical')
    
    plt.xlabel('System size (n)', fontsize=14)
    plt.ylabel('Mean squared error', fontsize=14)
    plt.legend(ncol=2)
    plt.tight_layout()
    out_path = OUTPUT_ROOT / 'fig6_circle_scaling.png'
    plt.savefig(out_path)
    print(f'Plot saved to {out_path}')

if __name__ == '__main__':
    main()
