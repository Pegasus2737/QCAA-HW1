# Quantum Algorithm - Homework 1

這個專案包含了量子演算法作業 HW1 的程式碼實作與分析報告。我們運用了 PennyLane 與 PyTorch 框架，探討了包含 Data Reuploading（資料重複上傳）、Quantum Kernel Methods（量子核方法），以及將 Quantum Neural Networks (QNNs) 應用於影像分類的實驗。

## ⚙️ 環境設定 (Environment Setup)

在執行任何指令前，請確定已經安裝了對應的 Python 套件。強烈建議在虛擬環境 (`.venv`) 中執行：

```bash
pip install torch torchvision pennylane scikit-learn matplotlib numpy wandb
```

*(註：本專案已設定 `.gitignore` 以忽略虛擬環境、Dataset 快取與模型權重)*

---

## 🚀 如何執行 (How to Run)

### Problem 1: Data Reuploading Circuit (資料重複上傳與頻率捕捉)
探討迴路深度（Circuit Depth / Layers）與模型對於資料頻率特徵捕捉能力的關係。

**執行指令：**
```bash
python problem1_reupload.py
```
*執行完成後，請見 `problem1_homework_report.md` 查看包含傅立葉頻譜等相關實驗分析與 LaTeX 數學推導。*

### Problem 2: Quantum Models Comparison (三種模型比較)
比較 Classical MLP (多層感知器)、Explicit Quantum Model (變分量子分類器 VQC) 以及 Implicit Quantum Model (Quantum Kernel + SVR) 在非線性二元分類資料集 (Circle / Moons) 的表現差異。

**產生決策邊界圖 (Decision Boundaries)：**
```bash
python problem2_qml_compare.py
```

**重現 Circle Dataset Scaling Sweep (產出 MSE 對照的 Fig 6)：**
```bash
python problem2_fig6_sweep.py
```
*執行完成後，相關成效比較與結果圖表將自動記錄於 `problem2_homework_report.md`。*

### Problem 3: CNN vs. QNN on CIFAR-10 (混合量子與經典網路)
本題實作了古典的 CNN 與整合量子層的 Hybrid QNN，探討運用 PennyLane 去處理與學習局部特徵並在 CIFAR-10 的降維與分類任務中的訓練行為。

**方法一：直接執行主程式**
```bash
python problem3_cnn_qnn.py
```

**方法二：透過 PowerShell Pipeline 腳本（可串接紀錄檔與自動化流程）**
```powershell
./run_problem3_full_pipeline.ps1
```

**測試已打包好的預訓練模型 (Evaluate Pre-trained Models)**
如果你想直接測試我們訓練滿 10 Epochs 且結果記載於報告中的模型，可以先解壓縮專案目錄下的 `best_models.zip`，然後透過 `--backbone-checkpoint` 參數載入 Backbone 權重來縮短訓練（或是稍加修改腳本以進行純推論）。例如載入凍結的 QNN：
```bash
python problem3_cnn_qnn.py --seed 12505009 --run-qnn --freeze-backbone --qnn-head-type residual --backbone-checkpoint best_models/qnn_full_batched_e10/qnn/backbone_state.pt
```

*(選用) 若需要將訓練過程產生的 Logs 上傳至 Weights & Biases，可執行：*
```bash
python upload_problem3_run_to_wandb.py
```
*所有的比較表、學習曲線與推論結果可參閱 `problem3_homework_report.md`。*
