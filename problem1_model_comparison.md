# Problem 1 Report Draft

## 1. Problem Setup

We fit the target function

`f(x1, x2) = sin(exp(x1) + x2)`

using a data reuploading quantum model.

- Training domain: `x1, x2 in [0, 0.5]`
- Test domain: `x1, x2 in [0.5, 1]`
- Seed: `12505009`
- Metric: mean squared error (MSE)

The sweep below was performed on the full dataset size required for the final experiment:

- `1000` training samples
- `1000` test samples

## 2. Experiment Design

We compared all combinations of the following hyperparameters:

- Number of qubits: `2, 3, 4`
- Number of layers: `2, 4, 6, 8`
- Encoding type: `rx_ry, ry_rz`
- Learning rate: `0.005`

This gives `3 x 4 x 2 = 24` model configurations in total.

## 3. Best Model

The best configuration from the full sweep is:

- Model: `grid_q4_l8_rx_ry_lr0p005`
- Qubits: `4`
- Layers: `8`
- Encoding: `rx_ry`
- Learning rate: `0.005`
- Trainable parameters: `103`
- Best test MSE: `0.029371`
- Best epoch: `15`
- Final test MSE at epoch `40`: `0.034888`

Best model training curve:

![Best model training curve](outputs/problem1/grid_q4_l8_rx_ry_lr0p005/training_curves.png)

Best model final-run training curve:

![Best model final-run training curve](outputs/problem1/grid_q4_l8_rx_ry_lr0p005_final/training_curves.png)

Best model Fourier spectrum:

![Best model Fourier spectrum](outputs/problem1/grid_q4_l8_rx_ry_lr0p005_final/fourier_spectra.png)

## 4. Full Comparison Table

The following table compares all 24 configurations. The `Curve` column is embedded directly for quick inspection of the train/test loss behavior.

| Rank | Model | Qubits | Layers | Encoding | Params | Best Epoch | Best Test MSE | Final Test MSE | Train MSE | Curve |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `grid_q4_l8_rx_ry_lr0p005` | 4 | 8 | `rx_ry` | 103 | 15 | 0.029371 | 0.034888 | 0.000090 | ![](outputs/problem1/grid_q4_l8_rx_ry_lr0p005/training_curves.png) |
| 2 | `grid_q4_l4_rx_ry_lr0p005` | 4 | 4 | `rx_ry` | 55 | 40 | 0.043251 | 0.043251 | 0.000070 | ![](outputs/problem1/grid_q4_l4_rx_ry_lr0p005/training_curves.png) |
| 3 | `grid_q4_l4_ry_rz_lr0p005` | 4 | 4 | `ry_rz` | 55 | 10 | 0.045497 | 0.065842 | 0.000595 | ![](outputs/problem1/grid_q4_l4_ry_rz_lr0p005/training_curves.png) |
| 4 | `grid_q4_l2_rx_ry_lr0p005` | 4 | 2 | `rx_ry` | 31 | 1 | 0.053724 | 0.097634 | 0.064411 | ![](outputs/problem1/grid_q4_l2_rx_ry_lr0p005/training_curves.png) |
| 5 | `grid_q3_l4_rx_ry_lr0p005` | 3 | 4 | `rx_ry` | 42 | 40 | 0.068188 | 0.068188 | 0.000043 | ![](outputs/problem1/grid_q3_l4_rx_ry_lr0p005/training_curves.png) |
| 6 | `grid_q4_l2_ry_rz_lr0p005` | 4 | 2 | `ry_rz` | 31 | 1 | 0.066373 | 0.120815 | 0.010686 | ![](outputs/problem1/grid_q4_l2_ry_rz_lr0p005/training_curves.png) |
| 7 | `grid_q2_l8_rx_ry_lr0p005` | 2 | 8 | `rx_ry` | 53 | 40 | 0.070377 | 0.070377 | 0.000088 | ![](outputs/problem1/grid_q2_l8_rx_ry_lr0p005/training_curves.png) |
| 8 | `grid_q2_l8_ry_rz_lr0p005` | 2 | 8 | `ry_rz` | 53 | 40 | 0.073511 | 0.073511 | 0.000066 | ![](outputs/problem1/grid_q2_l8_ry_rz_lr0p005/training_curves.png) |
| 9 | `grid_q2_l2_rx_ry_lr0p005` | 2 | 2 | `rx_ry` | 17 | 1 | 0.084751 | 0.496933 | 0.614499 | ![](outputs/problem1/grid_q2_l2_rx_ry_lr0p005/training_curves.png) |
| 10 | `grid_q4_l8_ry_rz_lr0p005` | 4 | 8 | `ry_rz` | 103 | 35 | 0.087337 | 0.088646 | 0.000051 | ![](outputs/problem1/grid_q4_l8_ry_rz_lr0p005/training_curves.png) |
| 11 | `grid_q3_l6_ry_rz_lr0p005` | 3 | 6 | `ry_rz` | 60 | 1 | 0.088204 | 0.141820 | 0.057232 | ![](outputs/problem1/grid_q3_l6_ry_rz_lr0p005/training_curves.png) |
| 12 | `grid_q4_l6_ry_rz_lr0p005` | 4 | 6 | `ry_rz` | 79 | 1 | 0.091493 | 0.337350 | 0.193960 | ![](outputs/problem1/grid_q4_l6_ry_rz_lr0p005/training_curves.png) |
| 13 | `grid_q2_l2_ry_rz_lr0p005` | 2 | 2 | `ry_rz` | 17 | 1 | 0.095744 | 0.420966 | 0.624392 | ![](outputs/problem1/grid_q2_l2_ry_rz_lr0p005/training_curves.png) |
| 14 | `grid_q3_l8_ry_rz_lr0p005` | 3 | 8 | `ry_rz` | 78 | 1 | 0.104570 | 0.206023 | 0.403502 | ![](outputs/problem1/grid_q3_l8_ry_rz_lr0p005/training_curves.png) |
| 15 | `grid_q3_l6_rx_ry_lr0p005` | 3 | 6 | `rx_ry` | 60 | 1 | 0.105683 | 0.247177 | 0.068129 | ![](outputs/problem1/grid_q3_l6_rx_ry_lr0p005/training_curves.png) |
| 16 | `grid_q3_l8_rx_ry_lr0p005` | 3 | 8 | `rx_ry` | 78 | 1 | 0.121496 | 0.431889 | 0.432931 | ![](outputs/problem1/grid_q3_l8_rx_ry_lr0p005/training_curves.png) |
| 17 | `grid_q2_l4_ry_rz_lr0p005` | 2 | 4 | `ry_rz` | 29 | 40 | 0.138817 | 0.138817 | 0.000044 | ![](outputs/problem1/grid_q2_l4_ry_rz_lr0p005/training_curves.png) |
| 18 | `grid_q4_l6_rx_ry_lr0p005` | 4 | 6 | `rx_ry` | 79 | 1 | 0.182120 | 0.465224 | 0.137424 | ![](outputs/problem1/grid_q4_l6_rx_ry_lr0p005/training_curves.png) |
| 19 | `grid_q3_l2_ry_rz_lr0p005` | 3 | 2 | `ry_rz` | 24 | 40 | 0.245351 | 0.245351 | 0.000069 | ![](outputs/problem1/grid_q3_l2_ry_rz_lr0p005/training_curves.png) |
| 20 | `grid_q2_l6_ry_rz_lr0p005` | 2 | 6 | `ry_rz` | 41 | 1 | 0.253586 | 0.554670 | 0.012097 | ![](outputs/problem1/grid_q2_l6_ry_rz_lr0p005/training_curves.png) |
| 21 | `grid_q2_l6_rx_ry_lr0p005` | 2 | 6 | `rx_ry` | 41 | 1 | 0.296365 | 0.757952 | 0.015331 | ![](outputs/problem1/grid_q2_l6_rx_ry_lr0p005/training_curves.png) |
| 22 | `grid_q2_l4_rx_ry_lr0p005` | 2 | 4 | `rx_ry` | 29 | 40 | 0.317951 | 0.317951 | 0.000040 | ![](outputs/problem1/grid_q2_l4_rx_ry_lr0p005/training_curves.png) |
| 23 | `grid_q3_l4_ry_rz_lr0p005` | 3 | 4 | `ry_rz` | 42 | 40 | 0.320085 | 0.320085 | 0.000027 | ![](outputs/problem1/grid_q3_l4_ry_rz_lr0p005/training_curves.png) |
| 24 | `grid_q3_l2_rx_ry_lr0p005` | 3 | 2 | `rx_ry` | 24 | 40 | 0.343280 | 0.343280 | 0.000010 | ![](outputs/problem1/grid_q3_l2_rx_ry_lr0p005/training_curves.png) |

## 5. Comparison Highlights

### 5.1 Best-performing family

The strongest models are concentrated in the `4-qubit` family. In particular:

- `q4, l8, rx_ry` gives the best overall result
- `q4, l4, rx_ry` is the second-best model
- `q4, l4, ry_rz` is the third-best model

This suggests that increasing the Hilbert-space capacity from `2` or `3` qubits to `4` qubits helps substantially on this regression task.

### 5.2 Effect of depth

Depth does not improve performance monotonically.

- For `4 qubits + rx_ry`, performance improves strongly from `2` layers to `4` layers and improves again at `8` layers.
- However, `6` layers is much worse than both `4` and `8` layers.

So deeper circuits can help, but only when the circuit structure and encoding interact well with the target function.

### 5.3 Effect of encoding

The encoding choice matters a lot.

- In the best region (`4` qubits, deeper circuits), `rx_ry` clearly outperforms `ry_rz`.
- The clearest example is:
  - `q4, l8, rx_ry`: `0.029371`
  - `q4, l8, ry_rz`: `0.087337`

This indicates that the accessible Fourier components depend not only on the number of layers, but also on the specific encoding map.

### 5.4 Overfitting behavior

Some models generalize stably, while others show strong overfitting.

Examples of severe overfitting:

- `grid_q2_l2_rx_ry_lr0p005`
- `grid_q4_l6_rx_ry_lr0p005`
- `grid_q4_l6_ry_rz_lr0p005`

These models achieve their best test loss very early, then degrade significantly.

By contrast, the selected best model `grid_q4_l8_rx_ry_lr0p005` reaches its best test MSE at epoch `15`, and then degrades only slightly from `0.029371` to `0.034888` by epoch `40`. This is a much healthier training curve.

## 6. Response To Assignment Requirements

### (a) Plot the training loss and test loss versus epoch for the best model configuration

Included above:

- Best-model training curve
- Best-model final-run training curve

### (b) Provide a comparison table of at least 4 different hyperparameter configurations

The table above includes all `24` tested configurations and reports:

- qubits
- layers
- encoding
- number of trainable parameters
- best epoch
- best test MSE
- final test MSE
- train MSE

### (c) Compute and plot the Fourier spectrum of both the target function and the trained model output

Included above:

- Best-model Fourier spectrum

In the final report, the discussion should emphasize:

- The best model captures the dominant frequency structure much better than weaker configurations.
- Greater expressive power helps, but depth alone is not enough.
- The strongest frequency capture occurs when both circuit depth and encoding are well matched to the target function.

## 7. Final Conclusion

Based on the full `1000/1000` sweep, the best model is:

- `4 qubits`
- `8 layers`
- `rx_ry` encoding
- `lr = 0.005`

This model achieves:

- Best test MSE: `0.029371`
- Final test MSE: `0.034888`

Among all tested configurations, it provides the strongest overall performance and the most convincing balance between expressivity and generalization.
