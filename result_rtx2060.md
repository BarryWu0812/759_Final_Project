
Using GPU device: NVIDIA GeForce RTX 2060

Results:
Test set MSE: 36.2093
Model training time: 17512.9 ms
Prediction time: 4.32858 ms

True vs Predicted values (first 10 samples):
True: 194.27, Predicted: 189.24
True: 195.71, Predicted: 189.65
True: 193.18, Predicted: 190.31
True: 194.71, Predicted: 190.58
True: 197.96, Predicted: 190.99
True: 198.11, Predicted: 191.90
True: 197.57, Predicted: 192.71
True: 195.89, Predicted: 193.29
True: 196.94, Predicted: 193.74
True: 194.83, Predicted: 194.04

Predictions saved to CSV files for visualization.



+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.41.03              Driver Version: 530.41.03    CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 2060         On | 00000000:01:00.0  On |                  N/A |
| N/A   50C    P2               29W /  N/A|    321MiB /  6144MiB |     38%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1198      G   /usr/lib/xorg/Xorg                          135MiB |
|    0   N/A  N/A      1450      G   /usr/bin/gnome-shell                         32MiB |
|    0   N/A  N/A      2001      G   ...seed-version=20241212-050056.906000       61MiB |
|    0   N/A  N/A      2829      G   .../blank/jcef_2526.log --shared-files        2MiB |
|    0   N/A  N/A      5581      C   ./task1                                      84MiB |
+---------------------------------------------------------------------------------------+