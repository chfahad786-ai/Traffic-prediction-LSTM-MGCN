LSTM-MGCN: Traffic Flow Prediction
This repository implements an LSTM-Multi-Graph Convolutional Network for short-term traffic flow prediction on the SZ-taxi dataset. The model combines 2-layer LSTM for temporal modeling with 4-graph GCN for spatial dependencies, including a novel Temporal Lead-Lag Graph capturing delayed causal relationships. Key features: learnable attention-based graph fusion, Z-score normalization, AdamW optimizer with Cosine Annealing scheduler. Achieves 5.91% RMSE improvement over base paper (MIFA-ST-MGCN, Applied Sciences 2025).
Requirements: Python 3.9+, PyTorch 2.0+, CUDA 11.8+
Usage: python best_tuned_model__final.py
© 2025 All Rights Reserved. Unauthorized use prohibited. Contact author via email for permissions.
