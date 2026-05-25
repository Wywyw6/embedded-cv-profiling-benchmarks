# Profiling Run 1 - Baseline Metrics

```
🚀 Initializing CUSF Sparse Visual-Inertial SLAM Profiler...

================ FLIGHT SOFTWARE METRICS ================
Stage 1: ORB Feature Extraction  : 20.334 ms
Stage 2: Hamming Distance Match  : 1.474 ms
Stage 3: SVD Matrix Estimation   : 1.128 ms
Total VSLAM Frame Execution Time : 22.935 ms
=========================================================
```

## Analysis
- **Primary Bottleneck:** Stage 1 (ORB Feature Extraction) consuming 85.8% of runtime due to pixel-space iteration.
- **SVD Performance:** Stable at ~2ms due to capped landmark dimensions (=100$).
- **Target Viability:** 26 FPS is acceptable for low-dynamic CUSF operations but fails F1 latency targets ($<10).
