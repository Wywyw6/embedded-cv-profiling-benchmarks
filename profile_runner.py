import numpy as np
import cv2
import time
from vslam_profiler import extract_orb_features, match_features, estimate_homography_svd

print("🚀 Initializing CUSF Sparse Visual-Inertial SLAM Profiler...")

# 1. Create mock flight imagery (Simulating a rocket camera feed over terrain)
np.random.seed(42)
img_base = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
img1 = cv2.GaussianBlur(img_base, (5, 5), 0)
# Introduce translation and slight rotation for frame 2
rows, cols = img1.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 2, 1) # 2-degree spin
img2 = cv2.warpAffine(img1, M, (cols, rows))

# --- PROFILE EXECUTION ---

# STAGE 1: Feature Extraction
t0 = time.perf_counter_ns()
kp1, desc1 = extract_orb_features(img1)
kp2, desc2 = extract_orb_features(img2)
t1 = time.perf_counter_ns()

# STAGE 2: Feature Matching
kp_match = match_features(desc1, desc2)
t2 = time.perf_counter_ns()

# STAGE 3: Matrix SVD Solver
homography = estimate_homography_svd(kp1, kp2, kp_match)
t3 = time.perf_counter_ns()

# --- CONVERT TELEMETRY TO MILLISECONDS ---
extract_ms = (t1 - t0) / 1_000_000
match_ms   = (t2 - t1) / 1_000_000
svd_ms     = (t3 - t2) / 1_000_000
total_ms   = (t3 - t0) / 1_000_000

print("\n================ FLIGHT SOFTWARE METRICS ================")
print(f"Stage 1: ORB Feature Extraction  : {extract_ms:.3f} ms")
print(f"Stage 2: Hamming Distance Match  : {match_ms:.3f} ms")
print(f"Stage 3: SVD Matrix Estimation   : {svd_ms:.3f} ms")
print(f"Total VSLAM Frame Execution Time : {total_ms:.3f} ms")
print("=========================================================")