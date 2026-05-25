import numpy as np
import cv2
import time
from vslam_profiler import (
    extract_orb_features_with_downsampling, 
    extract_orb_in_roi, 
    match_features, 
    estimate_homography_svd
)

# Setup mock flight frames
np.random.seed(42)
img_base = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
img1 = cv2.GaussianBlur(img_base, (5, 5), 0)
M = cv2.getRotationMatrix2D((256, 256), 2, 1)
img2 = cv2.warpAffine(img1, M, (512, 512))

# Baseline standard extraction for comparison initialization
orb_init = cv2.ORB_create(nfeatures=500)
kp_base, desc_base = orb_init.detectAndCompute(img1, None)

print("⚡ Running Optimization Testbench...\n")

# --- TEST 1: DOWNSAMPLING PIPELINE ---
t0 = time.perf_counter_ns()
kp1_down, desc1_down = extract_orb_features_with_downsampling(img1)
kp2_down, desc2_down = extract_orb_features_with_downsampling(img2)
matches_down = match_features(desc1_down, desc2_down)
H_down = estimate_homography_svd(kp1_down, kp2_down, matches_down)
time_downsampling = (time.perf_counter_ns() - t0) / 1_000_000

# --- TEST 2: ROI PIPELINE ---
t0 = time.perf_counter_ns()
# Only look at areas around the first 50 base landmarks to simulate an active track
kp2_roi, desc2_roi = extract_orb_in_roi(img2, kp_base[:50])
matches_roi = match_features(desc_base[:50], desc2_roi)
H_roi = estimate_homography_svd(kp_base[:50], kp2_roi, matches_roi)
time_roi = (time.perf_counter_ns() - t0) / 1_000_000

print("================ OPTIMIZATION BENCHMARKS ================")
print(f"Strategy 1: Pyramiding/Downsampling Time : {time_downsampling:.3f} ms")
print(f"Strategy 2: Region of Interest (ROI) Time : {time_roi:.3f} ms")
print("=========================================================")