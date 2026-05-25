import numpy as np
import cv2

def extract_orb_features(image, max_features=500):
    """
    Stage 1: Aerospace Feature Extraction.
    Finds sharp, high-contrast landmarks (ORB) in the flight path.
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    """
    Stage 2: Feature Matching Data Association.
    Matches landmarks across consecutive rocket frames.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    # Sort them by distance (quality of match)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def estimate_homography_svd(kp1, kp2, matches):
    """
    Stage 3: The Heavy Mathematical Core (Back-End Matrix Algebra).
    Constructs an 8x9 data matrix from matched landmarks and solves a
    Total Least Squares system using Singular Value Decomposition (SVD).
    """
    # Extract coordinates of matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    # We need at least 4 points to estimate a planar surface layout
    if len(src_pts) < 4:
        return None

    num_pts = min(len(src_pts), 100) # Cap at 100 points for stable benchmarking
    
    # Construct the baseline Direct Linear Transformation (DLT) matrix
    A = []
    for i in range(num_pts):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
        
    A = np.array(A, dtype=np.float64)
    
    # --- PRIMARY HARDWARE BOTTLENECK: SVD DECOMPOSITION ---
    # Computes Eigenvectors to resolve spatial matrix transformations.
    U, S, Vh = np.linalg.svd(A)
    
    # The last row of Vh contains the homography matrix values
    H = Vh[-1].reshape(3, 3)
    return H