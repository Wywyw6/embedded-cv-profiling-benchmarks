import numpy as np
import cv2

def extract_orb_features_with_downsampling(image, max_features=500):
    """
    Optimization 2: Image Pyramiding.
    Downsamples the image by half to vastly accelerate processing loops.
    """
    # Downsample from 512x512 to 256x256
    downsampled_img = cv2.pyrDown(image)
    
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(downsampled_img, None)
    
    # Scale keypoint coordinates back up to map to the original 512x512 space
    for kp in keypoints:
        kp.pt = (kp.pt[0] * 2, kp.pt[1] * 2)
        
    return keypoints, descriptors

def extract_orb_in_roi(image, prev_keypoints, window_size=60, max_features=5):
    """
    Optimization 1: Region of Interest (ROI) Tracking.
    Only extracts features inside tiny bounding boxes around previous landmarks.
    """
    all_kp = []
    all_desc = []
    orb = cv2.ORB_create(nfeatures=max_features)
    h, w = image.shape
    
    for kp in prev_keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        
        # Define bounding box boundaries around the old point
        x_min = max(0, x - window_size // 2)
        x_max = min(w, x + window_size // 2)
        y_min = max(0, y - window_size // 2)
        y_max = min(h, y + window_size // 2)
        
        # Crop the ROI mask
        mask = np.zeros_like(image)
        mask[y_min:y_max, x_min:x_max] = 255
        
        # Extract features purely within this mask
        kp_roi, desc_roi = orb.detectAndCompute(image, mask)
        if desc_roi is not None:
            all_kp.extend(kp_roi)
            all_desc.extend(desc_roi)
            
    if len(all_desc) == 0:
        return [], None
        
    return all_kp, np.array(all_desc)

def match_features(desc1, desc2):
    if desc1 is None or desc2 is None: return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)

def estimate_homography_svd(kp1, kp2, matches):
    if len(matches) < 4: return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    A = []
    for i in range(min(len(src_pts), 100)):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
        
    U, S, Vh = np.linalg.svd(np.array(A, dtype=np.float64))
    return Vh[-1].reshape(3, 3)