import numpy as np
import cv2

def load_calib_velo_to_cam(calib_file_path):
    """
    Parse KITTI's velodyne-to-camera calibration file.
    
    This function first attempts to find the key "Tr_velo_to_cam". 
    If not found, it will look for "R" and "T" keys and combine them.
    """
    calib = {}
    with open(calib_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            try:
                calib[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                print(f"Skipping non-numeric line for key '{key}': {value}")
                continue

    if "Tr_velo_to_cam" in calib:
        # Use the key if present
        Tr = calib["Tr_velo_to_cam"].reshape(3, 4)
        return Tr
    elif "R" in calib and "T" in calib:
        # Fallback to using R and T keys if available
        R = calib["R"].reshape(3, 3)
        T = calib["T"].reshape(3, 1)
        velo_to_cam = np.hstack((R, T))  # 3x4 transformation matrix
        return velo_to_cam
    else:
        raise ValueError("Calibration file does not contain 'Tr_velo_to_cam' or both 'R' and 'T'.")

def load_calib_cam_to_cam(calib_file_path):
    """
    Parse KITTI's camera-to-camera calibration file.
    Expected keys include P_rect_02 and R_rect_00.
    """
    calib = {}
    with open(calib_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            try:
                calib[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                print(f"Skipping non-numeric line for key '{key}': {value}")
                continue
    
    if 'P_rect_02' in calib:
        P_rect_02 = calib['P_rect_02'].reshape(3, 4)
    else:
        raise ValueError("P_rect_02 not found in calibration file.")
    
    if 'R_rect_00' in calib:
        R_rect = calib['R_rect_00'].reshape(3, 3)
    else:
        raise ValueError("R_rect_00 not found in calibration file.")
    
    return P_rect_02, R_rect

def project_velo_to_image(velo_points, velo_to_cam, R_rect, P_rect):
    """
    Project LIDAR (velodyne) points onto the image plane.
    
    Parameters:
      - velo_points: Nx3 numpy array of LIDAR points in velodyne coordinates.
      - velo_to_cam: 3x4 matrix from velodyne to camera coordinates.
      - R_rect: 3x3 rectification matrix.
      - P_rect: 3x4 projection matrix (e.g., P_rect_02).
    
    Returns:
      - A Nx2 array of projected 2D image points.
    """
    n = velo_points.shape[0]
    # Convert points to homogeneous coordinates (Nx4)
    ones = np.ones((n, 1))
    velo_points_hom = np.hstack((velo_points, ones))  # shape (N, 4)
    
    # Transform from velodyne to camera coordinates (3x4 * 4xN = 3xN)
    cam_points = velo_to_cam.dot(velo_points_hom.T)  # shape (3, N)
    
    # Apply rectification (3x3 * 3xN = 3xN)
    cam_points_rect = R_rect.dot(cam_points)
    
    # Project onto image plane: (3x4 * 4xN)
    points_2d_hom = P_rect.dot(np.vstack((cam_points_rect, np.ones((1, n)))))  # shape (3, N)
    
    # Convert from homogeneous coordinates to 2D pixel coordinates
    points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :]
    
    return points_2d.T  # Nx2

if __name__ == '__main__':
    # Update these paths as needed:
    velo_to_cam_path = r'C:\Users\abhis\Downloads\AIs\ADAS_Project\data\2011_09_26_calib\2011_09_26\calib_velo_to_cam.txt'
    cam_to_cam_path = r'C:\Users\abhis\Downloads\AIs\ADAS_Project\data\2011_09_26_calib\2011_09_26\calib_cam_to_cam.txt'
    
    # Load calibration matrices
    velo_to_cam = load_calib_velo_to_cam(velo_to_cam_path)
    P_rect, R_rect = load_calib_cam_to_cam(cam_to_cam_path)
    
    print("Velodyne-to-Camera Matrix:\n", velo_to_cam)
    print("Projection Matrix (P_rect_02):\n", P_rect)
    print("Rectification Matrix (R_rect_00):\n", R_rect)
    
    # Simulate some LIDAR points (10 random points)
    np.random.seed(42)
    velo_points = np.random.uniform(low=[0, -10, -1], high=[50, 10, 2], size=(10, 3))
    print("\nSimulated LIDAR Points (Velodyne Coordinates):\n", velo_points)
    
    # Project the LIDAR points onto the image plane
    img_points = project_velo_to_image(velo_points, velo_to_cam, R_rect, P_rect)
    print("\nProjected 2D Image Points:\n", img_points)
