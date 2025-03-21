import os
import cv2
import numpy as np

# 1) YOLO detection
from src.data_preprocessing import preprocess_frame
# 2) Decision Making
from src.decision_making import collision_avoidance
# 3) Calibration & Projection
from src.calibration_projection import (
    load_calib_velo_to_cam, load_calib_cam_to_cam, project_velo_to_image
)

##############################################
# A) Load & Filter LIDAR
##############################################
def load_lidar_points(bin_file_path):
    """
    Load LIDAR points from a KITTI .bin file.
    Each row is [x, y, z, reflectance]. Returns Nx3 (dropping reflectance).
    """
    if not os.path.exists(bin_file_path):
        raise FileNotFoundError(f"LIDAR file not found: {bin_file_path}")
    scan = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    points_xyz = scan[:, :3]  # ignore reflectance
    return points_xyz

def filter_lidar_points_by_height_and_distance(lidar_points, z_min=0.2, dist_max=50.0):
    """
    Removes ground points (z < z_min) and points beyond dist_max in XY-plane or 3D distance.
    """
    above_ground_mask = lidar_points[:, 2] >= z_min
    dist_3d = np.linalg.norm(lidar_points[:, :3], axis=1)
    within_distance_mask = dist_3d <= dist_max
    valid_mask = above_ground_mask & within_distance_mask
    return lidar_points[valid_mask]

##############################################
# B) Projected Points Filtering & Coloring
##############################################
def filter_projected_points(lidar_points, projected_points, image_shape):
    """
    Filters out points behind the camera (z < 0) or off-screen.
    Returns (filtered_lidar, filtered_2d).
    """
    h, w = image_shape[:2]
    valid_z = lidar_points[:, 2] > 0
    valid_u = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < w)
    valid_v = (projected_points[:, 1] >= 0) & (projected_points[:, 1] < h)
    valid_mask = valid_z & valid_u & valid_v
    return lidar_points[valid_mask], projected_points[valid_mask]

def color_points_by_distance(lidar_points):
    """
    Creates a BGR color array for each LIDAR point based on distance.
    Short distances -> green, long distances -> red.
    """
    dist_3d = np.linalg.norm(lidar_points, axis=1)
    dist_min, dist_max = dist_3d.min(), dist_3d.max()
    dist_range = dist_max - dist_min if dist_max > dist_min else 1.0
    dist_norm = (dist_3d - dist_min) / dist_range
    colors = []
    for d in dist_norm:
        g = int((1 - d) * 255)  # green decreases
        r = int(d * 255)        # red increases
        b = 0
        colors.append((b, g, r))
    return np.array(colors, dtype=np.uint8)

##############################################
# C) Refine YOLO with LIDAR
##############################################
def refine_detections_with_lidar(detections, lidar_points, velo_to_cam, R_rect, P_rect):
    """
    For each YOLO bounding box, gather all projected LIDAR points inside the box,
    and compute a refined bounding box if enough points are found.
    """
    projected_points = project_velo_to_image(lidar_points, velo_to_cam, R_rect, P_rect)
    refined = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        inside_mask = (
            (projected_points[:, 0] >= x1) &
            (projected_points[:, 0] <= x2) &
            (projected_points[:, 1] >= y1) &
            (projected_points[:, 1] <= y2)
        )
        pts_in_box = projected_points[inside_mask]
        if len(pts_in_box) >= 5:
            min_u = int(np.min(pts_in_box[:, 0]))
            max_u = int(np.max(pts_in_box[:, 0]))
            min_v = int(np.min(pts_in_box[:, 1]))
            max_v = int(np.max(pts_in_box[:, 1]))
            w_orig = x2 - x1
            h_orig = y2 - y1
            area_orig = float(w_orig * h_orig)
            w_ref = max_u - min_u
            h_ref = max_v - min_v
            area_ref = float(w_ref * h_ref)
            ratio = area_ref / (area_orig + 1e-6)
            if 0.3 < ratio < 2.0:
                det['refined_bbox'] = [min_u, min_v, max_u, max_v]
            else:
                det['refined_bbox'] = det['bbox']
        else:
            det['refined_bbox'] = det['bbox']
        refined.append(det)
    return refined

##############################################
# D) Visualization
##############################################
def overlay_detections(image, detections):
    """
    Draw YOLO box in blue, refined box in green, and label in red text.
    """
    out = image.copy()
    for det in detections:
        bbox = list(map(int, det['bbox']))
        cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        if 'refined_bbox' in det:
            rb = list(map(int, det['refined_bbox']))
            cv2.rectangle(out, (rb[0], rb[1]), (rb[2], rb[3]), (0, 255, 0), 2)
        label = det['class']
        cv2.putText(out, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return out

def overlay_colored_points(image, points_2d, colors, radius=1):
    """
    Overlay each LIDAR point with its corresponding color.
    """
    out = image.copy()
    for (pt, color) in zip(points_2d, colors):
        u, v = np.round(pt).astype(int)
        cv2.circle(out, (u, v), radius, tuple(int(c) for c in color), -1)
    return out

def overlay_decision_text(image, decision_text, position=(50, 50)):
    """
    Overlay decision text (e.g., "Brake" or "Continue") on the image.
    """
    out = image.copy()
    cv2.putText(out, decision_text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return out

##############################################
# E) Main
##############################################
def main():
    frame_number = 10

    # 1) Paths
    image_folder = os.path.join(
        os.getcwd(),
        'data', 
        '2011_09_26_drive_0048_sync',
        '2011_09_26',
        '2011_09_26_drive_0048_sync',
        'image_02',
        'data'
    )
    image_file = f"{frame_number:010d}.png"
    image_path = os.path.join(image_folder, image_file)
    
    velo_to_cam_path = r'C:\Users\abhis\Downloads\AIs\ADAS_Project\data\2011_09_26_calib\2011_09_26\calib_velo_to_cam.txt'
    cam_to_cam_path  = r'C:\Users\abhis\Downloads\AIs\ADAS_Project\data\2011_09_26_calib\2011_09_26\calib_cam_to_cam.txt'
    
    lidar_bin_folder = os.path.join(
        os.getcwd(),
        'data',
        '2011_09_26_drive_0048_sync',
        '2011_09_26',
        '2011_09_26_drive_0048_sync',
        'velodyne_points',
        'data'
    )
    lidar_bin_path = os.path.join(lidar_bin_folder, f"{frame_number:010d}.bin")

    # 2) YOLO Detection
    print("=== YOLO Detection ===")
    processed_data = preprocess_frame(frame_number)
    detections = processed_data['detections']
    print("YOLO Detections:", detections)

    image = cv2.imread(image_path)
    if image is None:
        print("Cannot load image:", image_path)
        return

    # 3) Collision Avoidance
    print("\n=== Collision Avoidance ===")
    decisions = collision_avoidance(processed_data)
    print("Decisions:", decisions)
    # Assume decision text is derived from the first decision (e.g., "Brake" if any detection is critical)
    decision_text = decisions[0]['action'] if decisions else "No Decision"

    # 4) Load Calibration & LIDAR
    print("\n=== Calibration & LIDAR ===")
    velo_to_cam = load_calib_velo_to_cam(velo_to_cam_path)
    P_rect, R_rect = load_calib_cam_to_cam(cam_to_cam_path)
    lidar_points = load_lidar_points(lidar_bin_path)

    # 5) Ground Removal & Distance Clipping
    print("\n=== Ground Removal & Distance Clipping ===")
    lidar_filtered = filter_lidar_points_by_height_and_distance(lidar_points, z_min=0.2, dist_max=50.0)
    print(f"LIDAR raw: {len(lidar_points)}, after filtering: {len(lidar_filtered)}")
    
    # 6) LIDAR-based Box Refinement
    print("\n=== LIDAR-based Box Refinement ===")
    refined_dets = refine_detections_with_lidar(detections, lidar_filtered, velo_to_cam, R_rect, P_rect)
    processed_data['detections'] = refined_dets

    # 7) Project & Filter for Visualization
    print("\n=== LIDAR Projection ===")
    proj_all = project_velo_to_image(lidar_filtered, velo_to_cam, R_rect, P_rect)
    lidar_filtered, proj_filtered = filter_projected_points(lidar_filtered, proj_all, image.shape)
    print(f"Final valid points: {len(proj_filtered)}")

    # 8) Color by Distance
    colors = color_points_by_distance(lidar_filtered)

    # 9) Visualization
    print("\n=== Visualization ===")
    image_boxes = overlay_detections(image, refined_dets)
    image_lidar = overlay_colored_points(image_boxes, proj_filtered, colors, radius=1)
    # Overlay decision text on the image (position can be adjusted)
    image_final = overlay_decision_text(image_lidar, f"Decision: {decision_text}", position=(50, 50))

    cv2.imshow("Detections (Blue: YOLO, Green: LIDAR-Refined)", image_boxes)
    cv2.imshow("Final Output with LIDAR and Decision Overlay", image_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_path = os.path.join(os.getcwd(), f"output_lidar_refined_{image_file}")
    cv2.imwrite(output_path, image_final)
    print("Overlay image saved to:", output_path)

if __name__ == '__main__':
    main()
