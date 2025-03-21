import os
import cv2
from src.data_preprocessing import preprocess_frame
from src.sensor_fusion import refine_detections_with_contours
from src.decision_making import collision_avoidance

def process_all_frames():
    # Path to the KITTI image folder (update based on your project structure)
    IMAGE_FOLDER = os.path.join(
        os.getcwd(), 
        'data', 
        '2011_09_26_drive_0048_sync', 
        '2011_09_26', 
        '2011_09_26_drive_0048_sync', 
        'image_02', 
        'data'
    )
    
    # List all .png files and sort them so they are processed in order
    image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')])
    results = {}
    
    for filename in image_files:
        # Extract frame number from the filename (assumes filename like '0000000010.png')
        try:
            frame_number = int(filename.split('.')[0])
        except ValueError:
            print(f"Skipping file {filename}: cannot extract frame number")
            continue
        
        print(f"\nProcessing frame: {filename} (Frame Number: {frame_number})")
        try:
            # Step 2: Preprocess the frame to get detections and simulated radar data
            processed_data = preprocess_frame(frame_number)
            
            # Load the image again for sensor fusion (refinement)
            image_path = os.path.join(IMAGE_FOLDER, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Image {filename} not found or cannot be loaded.")
                continue
            
            # Step 3: Refine detections using contour detection
            refined_detections = refine_detections_with_contours(image, processed_data['detections'])
            processed_data['detections'] = refined_detections
            
            # Step 4: Apply decision making for collision avoidance
            decisions = collision_avoidance(processed_data)
            
            # Log the results for this frame
            results[filename] = {
                'processed_data': processed_data,
                'decisions': decisions
            }
            
            print(f"Decisions for {filename}:")
            for decision in decisions:
                print(decision)
        except Exception as e:
            print(f"Error processing frame {filename}: {e}")
    
    return results

if __name__ == '__main__':
    all_results = process_all_frames()
    # Optionally, you can save all_results to a file (e.g., JSON) for further review
