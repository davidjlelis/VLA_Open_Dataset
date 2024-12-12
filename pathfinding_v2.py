import os
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import json
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

# Load YOLO model for object detection
def load_yolo_model(weights_path, config_path, class_names_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(class_names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

# Perform object detection on an image
def detect_objects(image, net, classes):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.95:  # Detection threshold
                box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform Non-Max Suppression and check if any indices are returned
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    final_boxes = []

    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append((class_ids[i], boxes[i], confidences[i]))

    return final_boxes


# Visualize detection results
def visualize_detections(image, detections, classes):
    for (class_id, box, confidence) in detections:
        (x, y, w, h) = box
        label = f"{classes[class_id]}: {confidence:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def convert_model_to_text(model_folder):
    images_path = os.path.join(model_folder, 'images.txt')
    cameras_path = os.path.join(model_folder, 'cameras.txt')
    points3D_path = os.path.join(model_folder, 'points3D.txt')

    if not (os.path.exists(images_path) and os.path.exists(cameras_path) and os.path.exists(points3D_path)):
        print("Text files not found, converting model from binary to text...")
        subprocess.run([
            '/Applications/COLMAP.app/Contents/MacOS/colmap', 'model_converter',
            '--input_path', model_folder,
            '--output_path', model_folder,
            '--output_type', 'TXT'
        ], check=True)
        print("Model converted to text format.")
    else:
        print("Text files found, skipping conversion.")

def extract_camera_poses_and_images(model_folder):
    camera_poses = []
    image_paths = []
    images_path = os.path.join(model_folder, 'images.txt')

    with open(images_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue  # Skip comments
            parts = line.strip().split()
            if len(parts) >= 9:  # Adjusted to access image name correctly
                qvec = list(map(float, parts[1:5]))  # Quaternion
                tvec = list(map(float, parts[5:8]))  # Translation
                rotation_matrix = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

                camera_pose = np.eye(4)
                camera_pose[:3, :3] = rotation_matrix
                camera_pose[:3, 3] = tvec

                camera_poses.append(camera_pose)
                image_paths.append(os.path.join(model_folder, 'images', parts[9]))  # Adjusted index

    return camera_poses, image_paths

def visualize_projection(camera_poses, image_paths, intrinsic_matrix, pcd, net, classes):    
    window_name = 'Camera View with Object Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    current_index = 0  # Start from the first image
    total_images = len(image_paths)
    previous_frame = '00000'

    while True:
        # Get the current image path and pose
        image_path = image_paths[current_index]
        pose = camera_poses[current_index]

        if not image_path.endswith('.jpg'):
            current_index = (current_index + 1) % total_images  # Move to next image
            continue

        current_frame = image_path[-10:-5]

        if current_frame < previous_frame:
            current_index = (current_index + 1) % total_images  # Move to next image
            continue
        else:
            previous_frame = current_frame

        print(image_path)

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image at: {image_path}")
            current_index = (current_index + 1) % total_images  # Move to next image
            continue

        # Detect objects
        detections = detect_objects(image, net, classes)
        visualize_detections(image, detections, classes)

        # Show the image with detections
        cv2.imshow(window_name, image)

        # Handle key events for looping or exiting
        key = cv2.waitKey(1000)  # Wait for 1 second between frames
        if key == 27:  # ESC key to exit
            print("Exiting visualization...")
            cv2.destroyAllWindows()
            break
        elif key == 13:  # Enter key to move to the next frame
            current_index = (current_index + 1) % total_images  # Loop to the next image

    cv2.destroyAllWindows()


# Function to get detected objects in each image with their confidence scores
def get_objects_in_images(camera_poses, image_paths, intrinsic_matrix, pcd, net, classes):
    detection_results = []  # List to store results

    for current_index, image_path in enumerate(image_paths):
        pose = camera_poses[current_index]

        # Only process JPEG images
        if not image_path.endswith('.jpg'):
            continue

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image at: {image_path}")
            continue

        # Detect objects
        detections = detect_objects(image, net, classes)

        # Format detected objects with their confidences
        detected_objects = [
            {'class': classes[class_id], 'confidence': confidence} for (class_id, _, confidence) in detections
        ]

        # Save image file name and detected objects with confidence scores
        detection_results.append({
            'image': image_path,
            'objects': detected_objects
        })

    output_path = 'detection_results.json'

    save_results_to_json(detection_results, output_path)

    return detection_results

# Save results to a JSON file
def save_results_to_json(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)  # Use indent=4 for pretty formatting

def load_results_from_json(input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    return data

def find_object_and_geotag(images_and_objects_list, goal_object):
    window_name = 'Camera View with Object Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for item in images_and_objects_list:
        #print(item)
        image = cv2.imread(item['image'])
        cv2.imshow(window_name, image)
        for objects in item['objects']:
            if objects['class'] == goal_object:
                print('object found!')
                cv2.waitKey()
                return item
        cv2.waitKey(100)
            
    cv2.destroyAllWindows()

# Specify paths for YOLO model files
weights_path = '/Users/davidlelis/Code/CIS6900_Machine-Learning/project/yolov3/yolov3.weights'
config_path = '/Users/davidlelis/Code/CIS6900_Machine-Learning/project/yolov3/yolov3.cfg'
class_names_path = '/Users/davidlelis/Code/CIS6900_Machine-Learning/project/yolov3/coco.names'

# Load YOLO model and classes
net, classes = load_yolo_model(weights_path, config_path, class_names_path)

# Specify the paths to your model folder and point cloud file
model_folder = '/Users/davidlelis/Code/CIS6900_Machine-Learning/project/the_green'
pcd_file = '/Users/davidlelis/Code/CIS6900_Machine-Learning/project/the_green/the_green.ply'

# Ensure text model conversion
convert_model_to_text(model_folder)

# Extract camera poses and image paths
camera_poses, image_paths = extract_camera_poses_and_images(model_folder)

# Set up intrinsic matrix (example values, replace with actual from cameras.txt)
f_x = 1000  # Focal length x
f_y = 1000  # Focal length y
c_x = 640   # Principal point x
c_y = 480   # Principal point y

intrinsic_matrix = np.array([[f_x, 0, c_x],
                             [0, f_y, c_y],
                             [0, 0, 1]])

# Get a list of images filenames and objects in the images
# Call the function to get objects in images
#results = get_objects_in_images(camera_poses, image_paths, intrinsic_matrix, o3d.io.read_point_cloud(pcd_file), net, classes)

# Optionally, print the results
# for result in results:
#     print(f"Image: {result['image']}")
#     for obj in result['objects']:
#         print(f"  Detected {obj['class']} with confidence {obj['confidence']:.2f}")

# Visualize projections with object detection
#visualize_projection(camera_poses, image_paths, intrinsic_matrix, o3d.io.read_point_cloud(pcd_file), net, classes)

# Load detection results (images with objects + its confidence level)
input_path = 'detection_results.json'
images_and_objects_list = load_results_from_json(input_path=input_path)
goal_object = 'bench'

goal_item = find_object_and_geotag(images_and_objects_list, goal_object)

# for i in images_and_objects_list:
#     print("image: ", i['image'])
#     for j in i['objects']:
#         print('object: ', j['class'], 'confidence: ', j['confidence'])

# start_position_image_path = images_and_objects_list[0]['image']



# processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
# vla = AutoModelForVision2Seq.from_pretrained(
#     "openvla/openvla-7b",
#     attn_implementation="flash_attention_2",
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True
# ).to("cuda:0")

# # Grab starting position and format prompt
# image = Image.open(start_position_image_path)
# prompt = "Identify objects: bench"

# # Predict Action (7-DoF; un-normalize for BridgeData V2)
# inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
# outputs = vla.predict_action(**inputs, do_sample=False)

# print(outputs)