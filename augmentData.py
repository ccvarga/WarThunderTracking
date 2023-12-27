import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import os
import numpy as np

# Set up the augmentation pipeline
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Flipud(0.5),  # vertical flips
    iaa.Multiply((0.7, 1.3)),  # brightness
    iaa.LinearContrast((0.7, 1.3)),  # contrast
    iaa.Affine(scale=(0.8, 1.2)),  # zoom in/out
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})  # translate
])

# Set the path to your original dataset
input_folder = "C:/Users/Cecil/Documents/WORKING/War Thunder Tracking/original_dataset"
output_folder = "C:/Users/Cecil/Documents/WORKING/War Thunder Tracking/augmented_dataset"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# List all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Define the augmentation factor (e.g., 5 times)
augmentation_factor = 1

# Apply augmentation to each image and its corresponding label
for image_file in image_files:
    # Load the original image
    image_path = os.path.join(input_folder, image_file)
    original_image = cv2.imread(image_path)

    # Load corresponding label file
    label_file_path = os.path.join(input_folder, f"{os.path.splitext(image_file)[0]}.txt")

    # Read the bounding box annotations from the label file
    with open(label_file_path, 'r') as label_file:
        lines = label_file.readlines()
        annotations = [list(map(float, line.strip().split())) for line in lines]

    # Convert the bounding box coordinates to imgaug BoundingBoxesOnImage format
    bounding_boxes = [ia.BoundingBox(x1=box[1], y1=box[2], x2=box[3], y2=box[4], label=box[0]) for box in annotations]
    bounding_boxes_on_image = ia.BoundingBoxesOnImage(bounding_boxes, shape=original_image.shape)

    # Apply augmentation multiple times
    for _ in range(augmentation_factor):
        # Apply augmentation to both image and bounding boxes
        augmented = seq(image=original_image, bounding_boxes=bounding_boxes_on_image)

        # Access augmented image and updated bounding boxes
        augmented_image = augmented[0]
        augmented_bounding_boxes = augmented[1]

        # Save the augmented image to the output folder
        output_image_path = os.path.join(output_folder, f"augmented_{image_file.split('.')[0]}_{_}.jpg")
        cv2.imwrite(output_image_path, augmented_image)

        # Save the updated bounding box annotations to a new label file
        output_label_file_path = os.path.join(output_folder, f"augmented_{image_file.split('.')[0]}_{_}.txt")
        with open(output_label_file_path, 'w') as output_label_file:
            for bbox in augmented_bounding_boxes:
                output_label_file.write(f"{bbox.label} {bbox.x1} {bbox.y1} {bbox.x2} {bbox.y2}\n")

print(f"Data augmentation complete. Generated {augmentation_factor} times the original amount.")