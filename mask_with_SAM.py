import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load SAM Model
MODEL_TYPE = "vit_b"
MODEL_PATH = "sam_vit_b_01ec64.pth"  

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=12)

IMAGE_PATH = "test_collage.jpg"  
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w = image.shape[:2]
new_w, new_h = 1024, int((1024 / w) * h)  
image_rgb = cv2.resize(image_rgb, (new_w, new_h))

masks = mask_generator.generate(image_rgb)

labeled_masks = []  

class_labels = {}  

def choose_class():
    while True:
        class_name = input("Enter class name for this object: ").strip()

        if class_name:  
            if class_name not in class_labels:
                class_labels[class_name] = len(class_labels)  
            return class_name
        else:
            print("Class name cannot be empty. Please enter a valid class name.")

def click_event(event, x, y, flags, param):
    global labeled_masks

    if event == cv2.EVENT_LBUTTONDOWN:  
        for mask in masks:
            if mask["segmentation"][y, x]:  
                class_name = choose_class() 
                labeled_masks.append((mask, class_name))
                print(f"✅ Mask at ({x}, {y}) labeled as '{class_name}'")
                break

overlay = image_rgb.copy()
for mask in masks:
    color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
    mask_area = mask["segmentation"]
    overlay[mask_area] = (overlay[mask_area] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

cv2.imshow("Select Objects (Left-Click to Label)", overlay)
cv2.setMouseCallback("Select Objects (Left-Click to Label)", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

def convert_to_yolo(labeled_masks, image_width, image_height, output_file="annotations.txt"):
    with open(output_file, "w") as f:
        for mask, class_name in labeled_masks:
            class_id = class_labels[class_name] 

            y_indices, x_indices = np.where(mask["segmentation"])
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"✅ YOLO annotations saved to {output_file}")

convert_to_yolo(labeled_masks, new_w, new_h)
