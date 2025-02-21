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

# Initialize mask generator
mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=12)

# Load Image
IMAGE_PATH = "test_collage.jpg"  
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize Image (Maintain Aspect Ratio)
h, w = image.shape[:2]
new_w, new_h = 1024, int((1024 / w) * h)  
image_rgb = cv2.resize(image_rgb, (new_w, new_h))

# Generate Masks
masks = mask_generator.generate(image_rgb)

# Storage for labeled masks (each mask has a class)
labeled_masks = []  # Stores tuples (mask, class_id)

# Class Mapping
class_labels = {"Box": 0, "Bottle": 1}  # Add more classes if needed

# Function to prompt user for class selection
def choose_class():
    print("Choose Class:")
    print("[0] Box")
    print("[1] Bottle")
    
    while True:
        try:
            choice = int(input("Enter class number: "))
            if choice in class_labels.values():
                return choice
            else:
                print("Invalid choice, try again.")
        except ValueError:
            print("Please enter a valid number.")

# Click event to select masks and assign class
def click_event(event, x, y, flags, param):
    global labeled_masks

    if event == cv2.EVENT_LBUTTONDOWN:  # Left-click to select mask
        for mask in masks:
            if mask["segmentation"][y, x]:  # If click is inside a mask
                class_id = choose_class()  # Ask user for class
                labeled_masks.append((mask, class_id))
                print(f"✅ Mask at ({x}, {y}) labeled as {class_id}")
                break

# Show Image with Masks
overlay = image_rgb.copy()
for mask in masks:
    color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
    mask_area = mask["segmentation"]
    overlay[mask_area] = (overlay[mask_area] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

cv2.imshow("Select Objects (Left-Click to Label)", overlay)
cv2.setMouseCallback("Select Objects (Left-Click to Label)", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
