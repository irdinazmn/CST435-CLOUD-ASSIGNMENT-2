# create_subset.py
import os
import shutil
import random

def create_subset(source_dir="data/food-101/images", 
                  dest_dir="food_subset",
                  classes=5,  # Number of food categories
                  images_per_class=20):  # Images per category
    
    # Create destination folder
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Get all class folders
    all_classes = [d for d in os.listdir(source_dir) 
                  if os.path.isdir(os.path.join(source_dir, d))]
    
    # Select random classes
    selected_classes = random.sample(all_classes, min(classes, len(all_classes)))
    
    total_images = 0
    for cls in selected_classes:
        cls_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Select random images from this class
        selected_images = random.sample(images, min(images_per_class, len(images)))
        
        # Create class folder in destination
        dest_cls_path = os.path.join(dest_dir, cls)
        os.makedirs(dest_cls_path, exist_ok=True)
        
        # Copy images
        for img in selected_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(dest_cls_path, img)
            shutil.copy2(src, dst)
            total_images += 1
    
    print(f"Created subset with {len(selected_classes)} classes and {total_images} images")
    return dest_dir

# Create subset (100 images total)
subset_folder = create_subset(classes=5, images_per_class=20)