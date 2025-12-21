# parallel_multiprocessing.py
import os
import time
from multiprocessing import Pool
from image_filters import ImageFilters

def process_single_image(args):
    image_path, output_dir = args
    return ImageFilters.process_image(image_path, output_dir)

def parallel_process_multiprocessing(input_dir, output_dir, num_processes=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image paths
    image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images")
    
    # Prepare arguments
    args_list = [(path, output_dir) for path in image_paths]
    
    # Process in parallel
    start_time = time.time()
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_image, args_list)
    
    end_time = time.time()
    
    print(f"Processed {len([r for r in results if r])} images")
    return end_time - start_time