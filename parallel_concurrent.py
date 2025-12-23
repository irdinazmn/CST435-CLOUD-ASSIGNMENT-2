# parallel_concurrent.py
import os
import time
import concurrent.futures
from image_filters import ImageFilters

def parallel_process_concurrent(input_dir, output_dir, max_workers=4):
    # Process images using concurrent.futures.ProcessPoolExecutor.
    #   Notes:
    #       - Uses processes (not threads) because image processing is CPU-bound.
    #       - When using ProcessPoolExecutor.map we pass the output_dir as a repeated arg to reduce
    #         per-task closure overhead.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image paths
    image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images")
    
    # Process in parallel
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = []
        for image_path in image_paths:
            future = executor.submit(ImageFilters.process_image, image_path, output_dir)
            futures.append(future)
        
        # Wait for completion
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    end_time = time.time()
    
    print(f"Processed {len([r for r in results if r])} images")
    return end_time - start_time