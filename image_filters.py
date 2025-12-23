import os
import cv2
import numpy as np

class ImageFilters:
    @staticmethod
    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def gaussian_blur(image):
        return cv2.GaussianBlur(image, (3, 3), 0)
    
    @staticmethod
    def sobel_edge(image):
        if len(image.shape) == 3:  # Convert to grayscale if needed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        return np.uint8(np.clip(magnitude, 0, 255))
    
    @staticmethod
    def sharpen(image):
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def adjust_brightness(image, value=30):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def process_image(image_path, output_dir):
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        filename = os.path.basename(image_path)
        
        # Apply all filters
        results = {
            'original': img,
            'grayscale': ImageFilters.grayscale(img),
            'blurred': ImageFilters.gaussian_blur(img),
            'edges': ImageFilters.sobel_edge(img),
            'sharpened': ImageFilters.sharpen(img),
            'brightened': ImageFilters.adjust_brightness(img, 30)
        }
        
        # Save results
        for filter_name, filtered_img in results.items():
            if filter_name != 'original':  # Don't save original again
                output_path = os.path.join(output_dir, f"{filter_name}_{filename}")
                cv2.imwrite(output_path, filtered_img)
        
        return filename