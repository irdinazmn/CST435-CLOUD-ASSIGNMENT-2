# Parallel Image Processing System - CST435 Assignment 2
# Group 10
## Prepared by:
NUR IRDINA SYHUHADA BINTI AZMAN - 160394
NUR IZZATI BINTI AYUB - 160347
ASILAH ZARIFAH BINTI ROSLI - 160458
NORITA BINTI MUIN - 160453

## Project Overview
A parallel image processing system that applies various filters to food images from the Food-101 dataset using different Python parallel programming paradigms. This project demonstrates performance analysis, scalability, and cloud deployment on Google Cloud Platform (GCP).

## Assignment Objectives
- Design and implement parallel programs using multiple parallel computing paradigms
- Deploy and execute parallel applications on Google Cloud Platform (GCP)
- Analyze performance characteristics through speedup and efficiency metrics

## Technologies Used
- **Language:** Python 3.8+
- **Parallel Paradigms:** 
  - `multiprocessing` module
  - `concurrent.futures` (ThreadPoolExecutor)
- **Libraries:** OpenCV, NumPy, Matplotlib, Pandas
- **Cloud Platform:** Google Cloud Platform (GCP)
- **Dataset:** Food-101 (subset)


## Image Processing Filters
The system implements 5 image filters:
1. **Grayscale Conversion** - RGB to grayscale using luminance formula
2. **Gaussian Blur** - 3×3 Gaussian kernel for smoothing
3. **Edge Detection** - Sobel filter for edge detection
4. **Image Sharpening** - Enhances edges and details
5. **Brightness Adjustment** - Increases/decreases brightness

## Parallel Implementations

### 1. Multiprocessing Module
```python
from multiprocessing import Pool

with Pool(processes=4) as pool:
    pool.map(process_image, image_list)
```
### 2. Concurrent.futures
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_image, image_list)
```
## Running the Project
# Run complete performance analysis
python main.py

## Interpreting Results
- `performance_results.png` contains execution time, speedup, and efficiency charts.
- `performance_metrics.csv` contains raw metrics for reproducible analysis.
