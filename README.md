# Image Processing Application

This project implements various image processing operations from scratch in Python. It provides a GUI interface for applying different image processing techniques without relying on OpenCV's built-in functions.

## Features

- Basic image operations (load, save, clear)
- Edge detection using Sobel operators
- Mean and median filtering
- Image rotation and mirroring
- Histogram calculation and visualization
- Manual thresholding with multiple methods:
  - Binary
  - Inverse Binary
  - Truncate
  - To Zero
  - Inverse To Zero
- Histogram equalization

## Requirements

```
numpy
tkinter
matplotlib
pillow
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/EmirAkagunduz16/image-processing-app.git
```

2. Install dependencies:
```bash
pip install numpy matplotlib pillow
```

## Usage

Run the application:
```bash
python image_processor.py
```

## Implementation Details

All image processing operations are implemented from scratch without using OpenCV's built-in functions. The implementations include:

- Custom histogram calculation
- Manual implementation of filters
- Custom edge detection using Sobel operators
- Custom thresholding algorithms
- Manual histogram equalization
