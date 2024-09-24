# LiDAR-Based Boat Detection and Tracking System

Welcome to the LiDAR-Based Boat Detection and Tracking System! This project provides a comprehensive solution for detecting and tracking boats using LiDAR point cloud data. The system is designed for maritime environments and can be utilized for surveillance, navigation assistance, and research purposes.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Running the System](#running-the-system)
  - [Configuration](#configuration)
- [Modules Overview](#modules-overview)
  - [Publisher](#publisher)
  - [Detector](#detector)
  - [Visualizer](#visualizer)
  - [Detection Methods](#detection-methods)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Real-Time Boat Detection**: Processes LiDAR point cloud data to detect boats in real time.
- **Multiple Detection Algorithms**: Includes various detection methods such as DBSCAN clustering, HDBSCAN, PCA-based shape analysis, and more.
- **Tracking with Kalman Filter**: Implements a Kalman Filter for tracking detected boats over time.
- **Visualization**: Provides a top-down view visualization of detected boats and their tracking information.
- **Configurable Parameters**: Allows users to adjust detection and tracking parameters to suit different environments and requirements.

## Installation

### Prerequisites
Before setting up the system, ensure that you have the following installed on your machine:
- Python 3.7 or later
- ROS 2 Foxy Fitzroy or later (for ROS-based communication)
- pip (Python package installer)

### Setup Instructions

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/lidar-boat-detection.git
    cd lidar-boat-detection
    ```

2. **Install Python Dependencies**    
    NumPy: pip install numpy
    Matplotlib: pip install matplotlib
    FilterPy (for kalman filter): pip install filterpy
    Scikit-learn (For DBSCAN clustering): pip install scikit-learn
    HDBSCAN (For the HDBSCAN clustering): pip install hdbscan
    SciPy: (For the Hungarian algorithm): pip install scipy
    CV2: pip install opencv-python
    ```bash
    pip install numpy
    pip install matplotlib
    pip install filterpy
    pip install scikit-learn
    pip install hdbscan
    pip install scipy
    pip install opencv-python
    ```
    Note: If you encounter any issues with package installations, ensure that your `pip` is up to date:
    ```bash
    pip install --upgrade pip
    ```

4. **Install ROS 2 Packages**
    Ensure that ROS 2 is properly installed and sourced. If not already installed, follow the official ROS 2 
    installation guide: [ROS 2 Installation](https://docs.ros.org/en/humble/Installation.html)

    Source ROS 2 in your terminal:
    ```bash
    source /opt/ros/humble/setup.bash
    ```

## Usage

### Running the System
The system consists of three main modules: Publisher, Detector, and Visualizer. These modules communicate via ROS 2 topics.

1. **Run the Publisher**  
   The publisher simulates or streams LiDAR point cloud data:
   ```bash
   python3 publisher.py
   ```

2. **Run the Detector**  
   The detector processes the point cloud data to detect and track boats:
   ```bash
   python3 detector.py
   ```

3. **Run the Visualizer (Optional)**  
   The visualizer provides an enhanced visualization of the pointclouds:
   ```bash
   python3 visualizer.py
   ```
   Note: Ensure that each script is run in a separate terminal window or terminal multiplexer pane, and that ROS 2 is sourced in each.

### Configuration
You can adjust various parameters in the detector.py script to optimize performance:

- **Detection Method Selection**
    ```python
    self.DETECTION_METHOD = 0  # Change this value (0-5) to select different detection methods
    ```

- **Detection Methods Available:**
    - 0: DBSCAN Clustering
    - 1: HDBSCAN Clustering
    - 2: 2D Clustering after Z-axis Squashing
    - 3: Line-Based Detection with Gap Filling
    - 4: Altitude-Based Planar Clustering
    - 5: PCA-Based Shape Analysis

- **Adjusting Detection Parameters**
    Within each detection method, you can adjust parameters such as eps, min_samples, and thresholds for clustering and filtering.

- **Tracking Parameters**
    ```python
    self.tracking_threshold = 15.0  # Maximum distance to consider the same boat (in meters)
    self.max_disappeared_time = 5.0  # Maximum time (in seconds) to retain a boat without detection
    ```

## Modules Overview

### Publisher
The publisher.py script simulates or streams LiDAR point cloud data and publishes it to the lidar_pointcloud ROS topic.

**Key Features:**
- Simulates LiDAR data for testing purposes.
- Can be adapted to read from actual LiDAR sensors or data files.

### Detector
The detector.py script subscribes to the LiDAR point cloud data, processes it using the selected detection method, and tracks boats using a Kalman Filter.

**Key Features:**
- Multiple detection methods for flexibility.
- Kalman Filter implementation for tracking.
- Visualization of detected boats and their tracking information.

### Visualizer
The visualizer.py script provides an enhanced visualization of the LiDAR point cloud and detection results.

**Key Features:**
- 3D visualization of point cloud data.
- Display of detected boats with bounding boxes and IDs.
- Interactive controls for exploring the data.

## Detection Methods
The system offers several detection methods to cater to different scenarios and data characteristics:

- **DBSCAN Clustering**: Density-based clustering algorithm. Suitable for well-separated clusters.
- **HDBSCAN Clustering**: Hierarchical density-based clustering. Can handle clusters of varying densities.
- **2D Clustering after Z-axis Squashing**: Projects 3D points onto the XY plane. Performs clustering in 2D space.
- **Line-Based Detection with Gap Filling**: Uses image processing techniques to fill gaps. Applies edge detection and RANSAC for line fitting.
- **Altitude-Based Planar Clustering**: Clusters points at different altitude slices. Groups planar clusters to form 3D boat detections.
- **PCA-Based Shape Analysis**: Uses Principal Component Analysis to detect elongated shapes. Suitable for detecting boats based on their elongated structure.

## Troubleshooting
- **No Boats Detected**: Ensure that the LiDAR data contains points representing boats. Adjust clustering parameters (eps, min_samples) to match data characteristics. Try different detection methods to find the most suitable one.
- **Multiple Detections at the Same Location**: Adjust the proximity threshold in the merge_nearby_detections function. Implement Non-Maximum Suppression to eliminate overlapping detections.
- **Tracking Issues (Boats Not Being Tracked Correctly)**: Verify that the Kalman Filter parameters are appropriately set. Ensure that the Mahalanobis distance is being calculated correctly in update_tracks. Increase tracking_threshold if necessary.
- **Visualization Not Displaying Correctly**: Ensure that all required Python packages are installed. Verify that matplotlib is set to use the correct backend (TkAgg). Check for any errors in the console output.

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. **Fork the Repository**: Click the Fork button at the top right of the repository page to create a copy in your GitHub account.
2. **Create a Feature Branch**
    ```bash
    git checkout -b feature/YourFeatureName
    ```
3. **Commit Your Changes**
    ```bash
    git commit -am 'Add some feature'
    ```
4. **Push to the Branch**
    ```bash
    git push origin feature/YourFeatureName
    ```
5. **Submit a Pull Request**: Open a pull request to the main repository with a detailed description of your changes.


**Disclaimer**: This system is intended for research and educational purposes. Use it responsibly and ensure compliance with all applicable laws and regulations when deploying in real-world scenarios.


