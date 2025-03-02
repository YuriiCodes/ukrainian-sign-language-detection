# Hand Gesture Recognition using MediaPipe Hands and OpenCV

This project demonstrates real-time hand gesture recognition using [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) and [OpenCV](https://opencv.org/). The script captures live video from a webcam, detects hand landmarks, calculates distances between key points, and overlays recognized gestures (represented by letters) on the video stream.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Future Enhancements](#future-enhancements)

---

## Overview

This code leverages MediaPipe’s hand tracking model to:
- **Detect hand landmarks** in real time from webcam video.
- **Compute distances** between important hand landmarks (e.g., between the thumb tip and index finger tip, and between fingertips and their base points).
- **Recognize hand gestures** based on predefined threshold values for these distances.
- **Display annotations** on the video stream by overlaying letters that represent the detected gestures.

The script has been streamlined to focus solely on real-time webcam input, removing any static image processing logic.

---

## Requirements

Ensure you have the following libraries installed before running the script:

- **OpenCV:** For capturing and processing video.
- **MediaPipe:** For detecting hand landmarks.
- **NumPy:** For numerical calculations.
- **Math:** (Python standard library) for mathematical operations.

Install the required packages using pip:

```bash
pip install opencv-python mediapipe numpy
```

---

## How It Works

1. **Video Capture:**  
   The script uses OpenCV to capture live video from the webcam.

2. **Preprocessing:**  
   Each frame is converted from BGR to RGB, as MediaPipe requires RGB images. For performance optimization, the frame is temporarily marked as non-writeable during processing.

3. **Hand Landmark Detection:**  
   MediaPipe processes each frame to detect hand landmarks. If any hands are detected, the coordinates of key points (such as fingertips, finger bases, and the wrist) are extracted.

4. **Distance Calculations:**  
   The script calculates various distances between the detected landmarks. These distances are then used to infer specific hand gestures based on simple threshold rules.

5. **Gesture Recognition and Annotation:**  
   Depending on the computed distances:
   - Specific letters or symbols (like a soft sign) are overlaid on the frame.
   - The distances are also displayed on the frame for debugging and analysis.

6. **Display:**  
   The annotated frame is flipped horizontally (to create a selfie-view display) and shown in a window titled "MediaPipe Hands". The video stream continues until the user presses the `Esc` key.

---

## Usage

1. **Connect a Webcam:**  
   Ensure that a webcam is connected to your system.

2. **Run the Script:**  
   Execute the Python script. A window displaying the live annotated video will appear.

3. **Exit:**  
   Press the `Esc` key to exit the video stream and close the window.

---

## Code Explanation

- **Imports and Initialization:**  
  The script imports OpenCV, MediaPipe, NumPy, and Math. It then initializes MediaPipe's drawing utilities and the hand detection model with specified parameters (like `min_detection_confidence` and `min_tracking_confidence`).

- **Video Capture and Processing Loop:**  
  The code captures video frames from the webcam. For each frame:
  - It converts the frame to RGB and processes it to detect hand landmarks.
  - It extracts key landmarks and calculates distances (e.g., between the thumb and index finger, or between the little finger tip and its base).
  - Based on these measurements, the script uses simple rules to determine which gesture is being displayed.
  - It overlays text annotations (letters/symbols) onto the frame corresponding to the detected gesture.
  - Finally, it displays the annotated frame in real time.

- **Drawing Functions:**  
  MediaPipe's drawing utilities are used to render hand landmarks and connections, while OpenCV's text drawing functions (`cv2.putText`) overlay the gesture labels.
