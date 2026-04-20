# 3D Gesture Tracking
Control your Blender viewport in real-time using nothing but your hands!

This add-on utilizes your webcam and advanced hand tracking technology (MediaPipe) to translate specific hand gestures into camera movements within the Blender 3D viewport. Instead of clicking buttons, you simply gesture to rotate, zoom, and pan your scene.

### Features
- Intuitive Control: Maps natural hand gestures to complex camera actions (Orbit, Zoom, Pan).
- Real-Time Tracking: Uses MediaPipe for high-accuracy, low-latency hand landmark detection.
- Dynamic Feedback: Provides an (optional) live camera preview showing the hand skeleton and the detected gesture.
- Easy Setup: Includes a dependency management system to automatically guide users through installing required libraries (OpenCV, MediaPipe).
- Blender Integration: Seamlessly integrated as a panel in the Viewport Sidebar.

### Pre-requisites
- Blender: Version 5.0 or newer.
- Python Libraries: opencv, mediapipe, msvc-runtime (auto-installer available)

### Installation
- Download the latest .zip from the releases
- Drag-and-drop the .zip into the Blender viewport
- Click `Ok` on the `Install from Disk` dialog
- Done!

### Usage
- First time usage requires installing dependencies. The tool auto-installs for you with one click (this process may take a minute depending on your system).
- Make sure you selected the correct camera and that it has a big enough FOV.
- Click "START"
- Follow the tooltip to use the gesture tracking!