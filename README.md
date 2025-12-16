# OAK Camera Gaze Tracking

Track where you're looking using OAK-D camera with MediaPipe.

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt --break-system-packages

# 2. Test setup
python test_setup.py

# 3. Run tracker
python mediapipe_gaze_tracker.py
```

## Features

- Real-time gaze tracking (25-30 FPS)
- Head pose estimation (yaw, pitch, roll)
- Eye & iris tracking (478 facial landmarks)
- Screen coordinate mapping
- Temporal smoothing
- 9-point calibration system

## How It Works

1. **Face Detection**: MediaPipe detects 478 facial landmarks
2. **Head Pose**: Calculates orientation using PnP algorithm
3. **Gaze Direction**: Tracks iris position in eyes
4. **Screen Mapping**: Combines head + gaze → (x,y) coordinates
5. **Smoothing**: Averages frames for stability

## Output

- Screen coordinates: (x, y) pixels where user looks
- Head pose: (yaw, pitch, roll) in degrees
- Gaze vector: (x, y) normalized direction

## Configuration

Change screen resolution:
```python
tracker.screen_width = 2560
tracker.screen_height = 1440
```

Adjust smoothing:
```python
tracker.gaze_history = deque(maxlen=10)
```

## Integration Example

Send gaze via UDP:
```python
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
message = f"{gaze_x},{gaze_y}"
sock.sendto(message.encode(), ("127.0.0.1", 5005))
```

## Calibration

For better accuracy:
```python
from calibration import GazeCalibration
cal = GazeCalibration(1920, 1080)
cal.run_calibration(your_gaze_callback)
```

## Troubleshooting

**"MediaPipe not found"**
```bash
pip install mediapipe --break-system-packages
```

**"No OAK device"**
- Check USB 3.0 connection
- Try different cable

**Low FPS**
- Close other camera apps
- Reduce resolution

**Inaccurate**
- Improve lighting
- Run calibration
- Keep head still

## Requirements

- OAK-D camera
- Python 3.7+
- USB 3.0 port
- opencv-python, depthai, numpy, mediapipe

## Use Cases

- Hands-free UI control
- Accessibility interfaces
- Attention tracking
- Gaming with eye control
- VR/AR interaction