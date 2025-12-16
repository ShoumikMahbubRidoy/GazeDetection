"""
Setup Verification Script
Compatible with DepthAI 3.2.1 - Using HandTracker API style
"""

import sys


def check_dependencies():
    """Check if packages are installed"""
    print("=" * 60)
    print("Checking Dependencies")
    print("=" * 60)
    
    packages = {
        'opencv-python': 'cv2',
        'depthai': 'depthai',
        'numpy': 'numpy',
        'mediapipe': 'mediapipe'
    }
    
    all_ok = True
    
    for pkg, import_name in packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {pkg:20s} v{version}")
        except ImportError:
            print(f"✗ {pkg:20s} NOT INSTALLED")
            all_ok = False
    
    print("=" * 60)
    
    if not all_ok:
        print("\nInstall: pip install opencv-python depthai numpy mediapipe")
        return False
    
    print("✓ All dependencies installed!\n")
    return True


def check_camera():
    """Check OAK camera"""
    print("=" * 60)
    print("Checking OAK Camera")
    print("=" * 60)
    
    try:
        import depthai as dai
        
        devices = dai.Device.getAllAvailableDevices()
        
        if not devices:
            print("✗ No OAK camera found")
            return False
        
        print(f"✓ Found {len(devices)} device(s)")
        for i, dev in enumerate(devices):
            print(f"  Device {i+1}: {dev.name}")
        
        # Test connection - HandTracker style
        pipeline = dai.Pipeline()
        
        cam = pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setVideoSize(640, 480)
        cam.setFps(30)
        
        xout = pipeline.createXLinkOut()
        xout.setStreamName("test")
        cam.video.link(xout.input)
        
        with dai.Device(pipeline) as device:
            q = device.getOutputQueue("test", maxSize=1, blocking=False)
            frame = q.get().getCvFrame()
            
            if frame is not None:
                print(f"✓ Camera working! {frame.shape[1]}x{frame.shape[0]}")
                print("=" * 60)
                return True
            
        return False
                
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_test():
    """Run quick camera test"""
    print("\n" + "=" * 60)
    print("Camera Test (5 seconds)")
    print("=" * 60)
    
    try:
        import depthai as dai
        import cv2
        import time
        
        pipeline = dai.Pipeline()
        
        cam = pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setVideoSize(640, 480)
        cam.setFps(30)
        
        xout = pipeline.createXLinkOut()
        xout.setStreamName("preview")
        cam.video.link(xout.input)
        
        with dai.Device(pipeline) as device:
            q = device.getOutputQueue("preview", maxSize=4, blocking=False)
            
            start = time.time()
            
            while time.time() - start < 5:
                frame = q.get().getCvFrame()
                
                cv2.putText(frame, "Camera Test - COLOR", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                time_left = int(5 - (time.time() - start))
                cv2.putText(frame, f"{time_left}s", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Test", frame)
                
                if cv2.waitKey(1) == ord('q'):
                    break
        
        cv2.destroyAllWindows()
        print("✓ Test complete!\n")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("OAK Gaze Tracker - Setup Verification")
    print("=" * 60)
    print()
    
    deps_ok = check_dependencies()
    if not deps_ok:
        return
    
    cam_ok = check_camera()
    if not cam_ok:
        return
    
    print("\n✓ Setup complete! Ready to run gaze tracker.\n")
    
    response = input("Run camera test? (y/n): ")
    if response.lower() == 'y':
        run_test()
    
    print("\nNext: python mediapipe_gaze_tracker.py\n")


if __name__ == "__main__":
    main()