import depthai as dai
import cv2
import numpy as np
from typing import Optional, Tuple


class OAKCamera:
    def __init__(self, fps: int = 30, resolution: Tuple[int, int] = (640, 480), use_rgb: bool = True):
        self.fps = fps
        self.resolution = resolution
        self.use_rgb = use_rgb  # True for RGB camera, False for IR camera
        self.device: Optional[dai.Device] = None
        self.pipeline: Optional[dai.Pipeline] = None
        self.q_rgb: Optional[dai.DataOutputQueue] = None
        
    def setup_pipeline(self) -> dai.Pipeline:
        pipeline = dai.Pipeline()
        
        if self.use_rgb:
            # RGB camera setup (for better hand detection)
            cam_rgb = pipeline.create(dai.node.ColorCamera)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            cam_rgb.setFps(self.fps)
            
            # Image Manipulation for resizing
            manip = pipeline.create(dai.node.ImageManip)
            manip.initialConfig.setResize(*self.resolution)
            manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
            
            # Output
            xout_rgb = pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            cam_rgb.video.link(manip.inputImage)
            manip.out.link(xout_rgb.input)
        else:
            # Mono camera setup (Infrared camera for OAK-D Pro)
            cam_mono = pipeline.create(dai.node.MonoCamera)
            cam_mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            cam_mono.setBoardSocket(dai.CameraBoardSocket.LEFT)  # Use LEFT mono camera
            cam_mono.setFps(self.fps)
            
            # Image Manipulation for resizing and format conversion
            manip = pipeline.create(dai.node.ImageManip)
            manip.initialConfig.setResize(*self.resolution)
            manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)  # Convert to RGB format
            
            # Output
            xout_rgb = pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            cam_mono.out.link(manip.inputImage)
            manip.out.link(xout_rgb.input)
        
        return pipeline
    
    def connect(self) -> bool:
        try:
            # Create pipeline
            self.pipeline = self.setup_pipeline()
            
            # Connect to device and start pipeline
            self.device = dai.Device(self.pipeline)
            
            # Get output queue
            self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            
            camera_type = "RGB" if self.use_rgb else "IR"
            print(f"Connected to OAK-D device: {self.device.getDeviceName()}")
            print(f"Camera mode: {camera_type}")
            print(f"USB Speed: {self.device.getUsbSpeed()}")
            print(f"Device Info: {self.device.getDeviceInfo()}")
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to OAK-D: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        if not self.q_rgb:
            return None
            
        try:
            in_rgb = self.q_rgb.get()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
        except Exception as e:
            print(f"Error getting frame: {e}")
            
        return None
    
    def get_device_info(self) -> dict:
        if not self.device:
            return {}
            
        try:
            device_info = self.device.getDeviceInfo()
            info = {
                "name": self.device.getDeviceName(),
                "mxid": device_info.getMxId(),
                "usb_speed": str(self.device.getUsbSpeed()),
                "platform": str(device_info.getPlatform()),
            }
            
            # Optional fields that may not exist in all versions
            try:
                info["product_name"] = device_info.getProductName()
            except:
                pass
            try:
                info["board_name"] = device_info.getBoardName()
            except:
                pass
                
            return info
        except Exception as e:
            print(f"Error getting device info: {e}")
            return {}
    
    def close(self):
        if self.device:
            self.device.close()
            self.device = None
        self.q_rgb = None
        self.pipeline = None
        print("OAK-D camera disconnected")


def test_camera_connection():
    print("Testing OAK-D camera connection...")
    
    # Test RGB camera first (better for hand detection)
    print("\n--- Testing RGB Camera ---")
    camera = OAKCamera(use_rgb=True)
    
    if not camera.connect():
        print("❌ Failed to connect to RGB camera")
        
        print("\n--- Testing IR Camera ---")
        camera.close()
        camera = OAKCamera(use_rgb=False)
        
        if not camera.connect():
            print("❌ Failed to connect to IR camera")
            return False
        
        print("✅ Successfully connected to IR camera (RGB camera unavailable)")
    else:
        print("✅ Successfully connected to RGB camera")
    
    # Display device info
    device_info = camera.get_device_info()
    print("\nDevice Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Test frame capture
    print("\nTesting frame capture...")
    frame = camera.get_frame()
    
    if frame is not None:
        print(f"✅ Frame captured successfully - Shape: {frame.shape}")
    else:
        print("❌ Failed to capture frame")
        camera.close()
        return False
    
    camera.close()
    return True


if __name__ == "__main__":
    test_camera_connection()