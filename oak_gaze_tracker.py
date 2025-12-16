"""
OAK Gaze Tracker - Using ISP output for proper RGB frames
"""

import cv2
import numpy as np
import depthai as dai
import time
from collections import deque


class OAKGazeTracker:
    def __init__(self):
        self.screen_width = 1920
        self.screen_height = 1080
        self.gaze_history = deque(maxlen=5)
        
        # Eye landmark indices
        self.LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE = [362, 263, 387, 386, 385, 380, 374, 373]
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        
    def create_pipeline(self):
        """Create pipeline using ISP output"""
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)
        
        # ColorCamera
        cam = pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setInterleaved(False)
        cam.setFps(30)
        
        # Set both video and preview size
        cam.setVideoSize(720, 720)
        cam.setPreviewSize(720, 720)
        cam.setIspScale(2, 3)
        
        # Use ISP output (RGB) instead of video (NV12/YUV)
        xout = pipeline.createXLinkOut()
        xout.setStreamName("isp")
        xout.input.setQueueSize(1)
        xout.input.setBlocking(False)
        cam.isp.link(xout.input)
        
        return pipeline
    
    def estimate_head_pose(self, landmarks, img_shape):
        """Calculate head orientation"""
        h, w = img_shape[:2]
        
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (-30.0, -30.0, -50.0),
            (30.0, -30.0, -50.0),
            (-20.0, 30.0, -30.0),
            (20.0, 30.0, -30.0)
        ], dtype=np.float64)
        
        image_points = np.array([
            landmarks[1], landmarks[33], landmarks[263],
            landmarks[61], landmarks[291]
        ], dtype=np.float64)
        
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4,1))
        success, rotation_vec, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
        
        if not success:
            return (0, 0, 0)
        
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        sy = np.sqrt(rotation_mat[0,0]**2 + rotation_mat[1,0]**2)
        
        if sy > 1e-6:
            pitch = np.arctan2(rotation_mat[2,1], rotation_mat[2,2])
            yaw = np.arctan2(-rotation_mat[2,0], sy)
            roll = np.arctan2(rotation_mat[1,0], rotation_mat[0,0])
        else:
            pitch = np.arctan2(-rotation_mat[1,2], rotation_mat[1,1])
            yaw = np.arctan2(-rotation_mat[2,0], sy)
            roll = 0
        
        return (np.degrees(yaw), np.degrees(pitch), np.degrees(roll))
    
    def estimate_gaze(self, landmarks):
        """Estimate gaze direction"""
        left_eye = np.mean([landmarks[i] for i in self.LEFT_EYE], axis=0)
        if len(landmarks) > 468:
            left_iris = np.mean([landmarks[i] for i in self.LEFT_IRIS], axis=0)
        else:
            left_iris = left_eye
        
        right_eye = np.mean([landmarks[i] for i in self.RIGHT_EYE], axis=0)
        if len(landmarks) > 468:
            right_iris = np.mean([landmarks[i] for i in self.RIGHT_IRIS], axis=0)
        else:
            right_iris = right_eye
        
        left_gaze = left_iris - left_eye
        right_gaze = right_iris - right_eye
        avg_gaze = (left_gaze + right_gaze) / 2
        
        norm = np.linalg.norm(avg_gaze)
        if norm > 0:
            avg_gaze = avg_gaze / norm
        
        return avg_gaze
    
    def calculate_screen_point(self, head_pose, gaze_dir):
        """Map to screen coordinates"""
        yaw, pitch, roll = head_pose
        gaze_x, gaze_y = gaze_dir
        
        total_yaw = yaw + gaze_x * 20
        total_pitch = pitch + gaze_y * 20
        
        x_norm = np.clip((total_yaw + 40) / 80, 0, 1)
        y_norm = np.clip((total_pitch + 30) / 60, 0, 1)
        
        screen_x = int(x_norm * self.screen_width)
        screen_y = int(y_norm * self.screen_height)
        
        return screen_x, screen_y
    
    def smooth_point(self, point):
        """Apply smoothing"""
        self.gaze_history.append(point)
        avg_x = int(np.mean([p[0] for p in self.gaze_history]))
        avg_y = int(np.mean([p[1] for p in self.gaze_history]))
        return avg_x, avg_y
    
    def draw_info(self, frame, landmarks, head_pose, gaze_point, gaze_dir):
        """Draw visualization"""
        h, w = frame.shape[:2]
        
        # Face landmarks
        for lm in landmarks:
            cv2.circle(frame, (int(lm[0]), int(lm[1])), 1, (0,255,0), -1)
        
        # Eyes
        for idx in self.LEFT_EYE + self.RIGHT_EYE:
            if idx < len(landmarks):
                pt = landmarks[idx]
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255,255,0), -1)
        
        # Iris
        if len(landmarks) > 468:
            for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
                pt = landmarks[idx]
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0,255,255), -1)
        
        # Gaze arrow
        center = np.mean(landmarks, axis=0)
        cx, cy = int(center[0]), int(center[1])
        ax = int(cx + gaze_dir[0] * 100)
        ay = int(cy + gaze_dir[1] * 100)
        cv2.arrowedLine(frame, (cx,cy), (ax,ay), (255,0,0), 3, tipLength=0.3)
        
        # Gaze point
        gx = int(gaze_point[0] / self.screen_width * w)
        gy = int(gaze_point[1] / self.screen_height * h)
        cv2.circle(frame, (gx,gy), 15, (0,0,255), -1)
        cv2.circle(frame, (gx,gy), 20, (255,255,255), 2)
        
        # Text
        yaw, pitch, roll = head_pose
        cv2.putText(frame, f"Gaze: ({gaze_point[0]}, {gaze_point[1]})", 
                   (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Head: Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}",
                   (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        return frame
    
    def run(self):
        """Main loop"""
        print("=" * 60)
        print("OAK Gaze Tracker with MediaPipe")
        print("=" * 60)
        
        try:
            import mediapipe as mp
            mp_face = mp.solutions.face_mesh
            face_mesh = mp_face.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            print("✓ MediaPipe initialized")
        except ImportError:
            print("✗ MediaPipe not found!")
            return
        
        # Find device
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print("✗ No OAK device found!")
            return
        
        print(f"✓ Found OAK device: {devices[0].name}")
        
        pipeline = self.create_pipeline()
        
        print("Starting pipeline...")
        with dai.Device(pipeline) as device:
            print(f"✓ Pipeline started - USB: {str(device.getUsbSpeed()).split('.')[-1]}")
            
            queue = device.getOutputQueue("isp", maxSize=1, blocking=False)
            
            print("\n✓ Gaze Tracker Running!")
            print("Look at the camera...")
            print("Press 'q' to quit, 'r' to reset\n")
            
            fps_count = 0
            fps_time = time.time()
            
            while True:
                in_frame = queue.get()
                if in_frame is None:
                    continue
                    
                frame = in_frame.getCvFrame()
                if frame is None or frame.size == 0:
                    continue
                
                h, w = frame.shape[:2]
                
                # Convert from RGB to BGR (ISP outputs RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # MediaPipe expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                
                if results.multi_face_landmarks:
                    face = results.multi_face_landmarks[0]
                    
                    # Convert landmarks
                    landmarks = []
                    for lm in face.landmark:
                        landmarks.append([int(lm.x * w), int(lm.y * h)])
                    landmarks = np.array(landmarks)
                    
                    # Calculate gaze
                    head_pose = self.estimate_head_pose(landmarks, frame.shape)
                    gaze_dir = self.estimate_gaze(landmarks)
                    screen_pt = self.calculate_screen_point(head_pose, gaze_dir)
                    screen_pt = self.smooth_point(screen_pt)
                    
                    # Draw
                    frame = self.draw_info(frame, landmarks, head_pose, 
                                          screen_pt, gaze_dir)
                else:
                    cv2.putText(frame, "No face detected", (10, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                # FPS
                fps_count += 1
                fps = fps_count / (time.time() - fps_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                cv2.imshow("OAK Gaze Tracker", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.gaze_history.clear()
        
        cv2.destroyAllWindows()
        print("\nTracker stopped!")


if __name__ == "__main__":
    tracker = OAKGazeTracker()
    tracker.run()
