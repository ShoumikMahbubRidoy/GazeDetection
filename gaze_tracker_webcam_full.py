#!/usr/bin/env python3
"""
Fullscreen Calibrated Gaze Tracker with Camera Preview
Shows small camera preview during calibration
"""
import cv2
import numpy as np
import time
from collections import deque


class FullscreenGazeTracker:
    def __init__(self):
        # Get actual screen resolution
        import tkinter as tk
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        
        self.gaze_history = deque(maxlen=10)
        
        # Eye landmarks
        self.LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE = [362, 263, 387, 386, 385, 380, 374, 373]
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        
        # Calibration
        self.calibration_mode = False
        self.calibration_data = []
        self.calibration_transform = None
        self.current_calib_idx = 0
        
        # Generate 9-point calibration grid
        margin = 0.1
        self.calib_screen_points = [
            (int(self.screen_width * x), int(self.screen_height * y))
            for y in [margin, 0.5, 1-margin]
            for x in [margin, 0.5, 1-margin]
        ]
        
        # Camera and MediaPipe
        self.cap = None
        self.face_mesh = None
    
    def estimate_head_pose(self, landmarks, img_shape):
        h, w = img_shape[:2]
        model_points = np.array([
            (0.0, 0.0, 0.0), (-30.0, -30.0, -50.0), (30.0, -30.0, -50.0),
            (-20.0, 30.0, -30.0), (20.0, 30.0, -30.0), (0.0, -50.0, -50.0)
        ], dtype=np.float64)
        
        image_points = np.array([
            landmarks[1], landmarks[33], landmarks[263],
            landmarks[61], landmarks[291], landmarks[199]
        ], dtype=np.float64)
        
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]
        ], dtype=np.float64)
        
        success, rotation_vec, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, np.zeros((4,1))
        )
        
        if not success:
            return None
        
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
        left_eye = np.mean([landmarks[i] for i in self.LEFT_EYE], axis=0)
        left_iris = np.mean([landmarks[i] for i in self.LEFT_IRIS], axis=0) if len(landmarks) > 468 else left_eye
        
        right_eye = np.mean([landmarks[i] for i in self.RIGHT_EYE], axis=0)
        right_iris = np.mean([landmarks[i] for i in self.RIGHT_IRIS], axis=0) if len(landmarks) > 468 else right_eye
        
        avg_gaze = ((left_iris - left_eye) + (right_iris - right_eye)) / 2
        norm = np.linalg.norm(avg_gaze)
        return avg_gaze / norm if norm > 0 else avg_gaze
    
    def calculate_screen_point(self, head_pose, gaze_dir):
        yaw, pitch, _ = head_pose
        gaze_x, gaze_y = gaze_dir
        
        if self.calibration_transform is not None:
            features = np.array([yaw, pitch, gaze_x, gaze_y, 1.0])
            screen_x, screen_y = self.calibration_transform @ features
        else:
            total_yaw = yaw + gaze_x * 30
            total_pitch = pitch + gaze_y * 30
            
            x_norm = np.clip((total_yaw + 50) / 100, 0, 1)
            y_norm = np.clip((total_pitch + 40) / 80, 0, 1)
            
            screen_x = x_norm * self.screen_width
            screen_y = y_norm * self.screen_height
        
        return int(np.clip(screen_x, 0, self.screen_width)), int(np.clip(screen_y, 0, self.screen_height))
    
    def smooth_point(self, point):
        self.gaze_history.append(point)
        if len(self.gaze_history) < 3:
            return point
        
        xs = [p[0] for p in self.gaze_history]
        ys = [p[1] for p in self.gaze_history]
        return int(np.median(xs)), int(np.median(ys))
    
    def compute_calibration(self):
        if len(self.calibration_data) < 5:
            print("✗ Not enough calibration data!")
            return False
        
        X, Y = [], []
        
        for sample in self.calibration_data:
            yaw, pitch, _ = sample['head_pose']
            gaze_x, gaze_y = sample['gaze_dir']
            screen_x, screen_y = sample['screen_point']
            
            X.append([yaw, pitch, gaze_x, gaze_y, 1.0])
            Y.append([screen_x, screen_y])
        
        X, Y = np.array(X), np.array(Y)
        
        try:
            self.calibration_transform = np.linalg.lstsq(X, Y, rcond=None)[0].T
            print(f"✓ Calibration computed from {len(self.calibration_data)} points!")
            return True
        except:
            print("✗ Calibration failed!")
            return False
    
    def run_calibration(self):
        """Fullscreen calibration with camera preview"""
        calib_win = "Calibration"
        cv2.namedWindow(calib_win, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(calib_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("\n" + "="*60)
        print("CALIBRATION MODE")
        print("="*60)
        print("1. Look at RED circle")
        print("2. Press SPACE when ready")
        print("3. Keep looking while GREEN (collecting)")
        print("4. Repeat for all 9 points")
        print("ESC to cancel")
        print("="*60 + "\n")
        
        self.calibration_data = []
        self.current_calib_idx = 0
        waiting_for_space = True
        samples = []
        
        while self.calibration_mode and self.current_calib_idx < len(self.calib_screen_points):
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Create fullscreen canvas
            canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            
            # Process face
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            face_detected = False
            if results.multi_face_landmarks:
                landmarks = np.array([[int(lm.x * w), int(lm.y * h)]
                                     for lm in results.multi_face_landmarks[0].landmark])
                
                head_pose = self.estimate_head_pose(landmarks, frame.shape)
                if head_pose is not None:
                    gaze_dir = self.estimate_gaze(landmarks)
                    face_detected = True
                    
                    # Collect samples when not waiting
                    if not waiting_for_space and len(samples) < 30:
                        samples.append((head_pose, gaze_dir))
            
            # Draw calibration target
            target = self.calib_screen_points[self.current_calib_idx]
            
            if waiting_for_space:
                color = (0, 0, 255)  # RED - waiting
                status = "Press SPACE"
            else:
                color = (0, 255, 0)  # GREEN - collecting
                status = f"Collecting {len(samples)}/30"
            
            # Draw target
            cv2.circle(canvas, target, 40, color, 4)
            cv2.circle(canvas, target, 10, color, -1)
            cv2.line(canvas, (target[0]-50, target[1]), (target[0]+50, target[1]), color, 3)
            cv2.line(canvas, (target[0], target[1]-50), (target[0], target[1]+50), color, 3)
            
            # Status text
            cv2.putText(canvas, status, (target[0]-100, target[1]-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Progress
            progress = f"Point {self.current_calib_idx + 1} / {len(self.calib_screen_points)}"
            cv2.putText(canvas, progress, (50, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Face status
            face_status = "Face: OK" if face_detected else "Face: NOT DETECTED"
            face_color = (0, 255, 0) if face_detected else (0, 0, 255)
            cv2.putText(canvas, face_status, (self.screen_width - 300, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, face_color, 3)
            
            # Camera preview (small)
            preview_h, preview_w = 240, 320
            preview = cv2.resize(frame, (preview_w, preview_h))
            canvas[self.screen_height-preview_h-20:self.screen_height-20, 
                   self.screen_width-preview_w-20:self.screen_width-20] = preview
            
            cv2.imshow(calib_win, canvas)
            
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                print("Calibration cancelled")
                self.calibration_mode = False
                break
            elif key == 32:  # SPACE
                if waiting_for_space and face_detected:
                    waiting_for_space = False
                    samples = []
            
            # Check if done collecting
            if not waiting_for_space and len(samples) >= 30:
                # Average samples
                avg_yaw = np.mean([s[0][0] for s in samples])
                avg_pitch = np.mean([s[0][1] for s in samples])
                avg_gaze_x = np.mean([s[1][0] for s in samples])
                avg_gaze_y = np.mean([s[1][1] for s in samples])
                
                self.calibration_data.append({
                    'head_pose': (avg_yaw, avg_pitch, 0),
                    'gaze_dir': (avg_gaze_x, avg_gaze_y),
                    'screen_point': target
                })
                
                print(f"✓ Point {self.current_calib_idx + 1} collected")
                
                self.current_calib_idx += 1
                waiting_for_space = True
                samples = []
        
        cv2.destroyWindow(calib_win)
        
        if self.current_calib_idx >= len(self.calib_screen_points):
            self.compute_calibration()
    
    def run(self):
        print("=" * 60)
        print("Fullscreen Calibrated Gaze Tracker")
        print("=" * 60)
        print(f"Screen: {self.screen_width}x{self.screen_height}")
        print("\nControls:")
        print("  c - Start calibration")
        print("  r - Reset calibration")
        print("  q - Quit")
        print("=" * 60)
        
        try:
            import mediapipe as mp
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            print("\n✓ MediaPipe initialized")
        except ImportError:
            print("\n✗ MediaPipe not found!")
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("✗ Cannot open webcam")
            return
        
        print("✓ Webcam ready\n")
        
        fps_count, fps_time = 0, time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = np.array([[int(lm.x * w), int(lm.y * h)]
                                     for lm in results.multi_face_landmarks[0].landmark])
                
                head_pose = self.estimate_head_pose(landmarks, frame.shape)
                if head_pose is not None:
                    gaze_dir = self.estimate_gaze(landmarks)
                    screen_pt = self.smooth_point(
                        self.calculate_screen_point(head_pose, gaze_dir)
                    )
                    
                    # Draw
                    for i, lm in enumerate(landmarks):
                        if i % 5 == 0:
                            cv2.circle(frame, tuple(lm), 1, (0, 255, 0), -1)
                    
                    for idx in self.LEFT_EYE + self.RIGHT_EYE:
                        cv2.circle(frame, tuple(landmarks[idx]), 2, (255, 255, 0), -1)
                    
                    if len(landmarks) > 468:
                        for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
                            cv2.circle(frame, tuple(landmarks[idx]), 3, (0, 255, 255), -1)
                    
                    # Draw gaze point on frame
                    gx = int(screen_pt[0] / self.screen_width * w)
                    gy = int(screen_pt[1] / self.screen_height * h)
                    cv2.circle(frame, (gx, gy), 12, (0, 0, 255), -1)
                    cv2.circle(frame, (gx, gy), 16, (255, 255, 255), 2)
                    
                    # Gaze arrow from face center
                    center = np.mean(landmarks, axis=0).astype(int)
                    ax, ay = center + (gaze_dir * 80).astype(int)
                    cv2.arrowedLine(frame, tuple(center), (ax, ay), (255, 0, 0), 2, tipLength=0.3)
                    
                    # Info
                    yaw, pitch, _ = head_pose
                    cv2.putText(frame, f"Gaze: ({screen_pt[0]}, {screen_pt[1]})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Head: Y:{yaw:.1f} P:{pitch:.1f}", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    if self.calibration_transform is not None:
                        cv2.putText(frame, "CALIBRATED", (10, h-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Press 'c' to calibrate", (10, h-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
            else:
                cv2.putText(frame, "No face detected", (10, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            fps_count += 1
            fps = fps_count / (time.time() - fps_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Gaze Tracker", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and not self.calibration_mode:
                self.calibration_mode = True
                self.run_calibration()
                self.calibration_mode = False
            elif key == ord('r'):
                print("Reset calibration")
                self.calibration_transform = None
                self.calibration_data = []
                self.gaze_history.clear()
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nStopped!")


if __name__ == "__main__":
    FullscreenGazeTracker().run()