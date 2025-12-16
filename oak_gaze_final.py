#!/usr/bin/env python3
"""
OAK Gaze Tracker - Based on rgb_hand_detector approach
Uses HandTracker properly to get correct BGR frames
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import time
from collections import deque

# Add current directory to path for HandTracker imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from HandTracker import HandTracker
except ImportError as e:
    print(f"Error: Cannot import HandTracker: {e}")
    print("\nMake sure these files are in the same folder:")
    print("  - HandTracker.py")
    print("  - mediapipe_utils.py")
    print("  - FPS.py")
    print("  - models/ folder")
    sys.exit(1)


class OAKGazeTracker:
    """
    OAK-D Gaze Tracker using HandTracker for proper camera access
    """
    
    def __init__(self):
        import tkinter as tk
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        
        print(f"Screen: {self.screen_width}x{self.screen_height}")
        
        self.gaze_history = deque(maxlen=15)
        
        # Eye landmarks
        self.LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE = [362, 263, 387, 386, 385, 380, 374, 373]
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        
        # Calibration
        self.calibration_mode = False
        self.calibration_data = []
        self.model_x = None
        self.model_y = None
        self.current_calib_idx = 0
        
        # 13-point calibration grid
        margin = 0.08
        self.calib_screen_points = []
        for y in [margin, 1-margin]:
            for x in [margin, 1-margin]:
                self.calib_screen_points.append(
                    (int(self.screen_width * x), int(self.screen_height * y))
                )
        for x in [margin, 0.5, 1-margin]:
            if x != 0.5:
                self.calib_screen_points.append(
                    (int(self.screen_width * x), int(self.screen_height * 0.5))
                )
        for y in [margin, 0.5, 1-margin]:
            if y != 0.5:
                self.calib_screen_points.append(
                    (int(self.screen_width * 0.5), int(self.screen_height * y))
                )
        self.calib_screen_points.append(
            (int(self.screen_width * 0.5), int(self.screen_height * 0.5))
        )
        
        self.tracker = None
        self.face_mesh = None
    
    def estimate_head_pose(self, landmarks, img_shape):
        h, w = img_shape[:2]
        model_points = np.array([
            (0.0, 0.0, 0.0), (-30.0, -30.0, -50.0), (30.0, -30.0, -50.0),
            (-20.0, 30.0, -30.0), (20.0, 30.0, -30.0), (0.0, -50.0, -50.0),
            (-50.0, 0.0, -40.0), (50.0, 0.0, -40.0)
        ], dtype=np.float64)
        
        image_points = np.array([
            landmarks[1], landmarks[33], landmarks[263],
            landmarks[61], landmarks[291], landmarks[199],
            landmarks[234], landmarks[454]
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
    
    def estimate_gaze_advanced(self, landmarks):
        left_eye_center = np.mean([landmarks[i] for i in self.LEFT_EYE], axis=0)
        left_eye_width = np.linalg.norm(landmarks[33] - landmarks[133])
        left_iris = np.mean([landmarks[i] for i in self.LEFT_IRIS], axis=0) if len(landmarks) > 468 else left_eye_center
        
        right_eye_center = np.mean([landmarks[i] for i in self.RIGHT_EYE], axis=0)
        right_eye_width = np.linalg.norm(landmarks[362] - landmarks[263])
        right_iris = np.mean([landmarks[i] for i in self.RIGHT_IRIS], axis=0) if len(landmarks) > 468 else right_eye_center
        
        left_gaze = (left_iris - left_eye_center) / (left_eye_width / 2)
        right_gaze = (right_iris - right_eye_center) / (right_eye_width / 2)
        avg_gaze = (left_gaze + right_gaze) / 2
        
        return avg_gaze, left_gaze, right_gaze
    
    def extract_features(self, head_pose, gaze_data):
        yaw, pitch, roll = head_pose
        avg_gaze, left_gaze, right_gaze = gaze_data
        
        return np.array([
            yaw, pitch, roll,
            avg_gaze[0], avg_gaze[1],
            left_gaze[0], left_gaze[1],
            right_gaze[0], right_gaze[1]
        ])
    
    def calculate_screen_point(self, features):
        if self.model_x is None or self.model_y is None:
            yaw, pitch = features[0], features[1]
            avg_gaze_x, avg_gaze_y = features[3], features[4]
            
            total_yaw = yaw + avg_gaze_x * 25
            total_pitch = pitch + avg_gaze_y * 25
            
            x_norm = np.clip((total_yaw + 45) / 90, 0, 1)
            y_norm = np.clip((total_pitch + 35) / 70, 0, 1)
            
            return int(x_norm * self.screen_width), int(y_norm * self.screen_height)
        
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2)
        features_poly = poly.fit_transform([features])
        
        screen_x = self.model_x.predict(features_poly)[0]
        screen_y = self.model_y.predict(features_poly)[0]
        
        return (int(np.clip(screen_x, 0, self.screen_width)), 
                int(np.clip(screen_y, 0, self.screen_height)))
    
    def smooth_point_advanced(self, point):
        self.gaze_history.append(point)
        if len(self.gaze_history) < 5:
            return point
        
        points = np.array(list(self.gaze_history))
        median = np.median(points, axis=0)
        distances = np.linalg.norm(points - median, axis=1)
        threshold = np.percentile(distances, 80)
        filtered = points[distances <= threshold]
        
        if len(filtered) == 0:
            return point
        
        weights = np.linspace(0.5, 1.0, len(filtered))
        weights = weights / weights.sum()
        smoothed = np.average(filtered, axis=0, weights=weights)
        
        return int(smoothed[0]), int(smoothed[1])
    
    def compute_calibration(self):
        if len(self.calibration_data) < 9:
            print("✗ Not enough calibration data!")
            return False
        
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        
        X, Y_x, Y_y = [], [], []
        
        for sample in self.calibration_data:
            X.append(sample['features'])
            screen_x, screen_y = sample['screen_point']
            Y_x.append(screen_x)
            Y_y.append(screen_y)
        
        X = np.array(X)
        Y_x = np.array(Y_x)
        Y_y = np.array(Y_y)
        
        try:
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            self.model_x = Ridge(alpha=1.0)
            self.model_y = Ridge(alpha=1.0)
            
            self.model_x.fit(X_poly, Y_x)
            self.model_y.fit(X_poly, Y_y)
            
            print(f"✓ Calibration complete! {len(self.calibration_data)} points")
            return True
        except Exception as e:
            print(f"✗ Calibration failed: {e}")
            return False
    
    def run_calibration(self):
        calib_win = "OAK Calibration"
        cv2.namedWindow(calib_win, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(calib_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("\n" + "="*60)
        print("OAK-D CALIBRATION")
        print("="*60)
        print("13 points for best accuracy")
        print("Press SPACE when looking at circle")
        print("ESC to cancel")
        print("="*60 + "\n")
        
        self.calibration_data = []
        self.current_calib_idx = 0
        waiting_for_space = True
        samples = []
        
        while self.calibration_mode and self.current_calib_idx < len(self.calib_screen_points):
            # Use HandTracker's next_frame() - this gives proper BGR!
            frame, hands, bag = self.tracker.next_frame()
            if frame is None:
                continue
            
            canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            face_detected = False
            if results.multi_face_landmarks:
                landmarks = np.array([[int(lm.x * w), int(lm.y * h)]
                                     for lm in results.multi_face_landmarks[0].landmark])
                
                head_pose = self.estimate_head_pose(landmarks, frame.shape)
                if head_pose is not None:
                    gaze_data = self.estimate_gaze_advanced(landmarks)
                    features = self.extract_features(head_pose, gaze_data)
                    face_detected = True
                    
                    if not waiting_for_space and len(samples) < 40:
                        samples.append(features)
            
            target = self.calib_screen_points[self.current_calib_idx]
            
            color = (0, 0, 255) if waiting_for_space else (0, 255, 0)
            status = "Press SPACE" if waiting_for_space else f"Collecting {len(samples)}/40"
            
            cv2.circle(canvas, target, 50, color, 5)
            cv2.circle(canvas, target, 12, color, -1)
            cv2.line(canvas, (target[0]-60, target[1]), (target[0]+60, target[1]), color, 4)
            cv2.line(canvas, (target[0], target[1]-60), (target[0], target[1]+60), color, 4)
            
            cv2.putText(canvas, status, (target[0]-100, target[1]-70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            progress = f"Point {self.current_calib_idx + 1} / {len(self.calib_screen_points)}"
            cv2.putText(canvas, progress, (50, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            face_status = "OAK-D: OK" if face_detected else "Face: NOT DETECTED"
            face_color = (0, 255, 0) if face_detected else (0, 0, 255)
            cv2.putText(canvas, face_status, (self.screen_width - 300, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, face_color, 3)
            
            preview_h, preview_w = 240, 320
            preview = cv2.resize(frame, (preview_w, preview_h))
            canvas[self.screen_height-preview_h-20:self.screen_height-20, 
                   self.screen_width-preview_w-20:self.screen_width-20] = preview
            
            cv2.imshow(calib_win, canvas)
            
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                print("Calibration cancelled")
                self.calibration_mode = False
                break
            elif key == 32:
                if waiting_for_space and face_detected:
                    waiting_for_space = False
                    samples = []
            
            if not waiting_for_space and len(samples) >= 40:
                avg_features = np.mean(samples, axis=0)
                
                self.calibration_data.append({
                    'features': avg_features,
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
        print("OAK-D GAZE TRACKER")
        print("=" * 60)
        print("Using HandTracker for proper RGB output")
        print("\nControls:")
        print("  c - Calibrate")
        print("  r - Reset")
        print("  q - Quit")
        print("=" * 60)
        
        try:
            import mediapipe as mp
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.6, min_tracking_confidence=0.6
            )
            print("\n✓ MediaPipe initialized")
        except ImportError:
            print("\n✗ MediaPipe not found!")
            return
        
        print("✓ Initializing OAK-D via HandTracker...")
        
        # Initialize HandTracker exactly like rgb_hand_detector does
        self.tracker = HandTracker(
            input_src="rgb",
            use_lm=False,  # Don't need hand landmarks for gaze
            internal_fps=30,
            resolution="full"
        )
        
        print("✓ OAK-D ready!\n")
        
        fps_count, fps_time = 0, time.time()
        
        while True:
            # Use HandTracker's next_frame() - gives proper BGR frames!
            frame, hands, bag = self.tracker.next_frame()
            if frame is None:
                continue
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = np.array([[int(lm.x * w), int(lm.y * h)]
                                     for lm in results.multi_face_landmarks[0].landmark])
                
                head_pose = self.estimate_head_pose(landmarks, frame.shape)
                if head_pose is not None:
                    gaze_data = self.estimate_gaze_advanced(landmarks)
                    features = self.extract_features(head_pose, gaze_data)
                    screen_pt = self.smooth_point_advanced(
                        self.calculate_screen_point(features)
                    )
                    
                    # Draw
                    for idx in self.LEFT_EYE + self.RIGHT_EYE:
                        cv2.circle(frame, tuple(landmarks[idx]), 2, (255, 255, 0), -1)
                    
                    if len(landmarks) > 468:
                        for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
                            cv2.circle(frame, tuple(landmarks[idx]), 4, (0, 255, 255), -1)
                    
                    # Gaze point
                    gx = int(screen_pt[0] / self.screen_width * w)
                    gy = int(screen_pt[1] / self.screen_height * h)
                    cv2.circle(frame, (gx, gy), 15, (0, 0, 255), -1)
                    cv2.circle(frame, (gx, gy), 20, (255, 255, 255), 2)
                    
                    cv2.putText(frame, f"OAK-D: ({screen_pt[0]}, {screen_pt[1]})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    if self.model_x is not None:
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
            
            cv2.imshow("OAK-D Gaze Tracker", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and not self.calibration_mode:
                self.calibration_mode = True
                self.run_calibration()
                self.calibration_mode = False
            elif key == ord('r'):
                print("Reset calibration")
                self.model_x = None
                self.model_y = None
                self.calibration_data = []
                self.gaze_history.clear()
        
        self.tracker.exit()
        cv2.destroyAllWindows()
        print("\nStopped!")


if __name__ == "__main__":
    OAKGazeTracker().run()
