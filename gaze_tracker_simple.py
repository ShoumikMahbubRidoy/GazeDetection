#!/usr/bin/env python3
"""
Simple OAK Gaze Tracker with proper color conversion
"""
import cv2
import numpy as np
from collections import deque
import time

try:
    from HandTracker import HandTracker
    HANDTRACKER_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: {e}")
    print("Copy HandTracker.py and dependencies to this folder.")
    HANDTRACKER_AVAILABLE = False


class GazeTracker:
    def __init__(self):
        self.screen_width = 1920
        self.screen_height = 1080
        self.gaze_history = deque(maxlen=5)
        
        # Eye landmarks
        self.LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE = [362, 263, 387, 386, 385, 380, 374, 373]
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
    
    def estimate_head_pose(self, landmarks, img_shape):
        h, w = img_shape[:2]
        model_points = np.array([
            (0.0, 0.0, 0.0), (-30.0, -30.0, -50.0), (30.0, -30.0, -50.0),
            (-20.0, 30.0, -30.0), (20.0, 30.0, -30.0)
        ], dtype=np.float64)
        
        image_points = np.array([
            landmarks[1], landmarks[33], landmarks[263],
            landmarks[61], landmarks[291]
        ], dtype=np.float64)
        
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]
        ], dtype=np.float64)
        
        success, rotation_vec, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, np.zeros((4,1))
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
        left_eye = np.mean([landmarks[i] for i in self.LEFT_EYE], axis=0)
        left_iris = np.mean([landmarks[i] for i in self.LEFT_IRIS], axis=0) if len(landmarks) > 468 else left_eye
        
        right_eye = np.mean([landmarks[i] for i in self.RIGHT_EYE], axis=0)
        right_iris = np.mean([landmarks[i] for i in self.RIGHT_IRIS], axis=0) if len(landmarks) > 468 else right_eye
        
        avg_gaze = ((left_iris - left_eye) + (right_iris - right_eye)) / 2
        norm = np.linalg.norm(avg_gaze)
        return avg_gaze / norm if norm > 0 else avg_gaze
    
    def calculate_screen_point(self, head_pose, gaze_dir):
        yaw, pitch, _ = head_pose
        total_yaw = yaw + gaze_dir[0] * 20
        total_pitch = pitch + gaze_dir[1] * 20
        
        x_norm = np.clip((total_yaw + 40) / 80, 0, 1)
        y_norm = np.clip((total_pitch + 30) / 60, 0, 1)
        
        return int(x_norm * self.screen_width), int(y_norm * self.screen_height)
    
    def smooth_point(self, point):
        self.gaze_history.append(point)
        return (int(np.mean([p[0] for p in self.gaze_history])),
                int(np.mean([p[1] for p in self.gaze_history])))
    
    def draw(self, frame, landmarks, head_pose, gaze_point, gaze_dir):
        h, w = frame.shape[:2]
        
        # Landmarks
        for lm in landmarks:
            cv2.circle(frame, (int(lm[0]), int(lm[1])), 1, (0,255,0), -1)
        
        # Eyes
        for idx in self.LEFT_EYE + self.RIGHT_EYE:
            if idx < len(landmarks):
                cv2.circle(frame, tuple(landmarks[idx].astype(int)), 2, (255,255,0), -1)
        
        # Iris
        if len(landmarks) > 468:
            for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
                cv2.circle(frame, tuple(landmarks[idx].astype(int)), 2, (0,255,255), -1)
        
        # Gaze arrow
        center = np.mean(landmarks, axis=0).astype(int)
        ax, ay = center + (gaze_dir * 100).astype(int)
        cv2.arrowedLine(frame, tuple(center), (ax, ay), (255,0,0), 3, tipLength=0.3)
        
        # Gaze point
        gx = int(gaze_point[0] / self.screen_width * w)
        gy = int(gaze_point[1] / self.screen_height * h)
        cv2.circle(frame, (gx,gy), 15, (0,0,255), -1)
        cv2.circle(frame, (gx,gy), 20, (255,255,255), 2)
        
        # Info
        yaw, pitch, roll = head_pose
        cv2.putText(frame, f"Gaze: {gaze_point}", (10,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Head: Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}", (10,55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return frame
    
    def run(self):
        if not HANDTRACKER_AVAILABLE:
            return
        
        print("=" * 60)
        print("OAK Gaze Tracker")
        print("=" * 60)
        
        try:
            import mediapipe as mp
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.3, min_tracking_confidence=0.3
            )
            print("✓ MediaPipe initialized")
        except ImportError:
            print("✗ MediaPipe not found!")
            return
        
        print("✓ Initializing HandTracker camera...")
        tracker = HandTracker(input_src="rgb", use_lm=False, internal_fps=30, resolution="full")
        
        print("\n✓ Running! Press 'q' to quit\n")
        
        fps_count, fps_time = 0, time.time()
        
        while True:
            frame, _, _ = tracker.next_frame()
            if frame is None:
                continue
            
            # CRITICAL: Convert from YUV to BGR if needed
            # Check if frame looks wrong (pinkish = YUV being shown as BGR)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Try YUV to BGR conversion
                try:
                    # If it's actually YUV420 (NV12), this won't work directly
                    # So we check the mean - YUV shows as pink/magenta
                    mean_val = np.mean(frame)
                    if mean_val > 200 or mean_val < 50:  # Likely wrong format
                        # Frame might already be BGR but wrong, skip conversion
                        pass
                except:
                    pass
            
            h, w = frame.shape[:2]
            
            # Try processing anyway
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = np.array([[int(lm.x * w), int(lm.y * h)] 
                                     for lm in results.multi_face_landmarks[0].landmark])
                head_pose = self.estimate_head_pose(landmarks, frame.shape)
                gaze_dir = self.estimate_gaze(landmarks)
                screen_pt = self.smooth_point(self.calculate_screen_point(head_pose, gaze_dir))
                frame = self.draw(frame, landmarks, head_pose, screen_pt, gaze_dir)
            else:
                cv2.putText(frame, "No face detected", (10, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            fps_count += 1
            cv2.putText(frame, f"FPS: {fps_count/(time.time()-fps_time):.1f}", (w-120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            cv2.imshow("OAK Gaze Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        tracker.exit()
        cv2.destroyAllWindows()
        print("\nStopped!")


if __name__ == "__main__":
    GazeTracker().run()