"""
Simple Gaze Tracker - Webcam Version
Test MediaPipe Face Mesh with regular webcam first
"""

import cv2
import numpy as np
import time
from collections import deque


class WebcamGazeTracker:
    def __init__(self):
        self.screen_width = 1920
        self.screen_height = 1080
        self.gaze_history = deque(maxlen=5)
        
        # Eye landmark indices
        self.LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE = [362, 263, 387, 386, 385, 380, 374, 373]
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
    
    def run(self):
        print("=" * 60)
        print("Webcam Gaze Tracker - MediaPipe Test")
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
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Cannot open webcam")
            return
        
        print("\n✓ Webcam opened!")
        print("Press 'q' to quit\n")
        
        fps_count = 0
        fps_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            
            # MediaPipe processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                
                # Draw landmarks
                for lm in face.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Draw eyes
                for idx in self.LEFT_EYE + self.RIGHT_EYE:
                    lm = face.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
                
                # Draw iris if available
                if len(face.landmark) > 468:
                    for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
                        lm = face.landmark[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                
                cv2.putText(frame, "FACE DETECTED!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # FPS
            fps_count += 1
            fps = fps_count / (time.time() - fps_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Webcam Gaze Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nTest complete!")


if __name__ == "__main__":
    tracker = WebcamGazeTracker()
    tracker.run()
