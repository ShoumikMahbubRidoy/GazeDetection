"""
9-Point Calibration System for Gaze Tracking
"""

import cv2
import numpy as np
import json
from pathlib import Path


class GazeCalibration:
    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_points = self._generate_points()
        self.calibration_data = []
        self.transform_matrix = None
        self.config_file = Path("calibration_config.json")
    
    def _generate_points(self):
        """Generate 9-point grid"""
        points = []
        margin_x = self.screen_width * 0.15
        margin_y = self.screen_height * 0.15
        
        for i in range(3):
            for j in range(3):
                x = margin_x + (self.screen_width - 2*margin_x) * j / 2
                y = margin_y + (self.screen_height - 2*margin_y) * i / 2
                points.append((int(x), int(y)))
        
        return points
    
    def run_calibration(self, gaze_callback):
        """Run calibration procedure"""
        print("\n" + "=" * 60)
        print("9-Point Calibration")
        print("=" * 60)
        print("Look at each red dot and press SPACE\n")
        
        self.calibration_data = []
        
        for idx, point in enumerate(self.calibration_points):
            print(f"Point {idx+1}/9: {point}")
            
            frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            cv2.circle(frame, point, 20, (0, 0, 255), -1)
            cv2.circle(frame, point, 25, (255, 255, 255), 2)
            
            text = f"Point {idx+1}/9 - Look at red dot, press SPACE"
            cv2.putText(frame, text, (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Calibration", frame)
            
            while True:
                key = cv2.waitKey(100)
                
                if key == 27:  # ESC
                    cv2.destroyAllWindows()
                    return False
                
                elif key == 32:  # SPACE
                    samples = []
                    for _ in range(10):
                        data = gaze_callback()
                        if data is not None:
                            samples.append(data)
                        cv2.waitKey(50)
                    
                    if samples:
                        avg = np.mean(samples, axis=0)
                        self.calibration_data.append({
                            'screen_point': point,
                            'gaze_data': avg.tolist()
                        })
                        print(f"  ✓ Captured")
                    break
            
            cv2.destroyAllWindows()
        
        success = self._calculate_transform()
        if success:
            self.save_calibration()
            print("\n✓ Calibration complete!")
        
        return success
    
    def _calculate_transform(self):
        """Calculate transformation matrix"""
        if len(self.calibration_data) < 4:
            return False
        
        screen_pts = np.array([d['screen_point'] for d in self.calibration_data])
        gaze_pts = np.array([d['gaze_data'][:2] for d in self.calibration_data])
        
        try:
            ones = np.ones((len(gaze_pts), 1))
            gaze_hom = np.hstack([gaze_pts, ones])
            self.transform_matrix = np.linalg.lstsq(gaze_hom, screen_pts, rcond=None)[0]
            return True
        except:
            return False
    
    def apply_calibration(self, gaze_data):
        """Apply calibration to raw gaze data"""
        if self.transform_matrix is None:
            if len(gaze_data) >= 2:
                x = np.clip(gaze_data[0], -1, 1)
                y = np.clip(gaze_data[1], -1, 1)
                return int((x + 1) / 2 * self.screen_width), int((y + 1) / 2 * self.screen_height)
            return None
        
        gaze_2d = gaze_data[:2]
        gaze_hom = np.array([gaze_2d[0], gaze_2d[1], 1.0])
        screen_pt = self.transform_matrix.T @ gaze_hom
        
        x = int(np.clip(screen_pt[0], 0, self.screen_width))
        y = int(np.clip(screen_pt[1], 0, self.screen_height))
        
        return x, y
    
    def save_calibration(self):
        """Save to file"""
        data = {
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'calibration_data': self.calibration_data,
            'transform_matrix': self.transform_matrix.tolist() if self.transform_matrix is not None else None
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_calibration(self):
        """Load from file"""
        if not self.config_file.exists():
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            
            self.screen_width = data['screen_width']
            self.screen_height = data['screen_height']
            self.calibration_data = data['calibration_data']
            
            if data['transform_matrix']:
                self.transform_matrix = np.array(data['transform_matrix'])
            
            print(f"✓ Calibration loaded")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False