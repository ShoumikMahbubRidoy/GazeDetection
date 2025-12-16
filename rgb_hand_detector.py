#!/usr/bin/env python3
"""
RGB Hand Detector - FIXED VERSION
Works with existing HandTracker + HandTrackerRenderer
"""
import sys
from pathlib import Path
import cv2
import numpy as np

# CRITICAL FIX: Add utils directory to path BEFORE importing HandTracker
# This allows HandTracker.py to find mediapipe_utils and FPS
utils_path = Path(__file__).parent.parent / "utils"
detection_path = Path(__file__).parent
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))
if str(detection_path) not in sys.path:
    sys.path.insert(0, str(detection_path))

# Now import HandTracker (it can now find mediapipe_utils)
try:
    from HandTracker import HandTracker
    from HandTrackerRenderer import HandTrackerRenderer
except ImportError as e:
    print(f"Error: Cannot import HandTracker: {e}")
    print(f"Looking in: {detection_path}")
    print("\nMake sure HandTracker.py and HandTrackerRenderer.py are in:")
    print(f"  {detection_path}")
    sys.exit(1)

# Import FPS counter
try:
    from FPS import FPS
except ImportError:
    # Create simple fallback
    import time
    class FPS:
        def __init__(self):
            self.start_time = time.time()
            self.frame_count = 0
        def update(self):
            self.frame_count += 1
        def get_global(self):
            elapsed = time.time() - self.start_time
            return self.frame_count / elapsed if elapsed > 0 else 0


class RGBHandDetector:
    """
    RGB-based hand detector using Nakakawa-san's HandTracker
    Compatible with all existing app features
    
    This class wraps HandTracker to provide the same interface as the IR-based HandDetector,
    allowing easy switching between IR and RGB modes.
    """
    
    def __init__(self, 
                 fps: int = 30,
                 resolution=(640, 480),
                 pd_score_thresh: float = 0.15,
                 use_gesture: bool = True):
        """
        Initialize RGB hand detector
        
        Args:
            fps: Target FPS (will be clamped to 30 max for OAK-D camera)
            resolution: Output resolution (width, height)
            pd_score_thresh: Palm detection threshold (lower = more sensitive)
            use_gesture: Enable gesture recognition
        """
        self.fps_target = min(fps, 30)  # OAK-D RGB max is ~30fps
        self.resolution = resolution
        self.img_w, self.img_h = resolution
        
        print(f"Initializing RGB Hand Detector...")
        print(f"  Resolution: {resolution[0]}x{resolution[1]}")
        print(f"  Target FPS: {self.fps_target}")
        print(f"  Palm Detection Threshold: {pd_score_thresh}")
        
        # Create HandTracker (uses RGB camera from OAK-D)
        self.tracker = HandTracker(
            input_src="rgb",         # Use RGB camera
            solo=False,              # Detect multiple hands (max 2)
            lm_model="lite",         # Use lite landmark model (faster)
            lm_nb_threads=2,         # Use 2 inference threads
            internal_fps=self.fps_target,
            resolution="full",       # 1920x1080 sensor resolution
            internal_frame_height=resolution[1],
            pd_score_thresh=pd_score_thresh,
            lm_score_thresh=0.6,     # Landmark detection threshold
            single_hand_tolerance_thresh=10,
            use_gesture=use_gesture,
            stats=False              # Disable stats for cleaner output
        )
        
        # Create renderer (optional, for debugging)
        self.renderer = HandTrackerRenderer(tracker=self.tracker)
        
        # FPS counter for compatibility with IR detector interface
        self.fps_counter = FPS()
        
        print(f"✓ RGB Hand Detector initialized")
    
    def connect(self) -> bool:
        """
        Connect to camera
        
        Note: For HandTracker, connection is already established in __init__
        This method is kept for API compatibility with IR detector
        
        Returns:
            True if connected successfully
        """
        return True
    
    def get_frame_and_hands(self):
        """
        Get frame and detected hands
        
        Returns:
            tuple: (frame, hands, depth)
                - frame: RGB image (numpy array)
                - hands: List of detected hand objects with landmarks
                - depth: None (RGB mode has no depth)
        """
        self.fps_counter.update()
        
        # Get frame from tracker
        frame, hands, bag = self.tracker.next_frame()
        
        if frame is None:
            return None, [], None
        
        # Resize frame to target resolution if needed
        if frame.shape[:2][::-1] != self.resolution:
            orig_h, orig_w = frame.shape[:2]
            frame = cv2.resize(frame, self.resolution)
            
            # Scale landmarks to match resized frame
            scale_x = self.resolution[0] / orig_w
            scale_y = self.resolution[1] / orig_h
            for hand in hands:
                if hasattr(hand, 'landmarks') and hand.landmarks is not None:
                    # Convert to float, scale, then convert back
                    hand.landmarks = hand.landmarks.astype(float)
                    hand.landmarks[:, 0] *= scale_x
                    hand.landmarks[:, 1] *= scale_y
        
        # Return in format compatible with IR detector
        # depth is None for RGB mode
        return frame, hands, None
    
    def close(self):
        """Close tracker and renderer"""
        try:
            self.tracker.exit()
        except Exception as e:
            print(f"Error closing tracker: {e}")
        
        try:
            self.renderer.exit()
        except Exception as e:
            print(f"Error closing renderer: {e}")
        
        print("RGB Hand Detector closed.")


def test_rgb_detector():
    """Test the RGB hand detector"""
    print("="*70)
    print("RGB Hand Detector Test")
    print("="*70)
    print("Press 'q' to quit")
    print("="*70)
    
    # Initialize detector
    detector = RGBHandDetector(fps=30, resolution=(1280, 720))
    
    if not detector.connect():
        print("Failed to connect!")
        return
    
    print("✓ Connected! Starting detection...")
    
    frame_count = 0
    
    try:
        while True:
            frame, hands, _ = detector.get_frame_and_hands()
            
            if frame is None:
                continue
            
            frame_count += 1
            
            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Mirror landmarks
            for hand in hands:
                if hasattr(hand, 'landmarks') and hand.landmarks is not None:
                    hand.landmarks[:, 0] = w - hand.landmarks[:, 0]
            
            # Draw info
            cv2.putText(frame, f"Hands detected: {len(hands)}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            fps = detector.fps_counter.get_global()
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw hand skeletons
            for hand in hands:
                if hasattr(hand, 'landmarks') and hand.landmarks is not None:
                    # Draw connections
                    connections = [
                        (0,1),(1,2),(2,3),(3,4),  # Thumb
                        (0,5),(5,6),(6,7),(7,8),  # Index
                        (0,9),(9,10),(10,11),(11,12),  # Middle
                        (0,13),(13,14),(14,15),(15,16),  # Ring
                        (0,17),(17,18),(18,19),(19,20),  # Pinky
                        (5,9),(9,13),(13,17)  # Palm
                    ]
                    
                    for a, b in connections:
                        pt1 = tuple(hand.landmarks[a].astype(int))
                        pt2 = tuple(hand.landmarks[b].astype(int))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                    
                    # Draw points
                    for i, pt in enumerate(hand.landmarks):
                        radius = 6 if i in [0, 4, 8, 12, 16, 20] else 4
                        color = (0, 0, 255) if i in [0, 4, 8, 12, 16, 20] else (255, 255, 255)
                        cv2.circle(frame, tuple(pt.astype(int)), radius, color, -1)
                    
                    # Draw handedness
                    label = "RIGHT" if hand.handedness > 0.5 else "LEFT"
                    wrist = hand.landmarks[0].astype(int)
                    cv2.putText(frame, label, (wrist[0] - 30, wrist[1] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 255, 0) if hand.handedness > 0.5 else (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow("RGB Hand Detector Test", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        detector.close()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")
        print("Done!")


if __name__ == "__main__":
    test_rgb_detector()
