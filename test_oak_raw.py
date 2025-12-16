#!/usr/bin/env python3
"""
Simple test - just show what HandTracker outputs
"""
from rgb_hand_detector import RGBHandDetector
import cv2

print("Initializing detector...")
detector = RGBHandDetector(fps=30, resolution=(1280, 720))
detector.connect()

print("Getting frames... Press 'q' to quit")

while True:
    frame, hands, _ = detector.get_frame_and_hands()
    
    if frame is not None:
        # Show raw frame
        cv2.imshow("Raw OAK Output", frame)
        
        # Print color info
        mean_colors = cv2.mean(frame)[:3]
        print(f"BGR mean: {mean_colors[0]:.1f}, {mean_colors[1]:.1f}, {mean_colors[2]:.1f}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.close()
cv2.destroyAllWindows()
