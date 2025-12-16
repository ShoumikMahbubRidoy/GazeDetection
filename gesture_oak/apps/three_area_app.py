#!/usr/bin/env python3
"""
3-Area Detection Application
Detects index finger (ONE) or fist in 3 horizontal screen areas using RGB camera

Screen Layout:
┌─────────┬─────────┬─────────┐
│ Area 1  │ Area 2  │ Area 3  │
│ (Left)  │(Center) │ (Right) │
└─────────┴─────────┴─────────┘

Gestures:
- ONE: Index finger extended ☝
- FIST: All fingers closed ✊

UDP Messages:
- gesture/one: Index finger detected
- gesture/zero: Fist detected
- area/3section/1-3: Area number
- area/3section/0: No hand
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import detectors
# Note: This assumes rgb_hand_detector.py and three_area_detector.py are in gesture_oak/detection/
from gesture_oak.detection.rgb_hand_detector import RGBHandDetector
from gesture_oak.detection.three_area_detector import ThreeAreaDetector, GestureType


def draw_hand_skeleton(img: np.ndarray, hand, color=(0, 255, 0)):
    """
    Draw hand skeleton with landmarks
    
    Args:
        img: Image to draw on
        hand: Hand object with landmarks
        color: RGB color for skeleton
    """
    if not hasattr(hand, 'landmarks') or hand.landmarks is None:
        return
    
    lm = hand.landmarks
    
    # Handle 3D landmarks
    if len(lm.shape) == 2 and lm.shape[1] >= 2:
        lm = lm[:, :2]
    
    # Hand skeleton connections
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17)
    ]
    
    # Draw lines
    for a, b in connections:
        pt1 = tuple(lm[a].astype(int))
        pt2 = tuple(lm[b].astype(int))
        cv2.line(img, pt1, pt2, color, 2)
    
    # Draw landmark points
    for i, pt in enumerate(lm):
        # Larger circles for important points
        if i in [0, 4, 8, 12, 16, 20]:  # Wrist and fingertips
            radius = 7
            point_color = (0, 0, 255)
        else:
            radius = 4
            point_color = (255, 255, 255)
        
        cv2.circle(img, tuple(pt.astype(int)), radius, point_color, -1)
        cv2.circle(img, tuple(pt.astype(int)), radius + 1, (0, 0, 0), 1)


def draw_area_grid(img: np.ndarray, current_area: int, gesture_type: GestureType):
    """
    Draw 3-area grid with current area highlighted
    
    Args:
        img: Image to draw on
        current_area: Currently active area (1-3, or 0 for none)
        gesture_type: Current gesture type
    """
    h, w = img.shape[:2]
    
    # Area colors (BGR format for OpenCV)
    area_colors = {
        1: (0, 255, 255),    # Yellow (left)
        2: (0, 255, 0),      # Green (center)
        3: (0, 0, 255)       # Red (right)
    }
    
    # Calculate area boundaries
    third = w // 3
    
    # Draw vertical dividing lines
    cv2.line(img, (third, 0), (third, h), (255, 255, 255), 3)
    cv2.line(img, (third * 2, 0), (third * 2, h), (255, 255, 255), 3)
    
    # Highlight active area if gesture is detected
    if current_area > 0 and gesture_type != GestureType.NONE:
        overlay = img.copy()
        x_start = (current_area - 1) * third
        x_end = current_area * third
        color = area_colors.get(current_area, (255, 255, 255))
        
        # Fill area with semi-transparent color
        cv2.rectangle(overlay, (x_start, 0), (x_end, h), color, -1)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        
        # Draw thick border around active area
        cv2.rectangle(img, (x_start, 0), (x_end, h), color, 6)
    
    # Draw area numbers
    for i in range(1, 4):
        x_pos = int((i - 0.5) * third)
        y_pos = h // 2
        color = area_colors.get(i, (255, 255, 255))
        
        # Large area number
        cv2.putText(img, str(i), (x_pos - 35, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 3.5, color, 7)
        
        # Area label
        labels = {1: "LEFT", 2: "CENTER", 3: "RIGHT"}
        cv2.putText(img, labels[i], (x_pos - 70, y_pos + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def draw_reference_point(img: np.ndarray, detector: ThreeAreaDetector, color=(255, 0, 255)):
    """
    Draw the reference point used for area calculation
    
    Args:
        img: Image to draw on
        detector: ThreeAreaDetector instance
        color: RGB color for reference point
    """
    info = detector.get_state_info()
    ref_point = info.get('reference_point')
    
    if ref_point is not None:
        pt = (int(ref_point[0]), int(ref_point[1]))
        # Draw crosshair
        cv2.drawMarker(img, pt, color, cv2.MARKER_CROSS, 30, 3)
        # Draw circle
        cv2.circle(img, pt, 15, color, 3)
        
        # Label
        label = "INDEX TIP" if info['gesture'] == "ONE" else "PALM CENTER"
        cv2.putText(img, label, (pt[0] + 20, pt[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    """Main application loop"""
    print("="*70)
    print("3-AREA GESTURE DETECTION (RGB Mode)")
    print("="*70)
    print()
    print("Detection:")
    print("  ONE  ☝: Only index finger extended")
    print("  FIST ✊: All fingers closed")
    print()
    print("Screen Layout:")
    print("  ┌─────────┬─────────┬─────────┐")
    print("  │ Area 1  │ Area 2  │ Area 3  │")
    print("  │ (Yellow)│ (Green) │  (Red)  │")
    print("  └─────────┴─────────┴─────────┘")
    print()
    print("UDP Messages:")
    print("  gesture/one        : Index finger detected")
    print("  gesture/zero       : Fist detected")
    print("  area/3section/1-3  : Area number")
    print("  area/3section/0    : No hand")
    print()
    print("Controls:")
    print("  q : Quit")
    print("  r : Reset detector")
    print("  s : Save screenshot")
    print("  d : Toggle debug overlay")
    print("="*70)
    print()
    
    # Initialize RGB hand detector
    print("Initializing RGB hand detector...")
    hd = RGBHandDetector(
        fps=30,
        resolution=(1280, 720),
        pd_score_thresh=0.5,  # Moderate sensitivity
        use_gesture=False     # We're doing custom gesture detection
    )
    
    if not hd.connect():
        print("❌ Failed to connect to camera!")
        return
    
    # Initialize 3-area detector
    print("Initializing 3-area detector...")
    detector = ThreeAreaDetector(
        udp_ip="192.168.0.10",
        udp_port=9000,
        debug=False
    )
    
    print("✓ All systems ready!")
    print()
    
    saved_count = 0
    show_debug = False
    
    try:
        while True:
            # Get frame and hands
            frame, hands, _ = hd.get_frame_and_hands()
            
            if frame is None:
                continue
            
            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Mirror hand landmarks
            for hand in hands:
                if hasattr(hand, 'landmarks') and hand.landmarks is not None:
                    hand.landmarks[:, 0] = w - hand.landmarks[:, 0]
            
            # Create display frame
            disp = frame.copy()
            
            # Process first detected hand
            if hands:
                hand = hands[0]
                
                # Update detector
                gesture, area = detector.update(hand, frame.shape[:2])
                
                # Draw area grid (behind hand)
                draw_area_grid(disp, area, gesture)
                
                # Draw hand skeleton (on top of grid)
                skeleton_color = (0, 255, 0) if gesture != GestureType.NONE else (128, 128, 128)
                draw_hand_skeleton(disp, hand, color=skeleton_color)
                
                # Draw reference point
                if show_debug:
                    draw_reference_point(disp, detector)
                
                # Info panel (top-left)
                overlay = disp.copy()
                cv2.rectangle(overlay, (10, 10), (500, 160), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, disp, 0.3, 0, disp)
                
                y = 50
                
                # Gesture display
                if gesture == GestureType.ONE:
                    gesture_txt = "GESTURE: ONE ☝"
                    gesture_color = (0, 255, 255)  # Yellow
                elif gesture == GestureType.FIST:
                    gesture_txt = "GESTURE: FIST ✊"
                    gesture_color = (0, 0, 255)    # Red
                else:
                    gesture_txt = "GESTURE: NONE"
                    gesture_color = (128, 128, 128)  # Gray
                
                cv2.putText(disp, gesture_txt, (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, gesture_color, 2)
                y += 55
                
                # Area display
                if area > 0:
                    area_colors = {1: (0, 255, 255), 2: (0, 255, 0), 3: (0, 0, 255)}
                    area_names = {1: "LEFT", 2: "CENTER", 3: "RIGHT"}
                    cv2.putText(disp, f"AREA: {area} ({area_names[area]})", (20, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                               area_colors.get(area, (255, 255, 255)), 2)
                else:
                    cv2.putText(disp, "AREA: -", (20, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
                
                # Large area indicator (bottom-right)
                if area > 0 and gesture != GestureType.NONE:
                    area_colors = {1: (0, 255, 255), 2: (0, 255, 0), 3: (0, 0, 255)}
                    col = area_colors[area]
                    cv2.putText(disp, str(area), (w - 120, h - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 4.0, col, 10)
                    cv2.circle(disp, (w - 70, h - 60), 50, col, 5)
            
            else:
                # No hand detected
                detector.update(None, frame.shape[:2])
                
                # Draw grid without highlight
                draw_area_grid(disp, 0, GestureType.NONE)
                
                # Instructions
                cv2.putText(disp, "SHOW YOUR HAND", (w//2 - 250, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(disp, "Try ONE ☝ or FIST ✊", (w//2 - 200, h//2 + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # FPS counter (top-right)
            fps = hd.fps_counter.get_global()
            cv2.putText(disp, f"FPS: {fps:.1f}", (w - 150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Debug mode indicator
            if show_debug:
                cv2.putText(disp, "DEBUG", (w - 150, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Display frame
            cv2.imshow("3-Area Detection", disp)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                print("\nQuitting...")
                break
            
            elif key == ord('r'):
                # Reset detector
                detector.reset()
                print("Detector reset!")
            
            elif key == ord('s'):
                # Save screenshot
                filename = f"3area_{saved_count:04d}.jpg"
                cv2.imwrite(filename, disp)
                print(f"Saved screenshot: {filename}")
                saved_count += 1
            
            elif key == ord('d'):
                # Toggle debug overlay
                show_debug = not show_debug
                print(f"Debug overlay: {'ON' if show_debug else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        hd.close()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
