#!/usr/bin/env python3
"""
Smart Combined App - RGB Mode
Automatically switches between:
- Wrist Rotation (OPEN hand) → 4 positions
- 3-Area Pointing (ONE finger) → 3 areas
- FIST works in both modes
"""
import os, sys, cv2, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gesture_oak.detection.rgb_hand_detector import RGBHandDetector
from gesture_oak.detection.smart_combined_detector import (
    SmartCombinedDetector, DetectionMode, GestureType
)

# Landmarks
WRIST, INDEX_TIP = 0, 8


def draw_hand_skeleton(img, hand):
    """Draw hand skeleton"""
    if not hasattr(hand, 'landmarks') or hand.landmarks is None:
        return
    
    lm = hand.landmarks
    
    # Connection lines
    connections = [
        (0,1),(1,2),(2,3),(3,4),          # Thumb
        (0,5),(5,6),(6,7),(7,8),          # Index
        (0,9),(9,10),(10,11),(11,12),     # Middle
        (0,13),(13,14),(14,15),(15,16),   # Ring
        (0,17),(17,18),(18,19),(19,20),   # Pinky
        (5,9),(9,13),(13,17)              # Palm
    ]
    
    for a, b in connections:
        pt1 = tuple(lm[a].astype(int))
        pt2 = tuple(lm[b].astype(int))
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    
    # Landmark points
    for i, pt in enumerate(lm):
        radius = 6 if i in [0, 4, 8, 12, 16, 20] else 4
        color = (0, 0, 255) if i in [0, 4, 8, 12, 16, 20] else (255, 255, 255)
        cv2.circle(img, tuple(pt.astype(int)), radius, color, -1)


def draw_wrist_rotation_zones(img, wrist_pt, angle, current_position):
    """Draw 4-position wrist rotation zones (fan shape)"""
    cx, cy = int(wrist_pt[0]), int(wrist_pt[1])
    radius = 200
    
    boundaries = [0, 60, 90, 120, 180]
    zone_colors = {
        1: (0, 255, 255),    # Yellow
        2: (0, 255, 0),      # Green
        3: (255, 128, 0),    # Orange
        4: (0, 0, 255)       # Red
    }
    
    def angle_to_point(angle_deg):
        screen_angle = 180.0 - angle_deg
        rad = np.radians(screen_angle)
        x = cx + int(radius * np.cos(rad))
        y = cy - int(radius * np.sin(rad))
        return (x, y)
    
    # Draw zones
    for zone_num in range(1, 5):
        start_angle = boundaries[zone_num - 1]
        end_angle = boundaries[zone_num]
        color = zone_colors[zone_num]
        
        # Highlight current zone
        if zone_num == current_position:
            overlay = img.copy()
            pts = [np.array([cx, cy])]
            for a in range(int(start_angle), int(end_angle) + 1, 5):
                pt = angle_to_point(a)
                pts.append(np.array(pt))
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # Boundary lines
        start_pt = angle_to_point(start_angle)
        end_pt = angle_to_point(end_angle)
        cv2.line(img, (cx, cy), start_pt, (200, 200, 200), 2)
        cv2.line(img, (cx, cy), end_pt, (200, 200, 200), 2)
        
        # Arc
        cv2.ellipse(img, (cx, cy), (radius, radius), 0,
                   -int(end_angle), -int(start_angle), (200, 200, 200), 2)
        
        # Zone label
        mid_angle = (start_angle + end_angle) / 2
        label_pt = angle_to_point(mid_angle)
        label_pt = (int(label_pt[0] * 0.6 + cx * 0.4), int(label_pt[1] * 0.6 + cy * 0.4))
        cv2.putText(img, str(zone_num), label_pt,
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Current angle line
    if angle is not None:
        angle_pt = angle_to_point(angle)
        cv2.line(img, (cx, cy), angle_pt, (255, 255, 255), 3)
        cv2.circle(img, angle_pt, 8, (0, 255, 0), -1)
    
    # Wrist center
    cv2.circle(img, (cx, cy), 8, (255, 255, 255), -1)
    cv2.circle(img, (cx, cy), 10, (0, 0, 0), 2)


def draw_three_area_zones(img, index_tip, current_area):
    """Draw 3 vertical area zones"""
    h, w = img.shape[:2]
    area_width = w // 3
    
    # Zone colors
    area_colors = {
        1: (0, 255, 255),    # Yellow - Left
        2: (0, 255, 0),      # Green - Center
        3: (0, 0, 255)       # Red - Right
    }
    
    # Draw zones
    for area_num in range(1, 4):
        x1 = (area_num - 1) * area_width
        x2 = area_num * area_width
        color = area_colors[area_num]
        
        # Highlight current area
        if area_num == current_area:
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        
        # Border
        cv2.rectangle(img, (x1, 0), (x2, h), color, 3)
        
        # Area number (top)
        label_x = x1 + area_width // 2 - 20
        cv2.putText(img, f"Area {area_num}", (label_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    
    # Draw index finger tip indicator
    if index_tip is not None:
        tip_x, tip_y = int(index_tip[0]), int(index_tip[1])
        cv2.circle(img, (tip_x, tip_y), 15, (255, 0, 255), 3)
        cv2.circle(img, (tip_x, tip_y), 5, (255, 0, 255), -1)


def main():
    print("="*80)
    print("SMART COMBINED DETECTION - RGB MODE")
    print("="*80)
    print()
    print("🤚 OPEN HAND (2+ fingers)  → Wrist Rotation Mode (4 positions)")
    print("☝  ONE FINGER              → 3-Area Pointing Mode (3 areas)")
    print("✊ FIST                    → Universal gesture (both modes)")
    print()
    print("Wrist Rotation Zones:")
    print("  Position 1: 0-60°   (LEFT)")
    print("  Position 2: 60-90°  (LEFT-CENTER)")
    print("  Position 3: 90-120° (RIGHT-CENTER)")
    print("  Position 4: 120-180° (RIGHT)")
    print()
    print("3-Area Zones:")
    print("  Area 1: Left third")
    print("  Area 2: Center third")
    print("  Area 3: Right third")
    print()
    print("UDP Messages:")
    print("  gesture/zero  : FIST")
    print("  gesture/five  : OPEN hand")
    print("  gesture/one   : ONE finger")
    print("  area/menu/1-4 : Wrist rotation position")
    print("  area/color/1-3 : 3-area position")
    print()
    print("Controls: q=Quit | r=Reset | s=Save screenshot")
    print("="*80)
    
    # Initialize RGB detector
    print("\nInitializing RGB hand detector...")
    detector = RGBHandDetector(fps=30, resolution=(1280, 720), 
                               pd_score_thresh=0.5, use_gesture=False)
    
    if not detector.connect():
        print("❌ Failed to connect camera!")
        return
    
    # Initialize smart combined detector
    combined = SmartCombinedDetector(resolution=(1280, 720))
    
    print("✓ All systems ready!\n")
    
    saved_count = 0
    
    try:
        while True:
            # Get frame and hands
            frame, hands, _ = detector.get_frame_and_hands()
            if frame is None:
                continue
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Mirror landmarks
            for hand in hands:
                if hasattr(hand, 'landmarks') and hand.landmarks is not None:
                    hand.landmarks[:, 0] = w - hand.landmarks[:, 0]
            
            display = frame.copy()
            
            # Get first hand
            hand = hands[0] if hands else None
            
            # Update detector
            mode, gesture, position, angle = combined.update(hand, w, h)
            
            if hand is not None:
                # Get key landmarks
                wrist_pt = hand.landmarks[WRIST]
                index_tip = hand.landmarks[INDEX_TIP]
                
                # Draw based on mode
                if mode == DetectionMode.WRIST_ROTATION:
                    # Draw wrist rotation zones FIRST (background)
                    draw_wrist_rotation_zones(display, wrist_pt, angle, position)
                    # Draw hand skeleton ON TOP
                    draw_hand_skeleton(display, hand)
                    
                elif mode == DetectionMode.THREE_AREA:
                    # Draw 3-area zones FIRST (background)
                    draw_three_area_zones(display, index_tip, position)
                    # Draw hand skeleton ON TOP
                    draw_hand_skeleton(display, hand)
                
                else:
                    # Unknown mode - just draw hand
                    draw_hand_skeleton(display, hand)
                
                # Info panel (top-left)
                panel_w = 500
                panel_h = 200
                overlay = display.copy()
                cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
                
                y = 45
                
                # Mode indicator
                mode_text = "WRIST ROTATION" if mode == DetectionMode.WRIST_ROTATION else \
                           "3-AREA POINTING" if mode == DetectionMode.THREE_AREA else "UNKNOWN"
                mode_color = (0, 255, 255) if mode == DetectionMode.WRIST_ROTATION else \
                            (255, 128, 255) if mode == DetectionMode.THREE_AREA else (128, 128, 128)
                cv2.putText(display, f"Mode: {mode_text}", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
                y += 40
                
                # Gesture
                gesture_text = "FIST ✊" if gesture == GestureType.FIST else \
                              "OPEN 🤚" if gesture == GestureType.OPEN else \
                              "ONE ☝" if gesture == GestureType.ONE else "???"
                gesture_color = (0, 0, 255) if gesture == GestureType.FIST else \
                               (0, 255, 0) if gesture == GestureType.OPEN else \
                               (255, 0, 255) if gesture == GestureType.ONE else (128, 128, 128)
                cv2.putText(display, f"Gesture: {gesture_text}", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
                y += 40
                
                # Position
                if mode == DetectionMode.WRIST_ROTATION:
                    pos_text = f"Position: {position} / 4"
                elif mode == DetectionMode.THREE_AREA:
                    pos_text = f"Area: {position} / 3"
                else:
                    pos_text = "Position: --"
                
                cv2.putText(display, pos_text, (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                y += 40
                
                # Angle (if wrist rotation mode)
                if mode == DetectionMode.WRIST_ROTATION and angle is not None:
                    cv2.putText(display, f"Angle: {angle:.1f}°", (20, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
                
                # Large position indicator (bottom-right)
                if mode == DetectionMode.WRIST_ROTATION:
                    pos_colors = {
                        1: (0, 255, 255), 2: (0, 255, 0),
                        3: (255, 128, 0), 4: (0, 0, 255)
                    }
                elif mode == DetectionMode.THREE_AREA:
                    pos_colors = {
                        1: (0, 255, 255), 2: (0, 255, 0), 3: (0, 0, 255)
                    }
                else:
                    pos_colors = {}
                
                if position > 0:
                    col = pos_colors.get(position, (255, 255, 255))
                    cv2.putText(display, str(position), (w - 120, h - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 3.5, col, 8)
                    cv2.circle(display, (w - 70, h - 60), 45, col, 4)
            
            else:
                # NO HAND
                cv2.putText(display, "SHOW YOUR HAND", (w//2 - 180, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # FPS counter
            fps = detector.fps_counter.get_global()
            cv2.putText(display, f"FPS: {fps:.1f}", (w - 160, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # RGB mode indicator
            cv2.putText(display, "RGB MODE", (w - 160, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Smart Combined Detection (RGB)", display)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                combined.reset()
                print("🔄 Reset")
            elif key == ord('s'):
                filename = f"smart_combined_{saved_count:04d}.jpg"
                cv2.imwrite(filename, display)
                print(f"💾 Saved: {filename}")
                saved_count += 1
    
    except KeyboardInterrupt:
        print("\n⏹ Stopped by user")
    
    finally:
        detector.close()
        cv2.destroyAllWindows()
        print("✓ Done!")


if __name__ == "__main__":
    main()