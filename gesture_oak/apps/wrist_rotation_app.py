#!/usr/bin/env python3
"""
Wrist Rotation Detection App - RGB Version
Uses RGB camera with Nakakawa-san's HandTracker
All features from original IR version, now with RGB camera
"""
import os, sys, cv2, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gesture_oak.detection.rgb_hand_detector import RGBHandDetector
from gesture_oak.detection.wrist_rotation_detector import (
    WristRotationDetector, HandState, RotationPosition, WRIST
)

def draw_hand_landmarks(img, hand):
    """Draw hand skeleton"""
    if not hasattr(hand, 'landmarks') or hand.landmarks is None: 
        return
    lm = hand.landmarks
    
    # Lines
    lines = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    for a,b in lines:
        cv2.line(img, tuple(lm[a].astype(int)), tuple(lm[b].astype(int)), (0,255,0), 2)
    
    # Points
    for i,p in enumerate(lm):
        r = 6 if i in [0,4,8,12,16,20] else 4
        col = (0,0,255) if i in [0,4,8,12,16,20] else (255,255,255)
        cv2.circle(img, tuple(p.astype(int)), r, col, -1)


def draw_position_zones(img, wrist_pt, angle, current_position):
    """
    Draw 4 position zones as a fan/cone from wrist
    Zones follow the hand as it moves!
    """
    cx, cy = int(wrist_pt[0]), int(wrist_pt[1])
    h, w = img.shape[:2]
    
    # Radius of the fan (based on distance from wrist to edge)
    radius = 200
    
    # Zone boundaries (in degrees)
    # LEFT=0°, UP=90°, RIGHT=180°
    boundaries = [0, 60, 90, 120, 180]
    
    # Zone colors
    zone_colors = {
        1: (0, 255, 255),    # Position 1 - Yellow
        2: (0, 255, 0),      # Position 2 - Green  
        3: (255, 128, 0),    # Position 3 - Orange
        4: (0, 0, 255)       # Position 4 - Red
    }
    
    def angle_to_point(angle_deg):
        """Convert angle to point on screen"""
        screen_angle = 180.0 - angle_deg
        rad = np.radians(screen_angle)
        x = cx + int(radius * np.cos(rad))
        y = cy - int(radius * np.sin(rad))
        return (x, y)
    
    # Draw each zone
    for zone_num in range(1, 5):
        start_angle = boundaries[zone_num - 1]
        end_angle = boundaries[zone_num]
        
        color = zone_colors[zone_num]
        
        # If this is the current position, make it bright/thick
        if zone_num == current_position:
            thickness = -1  # Fill
            alpha = 0.3
        else:
            thickness = 2
            alpha = 0.15
        
        # Create overlay for transparency
        overlay = img.copy()
        
        # Draw filled sector for current position
        if zone_num == current_position:
            # Create polygon for filled sector
            pts = [np.array([cx, cy])]
            # Add arc points
            for a in range(int(start_angle), int(end_angle) + 1, 5):
                pt = angle_to_point(a)
                pts.append(np.array(pt))
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Draw boundary lines
        start_pt = angle_to_point(start_angle)
        end_pt = angle_to_point(end_angle)
        
        cv2.line(img, (cx, cy), start_pt, (200, 200, 200), 2)
        cv2.line(img, (cx, cy), end_pt, (200, 200, 200), 2)
        
        # Draw arc
        cv2.ellipse(img, (cx, cy), (radius, radius), 
                   0,  # rotation
                   -int(end_angle),  # start
                   -int(start_angle),  # end
                   (200, 200, 200), 2)
        
        # Draw zone number
        mid_angle = (start_angle + end_angle) / 2
        label_pt = angle_to_point(mid_angle)
        label_pt = (int(label_pt[0] * 0.6 + cx * 0.4), int(label_pt[1] * 0.6 + cy * 0.4))
        cv2.putText(img, str(zone_num), label_pt,
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Draw current angle indicator line
    if angle is not None:
        angle_pt = angle_to_point(angle)
        cv2.line(img, (cx, cy), angle_pt, (255, 255, 255), 3)
        cv2.circle(img, angle_pt, 8, (0, 255, 0), -1)
    
    # Draw wrist center
    cv2.circle(img, (cx, cy), 8, (255, 255, 255), -1)
    cv2.circle(img, (cx, cy), 10, (0, 0, 0), 2)


def main():
    print("="*70)
    print("WRIST ROTATION DETECTION - RGB MODE")
    print("="*70)
    print("Using RGB camera with HandTracker")
    print()
    print("Hand State: FISTED / OPEN (for display)")
    print("Position: 1/2/3/4 (ALWAYS updates based on angle)")
    print()
    print("Position Mapping:")
    print("  1: 0-60° (LEFT)")
    print("  2: 60-90° (LEFT-CENTER)")
    print("  3: 90-120° (RIGHT-CENTER)")
    print("  4: 120-180° (RIGHT)")
    print()
    print("UDP Messages:")
    print("  gesture/five  : Open hand")
    print("  gesture/zero  : Fisted hand")
    print("  area/menu/1-4 : Position")
    print("  area/menu/0   : No hand")
    print()
    print("Controls: q=Quit | r=Reset | s=Save")
    print("="*70)

    # Initialize RGB hand detector
    print("\nInitializing RGB hand detector...")
    hd = RGBHandDetector(fps=30, resolution=(1280, 720), pd_score_thresh=0.5, use_gesture=False)
    
    if not hd.connect():
        print("❌ Failed to connect!"); 
        return

    # Initialize wrist rotation detector
    det = WristRotationDetector()
    print("✓ Ready!")
    
    saved = 0
    try:
        while True:
            # Get frame
            frame, hands, depth = hd.get_frame_and_hands()
            if frame is None: 
                continue

            # Mirror frame
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Mirror landmarks
            for hnd in hands:
                if hasattr(hnd, 'landmarks') and hnd.landmarks is not None:
                    hnd.landmarks[:, 0] = w - hnd.landmarks[:, 0]

            disp = frame.copy()

            if hands:
                hand = hands[0]
                
                # Update detector
                pos, ang, state = det.update(hand)
                info = det.get_state_info()
                
                # Get wrist position
                wrist_pt = hand.landmarks[WRIST]
                
                # Draw position zones FIRST (behind hand)
                draw_position_zones(disp, wrist_pt, ang, pos.value)
                
                # Draw hand ON TOP of zones
                draw_hand_landmarks(disp, hand)

                # Info panel (top-left)
                overlay = disp.copy()
                cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, disp, 0.3, 0, disp)

                y = 40
                
                # Mode indicator
                cv2.putText(disp, "RGB MODE", (20, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y += 35
                
                # Hand State
                if state == HandState.FISTED:
                    txt, col = "FISTED = 1", (0, 0, 255)
                elif state == HandState.OPEN:
                    txt, col = "HAND OPEN = 1", (0, 255, 0)
                else:
                    txt, col = "PUT YOUR FISTED HAND", (128, 128, 128)
                
                cv2.putText(disp, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                y += 45

                # Position (ALWAYS shown!)
                pos_txt = f"Position: {pos.value}"
                cv2.putText(disp, pos_txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                y += 40

                # Angle
                if ang is not None:
                    cv2.putText(disp, f"Angle: {ang:.1f}°", (20, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

                # Large position indicator (bottom-right)
                pos_colors = {
                    1: (0, 255, 255),    # Yellow
                    2: (0, 255, 0),      # Green
                    3: (255, 128, 0),    # Orange
                    4: (0, 0, 255)       # Red
                }
                
                if pos.value > 0:
                    col = pos_colors.get(pos.value, (255, 255, 255))
                    cv2.putText(disp, str(pos.value), (w - 120, h - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 3.5, col, 8)
                    
                    # Position circle
                    cv2.circle(disp, (w - 70, h - 60), 45, col, 4)

            else:
                # NO HAND - notify detector
                det.update(None)
                
                # No hand message
                cv2.putText(disp, "PUT YOUR HAND IN VIEW", (w//2 - 220, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                
                # Mode indicator
                cv2.putText(disp, "RGB MODE", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # FPS
            fps = hd.fps_counter.get_global()
            cv2.putText(disp, f"FPS: {fps:.1f}", (w - 160, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show
            cv2.imshow("Wrist Rotation Detection (RGB)", disp)

            # Keys
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): 
                break
            elif k == ord('r'): 
                det.reset()
            elif k == ord('s'):
                fname = f"wrist_rgb_{saved:04d}.jpg"
                cv2.imwrite(fname, disp)
                print(f"Saved {fname}")
                saved += 1
                
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        hd.close()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()