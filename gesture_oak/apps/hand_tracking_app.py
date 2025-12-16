# src/gesture_oak/apps/hand_tracking_app.py
#!/usr/bin/env python3
import os
import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(0)  # let OpenCV choose best for this process
import numpy as np
from pathlib import Path
from ..detection.hand_detector import HandDetector
from ..detection.swipe_detector import SwipeDetector
from ..utils import mediapipe_utils as mpu

def draw_hand_landmarks(frame, hand):
    """Draw hand landmarks and bounding box on frame"""
    # Draw landmarks
    if hasattr(hand, 'landmarks') and hand.landmarks is not None:
        for idx, landmark in enumerate(hand.landmarks):
            x, y = int(landmark[0]), int(landmark[1])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if idx in [0, 4, 8, 12, 16, 20]:  # Fingertips and wrist
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

    # Draw bounding box
    if hasattr(hand, 'rect_points') and hand.rect_points is not None:
        points = np.array(hand.rect_points, dtype=np.int32)
        cv2.polylines(frame, [points], True, (0, 255, 255), 2)

    # Draw label and confidence with depth info
    if hasattr(hand, 'label') and hasattr(hand, 'lm_score'):
        depth_info = ""
        if hasattr(hand, 'depth'):
            depth_info = f" D:{hand.depth:.0f}mm"
        if hasattr(hand, 'depth_confidence'):
            depth_info += f" C:{hand.depth_confidence:.2f}"
        label_text = f"{hand.label}: {hand.lm_score:.2f}{depth_info}"
        if hasattr(hand, 'rect_x_center_a'):
            x = int(hand.rect_x_center_a - hand.rect_w_a // 2)
            y = int(hand.rect_y_center_a - hand.rect_h_a // 2 - 10)
            cv2.putText(
                frame, label_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 0), 2
            )

    # Draw gesture if available
    if hasattr(hand, 'gesture') and hand.gesture is not None:
        if hasattr(hand, 'rect_x_center_a'):
            x = int(hand.rect_x_center_a - hand.rect_w_a // 2)
            y = int(hand.rect_y_center_a + hand.rect_h_a // 2 + 20)
            cv2.putText(
                frame, f"Gesture: {hand.gesture}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

def _safe_fps(detector) -> float:
    """Return rolling/avg FPS safely regardless of FPS implementation."""
    fc = getattr(detector, "fps_counter", None)
    if fc is None:
        return 0.0
    # Prefer short-window/rolling if available
    for attr in ("get", "fps"):
        if hasattr(fc, attr):
            try:
                return float(getattr(fc, attr)())
            except Exception:
                pass
    # Fallback to average/global
    for attr in ("get_global", "avg_fps"):
        if hasattr(fc, attr):
            try:
                return float(getattr(fc, attr)())
            except Exception:
                pass
    return 0.0

def _safe_fps_avg(detector) -> float:
    """Return global/average FPS safely."""
    fc = getattr(detector, "fps_counter", None)
    if fc is None:
        return 0.0
    for attr in ("avg_fps", "get_global"):
        if hasattr(fc, attr):
            try:
                return float(getattr(fc, attr)())
            except Exception:
                pass
    # As a last resort compute frames/elapsed if available
    try:
        frames = fc.frames() if hasattr(fc, "frames") else 0
        elapsed = fc.elapsed() if hasattr(fc, "elapsed") else 0.0
        return (frames / elapsed) if elapsed and frames else 0.0
    except Exception:
        return 0.0

def main():
    print("OAK-D Hand Detection Demo with Swipe Detection")
    print("=" * 45)
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press 'r' to reset swipe statistics")

    # STOP flag path from launcher (for graceful external stop)
    stop_file_path = os.environ.get("TG25_STOP_FILE", "")
    stop_file = Path(stop_file_path) if stop_file_path else None

    # Initialize hand detector optimized for IR detection in dark environments
    detector = HandDetector(
        fps=30,
        resolution=(640, 480),
        pd_score_thresh=0.10,  # low for distance
        use_gesture=True,
        use_rgb=True  # Force IR camera for dark environments
    )

    # Initialize swipe detector
    swipe_detector = SwipeDetector(
        buffer_size=12,
        min_distance=80,
        min_duration=0.2,
        max_duration=3.0,
        min_velocity=30,
        max_velocity=1000,
        max_y_deviation=0.5
    )

    # Connect to device
    if not detector.connect():
        print("Failed to connect to OAK-D device")
        return

    print("Hand detection started. Showing live preview...")

    frame_count = 0
    last_swipe_alert = 0

    try:
        while True:
            # External graceful stop?
            if stop_file and stop_file.exists():
                print("Stop flag detected. Shutting down gracefully...")
                break

            # Get frame & detections
            frame, hands, depth_frame = detector.get_frame_and_hands()
            if frame is None:
                continue

            frame_count += 1
            if hasattr(detector, "fps_counter") and hasattr(detector.fps_counter, "update"):
                try:
                    detector.fps_counter.update()
                except Exception:
                    pass

            # Swipe input = center of first hand
            hand_center = None
            if hands:
                hand = hands[0]
                if hasattr(hand, 'rect_x_center_a') and hasattr(hand, 'rect_y_center_a'):
                    hand_center = (hand.rect_x_center_a, hand.rect_y_center_a)

            # Update swipe
            swipe_detected = swipe_detector.update(hand_center)

            # Draw overlays
            fps = _safe_fps(detector)
            cv2.putText(frame, f"", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Hands: {len(hands)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            depth_status = "" if depth_frame is not None else "Depth: OFF"
            cv2.putText(frame, depth_status, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if depth_frame is not None else (0, 0, 255), 2)

            if len(hands) == 0:
                cv2.putText(frame, "",
                            (frame.shape[1] // 2 - 200, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            stats = swipe_detector.get_statistics()
            cv2.putText(frame, f"Swipes: {stats['total_swipes_detected']}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            progress = swipe_detector.get_current_swipe_progress()
            if progress:
                state_color = (0, 255, 255) if progress['state'] != 'idle' else (128, 128, 128)
                cv2.putText(frame, f"State: {progress['state']}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
                if progress['distance'] > 0:
                    cv2.putText(frame, f"Distance: {progress['distance']:.0f}px (need: 80px)",
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
                cv2.putText(frame, f"Velocity: {progress['velocity']:.0f}px/s (need: 30-1000)",
                            (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
                cv2.putText(frame, f"Progress: {progress['progress']:.1%}", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

            # Swipe alert
            if swipe_detected:
                last_swipe_alert = frame_count
                print(f" LEFT-TO-RIGHT SWIPE DETECTED! (Total: {stats['total_swipes_detected']})")
            if frame_count - last_swipe_alert < 90:  # ~3s @30fps
                cv2.putText(frame, "SWIPE DETECTED!", (frame.shape[1] // 2 - 100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.arrowedLine(frame, (frame.shape[1] // 2 - 50, 80),
                                (frame.shape[1] // 2 + 50, 80), (0, 255, 0), 5)

            # Draw hands + optional logging
            for i, hand in enumerate(hands):
                draw_hand_landmarks(frame, hand)
                if frame_count % 2 == 0:
                    lab = getattr(hand, 'label', '?')
                    sc = getattr(hand, 'lm_score', 0.0)
                    print(f"Hand {i+1}: {lab} (confidence: {sc:.3f})")
                    if hasattr(hand, 'gesture') and hand.gesture:
                        print(f"  Gesture: {hand.gesture}")

            cv2.imshow("OAK-D Hand Detection with Swipe", frame)

            # Local keys (still allow 'q' to quit)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"hand_detection_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('r'):
                swipe_detector.reset_statistics()
                print("Swipe statistics reset")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # Always close resources
        try: detector.close()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass

        if hasattr(detector, "fps_counter") and hasattr(detector.fps_counter, "stop"):
            try: detector.fps_counter.stop()
            except Exception: pass

        frames = 0
        if hasattr(detector, "fps_counter") and hasattr(detector.fps_counter, "frames"):
            try: frames = int(detector.fps_counter.frames())
            except Exception: frames = 0

        avg_fps = _safe_fps_avg(detector)
        final_stats = swipe_detector.get_statistics() if hasattr(swipe_detector, "get_statistics") else {
            "total_swipes_detected": getattr(swipe_detector, "total_swipes", 0),
            "filtered_false_positives": getattr(swipe_detector, "filtered_fp", 0)
        }
        print("\nSession Statistics:")
        print(f"Total frames processed: {frames}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Total swipes detected: {final_stats.get('total_swipes_detected', 0)}")
        print(f"False positives filtered: {final_stats.get('filtered_false_positives', 0)}")
        print("Hand detection demo completed.")

if __name__ == "__main__":
    main()
