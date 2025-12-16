#!/usr/bin/env python3

import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(0)  # let OpenCV choose best for this process
import numpy as np
from ..detection.hand_detector import HandDetector
from ..detection.swipe_detector import SwipeDetector


def draw_swipe_trail(frame, swipe_detector):
    """スワイプの軌跡を描画"""
    if len(swipe_detector.position_buffer) < 2:
        return
    
    positions = list(swipe_detector.position_buffer)
    
    # 軌跡を線で描画
    for i in range(1, len(positions)):
        pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
        pt2 = (int(positions[i][0]), int(positions[i][1]))
        
        # 透明度を距離に応じて変更
        alpha = (i / len(positions)) * 0.8 + 0.2
        color = (int(255 * alpha), int(100 * alpha), int(100 * alpha))
        
        cv2.line(frame, pt1, pt2, color, 3)
    
    # 最新位置に円を描画
    if positions:
        latest_pos = (int(positions[-1][0]), int(positions[-1][1]))
        cv2.circle(frame, latest_pos, 8, (0, 255, 255), -1)


def draw_swipe_zone(frame, min_distance=120):
    """スワイプ検出ゾーンを描画"""
    height, width = frame.shape[:2]
    
    # 左端から最小距離の線を描画
    cv2.line(frame, (0, height//4), (0, 3*height//4), (128, 128, 128), 2)
    cv2.line(frame, (min_distance, height//4), (min_distance, 3*height//4), (128, 255, 128), 2)
    
    # ゾーン説明
    cv2.putText(frame, "Swipe Zone", (10, height//4 - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 255, 128), 2)
    cv2.putText(frame, f"Min: {min_distance}px", (10, 3*height//4 + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 255, 128), 1)


def main():
    print("OAK-D Swipe Detection Focused Demo")
    print("=" * 40)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'r' - Reset statistics")
    print("  '1' - Strict mode (高精度)")
    print("  '2' - Normal mode (標準)")
    print("  '3' - Loose mode (検出しやすい)")
    
    # Initialize hand detector optimized for IR detection
    detector = HandDetector(
        fps=30,
        resolution=(640, 480),
        pd_score_thresh=0.1,   # Very low for IR detection
        use_gesture=False,     # ジェスチャー認識は無効化してパフォーマンス向上
        use_rgb=True          # Force IR camera for dark environments
    )
    
    # Initialize swipe detector with relaxed parameters (Easy detection mode)
    swipe_detector = SwipeDetector(
        buffer_size=12,        # Smaller buffer for faster response
        min_distance=80,       # Reduced minimum distance
        min_duration=0.2,      # Shorter minimum duration
        max_duration=3.0,      # Longer maximum duration
        min_velocity=30,       # Lower minimum velocity
        max_velocity=1000,     # Higher maximum velocity
        max_y_deviation=0.5    # Allow more Y deviation
    )
    
    # Connect to device
    if not detector.connect():
        print("Failed to connect to OAK-D device")
        return
    
    print("Swipe detection demo started...")
    print("Perform left-to-right swipes to test detection!")
    
    frame_count = 0
    last_swipe_alert = 0
    current_mode = "Normal"
    
    try:
        while True:
            # Get frame and hand detections
            frame, hands, depth_frame = detector.get_frame_and_hands()
            
            if frame is None:
                continue
                
            frame_count += 1
            
            # スワイプ検出用の手の中心位置を取得
            hand_center = None
            if hands:
                hand = hands[0]
                if hasattr(hand, 'rect_x_center_a') and hasattr(hand, 'rect_y_center_a'):
                    hand_center = (hand.rect_x_center_a, hand.rect_y_center_a)
            
            # スワイプ検出を更新
            swipe_detected = swipe_detector.update(hand_center)
            
            # スワイプゾーンを描画（コメントアウト）
            # draw_swipe_zone(frame, swipe_detector.min_distance)
            
            # 手の軌跡を描画
            draw_swipe_trail(frame, swipe_detector)
            
            # 手のバウンディングボックスを簡単に描画
            for hand in hands:
                if hasattr(hand, 'rect_points') and hand.rect_points is not None:
                    points = np.array(hand.rect_points, dtype=np.int32)
                    cv2.polylines(frame, [points], True, (0, 255, 255), 2)
            
            # 情報表示
            y_offset = 30
            line_height = 25
            
            # FPS
            fps = detector.fps_counter.get()
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
            
            # モード表示
            cv2.putText(frame, f"Mode: {current_mode}", (frame.shape[1] - 150, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
            
            # 統計情報
            stats = swipe_detector.get_statistics()
            cv2.putText(frame, f"Swipes: {stats['total_swipes_detected']}", (frame.shape[1] - 150, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
            
            cv2.putText(frame, f"Filtered: {stats['false_positives_filtered']}", (frame.shape[1] - 150, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
            
            # スワイプ進行状況
            progress = swipe_detector.get_current_swipe_progress()
            if progress:
                state_colors = {
                    'idle': (128, 128, 128),
                    'detecting': (0, 255, 255),
                    'validating': (255, 255, 0),
                    'confirmed': (0, 255, 0)
                }
                color = state_colors.get(progress['state'], (255, 255, 255))
                
                cv2.putText(frame, f"State: {progress['state']}", (frame.shape[1] - 150, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += line_height
                
                if progress['distance'] > 0:
                    cv2.putText(frame, f"Dist: {progress['distance']:.0f}px", (frame.shape[1] - 150, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += line_height
                    
                    cv2.putText(frame, f"Vel: {progress['velocity']:.0f}px/s", (frame.shape[1] - 150, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += line_height
                    
                    # 進行度バー
                    bar_width = 100
                    bar_height = 10
                    bar_x = frame.shape[1] - 150
                    bar_y = y_offset
                    
                    # 背景
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                    # 進行度
                    progress_width = int(bar_width * min(progress['progress'], 1.0))
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), color, -1)
            
            # スワイプ検出時の視覚的フィードバック
            if swipe_detected:
                last_swipe_alert = frame_count
                print(f"🚀 SWIPE DETECTED! (#{stats['total_swipes_detected']}) - Mode: {current_mode}")
            
            # スワイプアラート表示（2秒間）
            if frame_count - last_swipe_alert < 60:  # 30fps * 2sec
                cv2.putText(frame, "SWIPE!", (frame.shape[1]//2 - 50, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                # 矢印
                cv2.arrowedLine(frame, (frame.shape[1]//2 - 80, 120), 
                               (frame.shape[1]//2 + 80, 120), (0, 255, 0), 8)
            
            # Display frame
            cv2.imshow("OAK-D Swipe Detection Demo", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"swipe_demo_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('r'):
                swipe_detector.reset_statistics()
                print("Statistics reset")
            elif key == ord('1'):  # Strict mode
                swipe_detector = SwipeDetector(
                    buffer_size=20, min_distance=150, min_duration=0.4,
                    max_duration=1.5, min_velocity=80, max_velocity=600,
                    max_y_deviation=0.2
                )
                current_mode = "Strict"
                print("Switched to Strict mode (high precision)")
            elif key == ord('2'):  # Normal mode
                swipe_detector = SwipeDetector(
                    buffer_size=15, min_distance=120, min_duration=0.3,
                    max_duration=2.0, min_velocity=50, max_velocity=800,
                    max_y_deviation=0.3
                )
                current_mode = "Normal"
                print("Switched to Normal mode")
            elif key == ord('3'):  # Loose mode
                swipe_detector = SwipeDetector(
                    buffer_size=10, min_distance=80, min_duration=0.2,
                    max_duration=3.0, min_velocity=30, max_velocity=1000,
                    max_y_deviation=0.5
                )
                current_mode = "Loose"
                print("Switched to Loose mode (easy detection)")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error during execution: {e}")
    
    finally:
        # Cleanup
        detector.close()
        cv2.destroyAllWindows()
        
        # Print final statistics
        total_fps = detector.fps_counter.get_global()
        final_stats = swipe_detector.get_statistics()
        print(f"\nSwipe Detection Demo Statistics:")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {total_fps:.2f}")
        print(f"Detection mode: {current_mode}")
        print(f"Total swipes detected: {final_stats['total_swipes_detected']}")
        print(f"False positives filtered: {final_stats['false_positives_filtered']}")
        if final_stats['total_swipes_detected'] > 0:
            precision = final_stats['total_swipes_detected'] / (final_stats['total_swipes_detected'] + final_stats['false_positives_filtered']) * 100
            print(f"Detection precision: {precision:.1f}%")
        print("Swipe detection demo completed.")


if __name__ == "__main__":
    main()