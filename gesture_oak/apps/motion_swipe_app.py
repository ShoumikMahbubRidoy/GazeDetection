#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from ..detection.motion_detector import MotionDetector
from ..detection.motion_swipe_detector import MotionSwipeDetector


class SimpleIRCamera:
    """
    IRカメラからの画像取得用のシンプルなクラス
    MediaPipeの複雑さを排除して軽量化
    """
    
    def __init__(self, fps=30, resolution=(640, 480)):
        self.fps_target = fps
        self.resolution = resolution
        self.device = None
        self.pipeline = None
        self.q_video = None
        self.q_depth = None
    
    def create_pipeline(self):
        """軽量なパイプラインを作成"""
        pipeline = dai.Pipeline()
        
        # IR Stereo Cameras
        cam_left = pipeline.createMonoCamera()
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_left.setFps(self.fps_target)
        
        cam_right = pipeline.createMonoCamera()
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_right.setFps(self.fps_target)
        
        # Depth
        depth = pipeline.createStereoDepth()
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(True)
        cam_left.out.link(depth.left)
        cam_right.out.link(depth.right)
        
        # Convert mono to RGB
        mono_to_rgb = pipeline.createImageManip()
        mono_to_rgb.initialConfig.setResize(*self.resolution)
        mono_to_rgb.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
        cam_left.out.link(mono_to_rgb.inputImage)
        
        # Outputs
        rgb_out = pipeline.createXLinkOut()
        rgb_out.setStreamName("rgb")
        mono_to_rgb.out.link(rgb_out.input)
        
        depth_out = pipeline.createXLinkOut()
        depth_out.setStreamName("depth")
        depth.depth.link(depth_out.input)
        
        return pipeline
    
    def connect(self):
        """デバイスに接続"""
        try:
            self.pipeline = self.create_pipeline()
            self.device = dai.Device(self.pipeline)
            
            print(f"Connected to device: {self.device.getDeviceName()}")
            print(f"USB Speed: {self.device.getUsbSpeed()}")
            
            # Setup queues
            self.q_video = self.device.getOutputQueue("rgb", maxSize=1, blocking=False)
            self.q_depth = self.device.getOutputQueue("depth", maxSize=1, blocking=False)
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to OAK-D: {e}")
            return False
    
    def get_frame(self):
        """フレームを取得"""
        try:
            # RGB frame
            in_rgb = self.q_video.get()
            frame = in_rgb.getCvFrame()
            
            # Depth frame
            depth_frame = None
            if self.q_depth.has():
                in_depth = self.q_depth.get()
                depth_frame = in_depth.getFrame()
            
            return frame, depth_frame
            
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None, None
    
    def close(self):
        """接続を閉じる"""
        if self.device:
            self.device.close()
            self.device = None


def draw_motion_objects(frame, motion_objects):
    """検出された動く物体を描画"""
    for obj in motion_objects:
        center = obj['center']
        bbox = obj['bbox']
        area = obj['area']
        
        # バウンディングボックス
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 中心点
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        # エリア表示
        cv2.putText(frame, f"Area: {int(area)}", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def draw_motion_trail(frame, motion_points):
    """モーション軌跡を描画"""
    if len(motion_points) < 2:
        return
    
    # 軌跡の線を描画
    points = [p['position'] for p in motion_points]
    for i in range(1, len(points)):
        # 透明度を時間経過で変更
        alpha = (i / len(points)) * 0.8 + 0.2
        color = (int(255 * alpha), int(100 * alpha), int(100 * alpha))
        cv2.line(frame, points[i-1], points[i], color, 3)
    
    # 最新位置
    if points:
        cv2.circle(frame, points[-1], 8, (0, 255, 255), -1)


def main():
    print("OAK-D Motion-Based Swipe Detection Demo")
    print("=" * 45)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'r' - Reset statistics and calibration")
    print("  'c' - Recalibrate background")
    print("")
    print("Motion Detection for 100cm Distance (Dark Environment)")
    
    # Initialize camera
    camera = SimpleIRCamera(fps=30, resolution=(640, 480))
    
    # Initialize motion detector
    motion_detector = MotionDetector(
        motion_threshold=25,
        min_area=200,      # 100cm距離用に小さく
        max_area=20000,    # より大きな範囲を許容
        blur_size=15       # より細かいディテール
    )
    
    # Initialize swipe detector
    swipe_detector = MotionSwipeDetector(
        min_distance=80,    # 100cm距離用に短く
        min_duration=0.3,
        max_duration=4.0,   # より長い時間を許容
        min_velocity=15,    # 遠距離用に低速
        max_velocity=500,
        max_y_deviation=0.5,  # より寛容に
        smoothing_window=7    # より強いスムージング
    )
    
    if not camera.connect():
        print("Failed to connect to OAK-D device")
        return
    
    print("Motion detection started. Please wait for calibration...")
    
    frame_count = 0
    last_swipe_alert = 0
    
    try:
        while True:
            frame, depth_frame = camera.get_frame()
            
            if frame is None:
                continue
            
            frame_count += 1
            
            # キャリブレーション状況表示
            if motion_detector.is_calibrating():
                progress = motion_detector.get_calibration_progress()
                cv2.putText(frame, f"Calibrating... {progress*100:.0f}%", 
                           (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # プログレスバー
                bar_width = 200
                bar_x = frame.shape[1]//2 - bar_width//2
                bar_y = frame.shape[0]//2 + 40
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 20), (0, 255, 255), -1)
            
            # モーション検出
            motion_objects = motion_detector.detect_motion(frame)
            
            # モーション軌跡更新
            motion_trail = motion_detector.update_motion_trail(motion_objects)
            
            # スワイプ検出
            swipe_detected = False
            if not motion_detector.is_calibrating():
                swipe_detected = swipe_detector.analyze_motion_trail(motion_trail)
                if swipe_detected:
                    last_swipe_alert = frame_count
            
            # 描画
            draw_motion_objects(frame, motion_objects)
            draw_motion_trail(frame, motion_trail)
            
            # 情報表示
            fps = motion_detector.fps_counter.get()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Objects: {len(motion_objects)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # スワイプ統計
            stats = swipe_detector.get_statistics()
            cv2.putText(frame, f"Swipes: {stats['total_swipes_detected']}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # スワイプ進行状況
            progress = swipe_detector.get_current_progress()
            if progress['state'] != 'idle':
                state_colors = {
                    'detecting': (0, 255, 255),
                    'validating': (255, 255, 0),
                    'confirmed': (0, 255, 0)
                }
                color = state_colors.get(progress['state'], (255, 255, 255))
                
                cv2.putText(frame, f"State: {progress['state']}", (frame.shape[1] - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if progress['distance'] > 0:
                    cv2.putText(frame, f"Distance: {progress['distance']:.0f}px", (frame.shape[1] - 200, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, f"Progress: {progress['progress']*100:.0f}%", (frame.shape[1] - 200, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # スワイプ検出アラート
            if frame_count - last_swipe_alert < 60:  # 2秒間表示
                cv2.putText(frame, "SWIPE DETECTED!", (frame.shape[1]//2 - 100, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.arrowedLine(frame, (frame.shape[1]//2 - 80, 130), 
                               (frame.shape[1]//2 + 80, 130), (0, 255, 0), 8)
            
            # 使用説明
            if not motion_detector.is_calibrating() and len(motion_objects) == 0:
                cv2.putText(frame, "Move your hand from left to right", (10, frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, "Keep movement smooth and steady", (10, frame.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("OAK-D Motion Swipe Detection", frame)
            
            # キー処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"motion_swipe_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('r'):
                swipe_detector.reset_statistics()
                motion_detector.reset_calibration()
                print("Statistics and calibration reset")
            elif key == ord('c'):
                motion_detector.reset_calibration()
                print("Background recalibration started")
            
            # 確認状態からのリセット
            if progress['state'] == 'confirmed':
                swipe_detector.reset_state()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error during execution: {e}")
    
    finally:
        camera.close()
        cv2.destroyAllWindows()
        
        # 最終統計
        final_stats = swipe_detector.get_statistics()
        print(f"\nMotion-Based Swipe Detection Statistics:")
        print(f"Total frames processed: {frame_count}")
        print(f"Total swipes detected: {final_stats['total_swipes_detected']}")
        print(f"False positives filtered: {final_stats['false_positives_filtered']}")
        if final_stats['total_swipes_detected'] > 0:
            precision = final_stats['total_swipes_detected'] / (final_stats['total_swipes_detected'] + final_stats['false_positives_filtered']) * 100
            print(f"Detection precision: {precision:.1f}%")
        print("Motion swipe detection demo completed.")


if __name__ == "__main__":
    main()