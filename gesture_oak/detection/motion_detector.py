#!/usr/bin/env python3

import cv2
import numpy as np
from collections import deque
from ..utils.FPS import FPS


class MotionDetector:
    """
    モーション検出ベースの手の動き検出器
    MediaPipeを使わず、動く物体を追跡してスワイプを検出
    """
    
    def __init__(self, 
                 motion_threshold=25,
                 min_area=500,
                 max_area=50000,
                 blur_size=21,
                 background_learning_rate=0.01):
        
        self.motion_threshold = motion_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.blur_size = blur_size
        self.background_learning_rate = background_learning_rate
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50
        )
        
        # Motion tracking
        self.motion_points = deque(maxlen=30)  # 30フレーム分の軌跡
        self.fps_counter = FPS()
        
        # Calibration
        self.calibration_frames = 0
        self.calibration_needed = 30  # 30フレームで背景学習
        
    def detect_motion(self, frame):
        """フレームから動きを検出"""
        if frame is None:
            return []
        
        self.fps_counter.update()
        
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ガウシアンブラー
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        
        # 背景差分
        fg_mask = self.bg_subtractor.apply(blurred, learningRate=self.background_learning_rate)
        
        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 有効な動きを抽出
        motion_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                # バウンディングボックス
                x, y, w, h = cv2.boundingRect(contour)
                
                # 中心点
                center_x = x + w // 2
                center_y = y + h // 2
                
                motion_objects.append({
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contour': contour
                })
        
        # キャリブレーション中は検出を抑制
        if self.calibration_frames < self.calibration_needed:
            self.calibration_frames += 1
            return []
        
        return motion_objects
    
    def update_motion_trail(self, motion_objects):
        """動きの軌跡を更新"""
        if motion_objects:
            # 最も大きな動きを追跡（おそらく手）
            largest_motion = max(motion_objects, key=lambda x: x['area'])
            center = largest_motion['center']
            
            # 軌跡に追加
            self.motion_points.append({
                'position': center,
                'timestamp': cv2.getTickCount(),
                'area': largest_motion['area']
            })
        
        return list(self.motion_points)
    
    def is_calibrating(self):
        """キャリブレーション中かどうか"""
        return self.calibration_frames < self.calibration_needed
    
    def get_calibration_progress(self):
        """キャリブレーション進行度 (0.0 - 1.0)"""
        return min(self.calibration_frames / self.calibration_needed, 1.0)
    
    def reset_calibration(self):
        """キャリブレーションをリセット"""
        self.calibration_frames = 0
        self.motion_points.clear()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50
        )