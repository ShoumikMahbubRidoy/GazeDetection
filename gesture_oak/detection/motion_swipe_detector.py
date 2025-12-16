#!/usr/bin/env python3

import cv2
import numpy as np
import time
from collections import deque


class MotionSwipeDetector:
    """
    モーション軌跡ベースのスワイプ検出器
    """
    
    def __init__(self,
                 min_distance=100,        # 最小スワイプ距離(px) - 100cm距離用に調整
                 min_duration=0.2,        # 最小継続時間(s)
                 max_duration=3.0,        # 最大継続時間(s)
                 min_velocity=20,         # 最小速度(px/s)
                 max_velocity=1000,       # 最大速度(px/s)
                 max_y_deviation=0.4,     # Y軸の最大偏差(ratio)
                 smoothing_window=5):     # 軌跡スムージング
        
        self.min_distance = min_distance
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.max_y_deviation = max_y_deviation
        self.smoothing_window = smoothing_window
        
        # 統計
        self.total_swipes_detected = 0
        self.false_positives_filtered = 0
        
        # 状態管理
        self.current_state = 'idle'  # idle, detecting, validating, confirmed
        self.detection_start_time = None
        self.detection_start_pos = None
        
        # 軌跡バッファ
        self.trail_buffer = deque(maxlen=50)
        
    def analyze_motion_trail(self, motion_points):
        """モーション軌跡を解析してスワイプを検出"""
        if len(motion_points) < 3:
            self.current_state = 'idle'
            return False
        
        current_time = time.time()
        
        # 軌跡をスムージング
        smoothed_trail = self._smooth_trail(motion_points)
        
        if len(smoothed_trail) < 3:
            return False
        
        # 現在の状態に応じて処理
        if self.current_state == 'idle':
            return self._start_detection(smoothed_trail, current_time)
        
        elif self.current_state == 'detecting':
            return self._continue_detection(smoothed_trail, current_time)
        
        elif self.current_state == 'validating':
            return self._validate_swipe(smoothed_trail, current_time)
        
        return False
    
    def _smooth_trail(self, motion_points):
        """軌跡をスムージング"""
        if len(motion_points) < self.smoothing_window:
            return [(p['position'], p['timestamp']) for p in motion_points]
        
        smoothed = []
        positions = [p['position'] for p in motion_points]
        timestamps = [p['timestamp'] for p in motion_points]
        
        for i in range(len(positions)):
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(positions), i + self.smoothing_window // 2 + 1)
            
            window_positions = positions[start_idx:end_idx]
            avg_x = sum(pos[0] for pos in window_positions) / len(window_positions)
            avg_y = sum(pos[1] for pos in window_positions) / len(window_positions)
            
            smoothed.append(((int(avg_x), int(avg_y)), timestamps[i]))
        
        return smoothed
    
    def _start_detection(self, trail, current_time):
        """検出開始"""
        if len(trail) >= 3:
            self.current_state = 'detecting'
            self.detection_start_time = current_time
            self.detection_start_pos = trail[0][0]
            self.trail_buffer.clear()
            self.trail_buffer.extend(trail)
        
        return False
    
    def _continue_detection(self, trail, current_time):
        """検出継続"""
        if not self.detection_start_time:
            self.current_state = 'idle'
            return False
        
        duration = current_time - self.detection_start_time
        
        # タイムアウトチェック
        if duration > self.max_duration:
            self.current_state = 'idle'
            return False
        
        # 軌跡バッファ更新
        self.trail_buffer.extend(trail[-3:])  # 最新の3ポイントを追加
        
        # 最低継続時間チェック
        if duration >= self.min_duration:
            self.current_state = 'validating'
        
        return False
    
    def _validate_swipe(self, trail, current_time):
        """スワイプ検証"""
        if not self.detection_start_pos or len(self.trail_buffer) < 5:
            self.current_state = 'idle'
            return False
        
        # 最新の位置
        latest_pos = trail[-1][0]
        
        # 距離計算
        total_distance = latest_pos[0] - self.detection_start_pos[0]  # X方向の移動
        y_distance = abs(latest_pos[1] - self.detection_start_pos[1])  # Y方向の移動
        
        duration = current_time - self.detection_start_time
        
        # スワイプ条件チェック
        if (total_distance >= self.min_distance and  # 十分な右方向移動
            duration >= self.min_duration and
            duration <= self.max_duration):
            
            # Y軸偏差チェック
            y_deviation_ratio = y_distance / max(abs(total_distance), 1)
            
            if y_deviation_ratio <= self.max_y_deviation:
                # 速度チェック
                velocity = total_distance / duration
                
                if self.min_velocity <= velocity <= self.max_velocity:
                    # スワイプ検出成功
                    self.total_swipes_detected += 1
                    self.current_state = 'confirmed'
                    
                    # 短時間待機後にリセット
                    self._schedule_reset()
                    
                    return True
        
        # 無効なスワイプ
        self.false_positives_filtered += 1
        self.current_state = 'idle'
        return False
    
    def _schedule_reset(self):
        """確認状態から戻るためのタイマー設定"""
        # 実際の実装では別スレッドやタイマーを使用
        # ここでは簡単のため、次のフレームでリセット
        pass
    
    def get_current_progress(self):
        """現在のスワイプ進行状況を取得"""
        if self.current_state == 'idle':
            return {
                'state': 'idle',
                'distance': 0,
                'velocity': 0,
                'progress': 0.0,
                'y_deviation': 0
            }
        
        if not self.detection_start_pos or len(self.trail_buffer) == 0:
            return {
                'state': self.current_state,
                'distance': 0,
                'velocity': 0,
                'progress': 0.0,
                'y_deviation': 0
            }
        
        latest_pos = self.trail_buffer[-1]['position'] if isinstance(self.trail_buffer[-1], dict) else self.trail_buffer[-1][0]
        
        distance = latest_pos[0] - self.detection_start_pos[0]
        y_distance = abs(latest_pos[1] - self.detection_start_pos[1])
        
        current_time = time.time()
        duration = current_time - (self.detection_start_time or current_time)
        
        velocity = distance / max(duration, 0.01)
        progress = min(distance / self.min_distance, 1.0)
        y_deviation = y_distance / max(abs(distance), 1)
        
        return {
            'state': self.current_state,
            'distance': distance,
            'velocity': velocity,
            'progress': max(progress, 0.0),
            'y_deviation': y_deviation
        }
    
    def get_statistics(self):
        """統計情報を取得"""
        return {
            'total_swipes_detected': self.total_swipes_detected,
            'false_positives_filtered': self.false_positives_filtered
        }
    
    def reset_statistics(self):
        """統計をリセット"""
        self.total_swipes_detected = 0
        self.false_positives_filtered = 0
    
    def reset_state(self):
        """状態をリセット"""
        if self.current_state == 'confirmed':
            self.current_state = 'idle'
            self.detection_start_time = None
            self.detection_start_pos = None
            self.trail_buffer.clear()