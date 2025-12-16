#!/usr/bin/env python3
"""
Smart Combined Detector - Automatically switches modes
Mode 1: Wrist Rotation (OPEN hand) → 4 positions
Mode 2: 3-Area Pointing (ONE finger) → 3 areas
"""
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np
import socket
import time

__all__ = ["SmartCombinedDetector", "DetectionMode", "GestureType"]

# MediaPipe landmarks
WRIST = 0
INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP = 8, 7, 6, 5
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18
THUMB_TIP, THUMB_CMC = 4, 1


class DetectionMode(Enum):
    """Current detection mode"""
    WRIST_ROTATION = auto()  # 4-position wrist rotation
    THREE_AREA = auto()      # 3-area pointing
    UNKNOWN = auto()


class GestureType(Enum):
    """Detected gestures"""
    UNKNOWN = 0
    FIST = auto()      # gesture/zero
    OPEN = auto()      # gesture/five  
    ONE = auto()       # gesture/one


def _distance(p1, p2):
    """Euclidean distance"""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


class SmartCombinedDetector:
    """
    Smart detector that switches between two modes:
    - OPEN hand (2+ fingers) → Wrist Rotation Mode (4 positions)
    - ONE finger → 3-Area Pointing Mode (3 areas)
    - FIST → Works in both modes
    
    UDP Messages:
    - gesture/zero, five, one
    - area/menu/1-4 (wrist rotation)
    - area/color/1-3 (3-area pointing)
    """
    
    def __init__(self, resolution=(1280, 720), udp_ip="192.168.0.10", udp_port=9000):
        self.width, self.height = resolution
        
        # UDP setup
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
            print(f"✓ UDP enabled: {udp_ip}:{udp_port}")
        except Exception as e:
            print(f"⚠ UDP socket error: {e}")
            self.sock = None
        
        # Current mode
        self.mode = DetectionMode.UNKNOWN
        self.current_gesture = GestureType.UNKNOWN
        
        # Gesture stability tracking (1-second debouncing)
        self._gesture_history = []
        self._gesture_stable_time = 1.0  # seconds (changed from 2.0)
        self._last_gesture_change_time = time.time()
        self._confirmed_gesture = GestureType.UNKNOWN
        
        # Wrist rotation state (Mode 1)
        self.wrist_position = 0  # 1-4
        self.wrist_angle = None
        
        # 3-area state (Mode 2)
        self.area_position = 0  # 1-3
        
        # UDP tracking
        self._last_gesture_msg = None
        self._last_position_msg = None
        
        # NO HAND delay
        self._no_hand_start_time = None
        self._no_hand_sent = False
        
        print("✓ Smart Combined Detector initialized")
        print("  - OPEN hand → Wrist Rotation (4 zones)")
        print("  - ONE finger → 3-Area Pointing")
        print("  - FIST → Universal gesture")
    
    def update(self, hand, frame_width, frame_height):
        """
        Main update - detects gesture and switches mode accordingly
        
        Returns:
            mode, gesture, position (1-4 or 1-3), angle (if wrist mode)
        """
        # Update resolution if changed
        self.width = frame_width
        self.height = frame_height
        
        # Extract landmarks
        lms = self._extract_landmarks(hand)
        
        if lms is None:
            # NO HAND
            if self._no_hand_start_time is None:
                self._no_hand_start_time = time.time()
            
            elapsed = time.time() - self._no_hand_start_time
            if elapsed >= 3.0 and not self._no_hand_sent:
                self._send_no_hand_udp()
                self._no_hand_sent = True
            
            return DetectionMode.UNKNOWN, GestureType.UNKNOWN, 0, None
        
        # Hand detected - reset NO HAND timer
        self._no_hand_start_time = None
        self._no_hand_sent = False
        
        # Detect gesture type
        gesture = self._detect_gesture(lms)
        
        # FINAL GESTURE STABILITY RULES:
        # - FIST: Needs 2s stability (deliberate action)
        # - OPEN: Needs 2s stability (mode switching)
        # - ONE from FIST: Instant (releasing action)
        # - ONE from OPEN: Needs 2s stability (mode switching)
        current_time = time.time()
        
        # Track gesture history
        self._gesture_history.append((gesture, current_time))
        cutoff_time = current_time - (self._gesture_stable_time + 0.5)
        self._gesture_history = [(g, t) for g, t in self._gesture_history if t > cutoff_time]
        
        # Determine if we need stability check
        needs_stability = False
        
        if gesture == GestureType.FIST:
            # FIST always needs 2s stability
            needs_stability = True
            
        elif gesture == GestureType.OPEN:
            # OPEN always needs 2s stability
            needs_stability = True
            
        elif gesture == GestureType.ONE:
            # ONE from FIST = instant (releasing action)
            # ONE from OPEN = needs 2s (mode switching)
            if self._confirmed_gesture == GestureType.FIST:
                needs_stability = False  # Instant release
            else:
                needs_stability = True   # Mode switch from OPEN
        
        # Apply stability check if needed
        if needs_stability:
            stable_duration = current_time - self._last_gesture_change_time
            recent_gestures = [g for g, t in self._gesture_history if t > current_time - self._gesture_stable_time]
            
            if len(recent_gestures) >= 5 and all(g == gesture for g in recent_gestures):
                if stable_duration >= self._gesture_stable_time:
                    # Confirm gesture after 2 seconds
                    if self._confirmed_gesture != gesture:
                        self._confirmed_gesture = gesture
                        self._last_gesture_change_time = current_time
                        
                        if gesture == GestureType.FIST:
                            print(f"✓ Gesture: FIST (stable for {stable_duration:.1f}s)")
                        elif gesture == GestureType.OPEN:
                            print(f"✓ Mode switch confirmed: OPEN (stable for {stable_duration:.1f}s)")
                        elif gesture == GestureType.ONE:
                            print(f"✓ Mode switch confirmed: ONE (stable for {stable_duration:.1f}s)")
            else:
                # Not stable yet, reset timer
                if self._confirmed_gesture != gesture:
                    self._last_gesture_change_time = current_time
        else:
            # Instant update (ONE from FIST)
            if gesture != self._confirmed_gesture:
                self._confirmed_gesture = gesture
                self._last_gesture_change_time = current_time
                print(f"✓ Mode switch confirmed: ONE (instant)")
        
        # Use confirmed gesture
        self.current_gesture = self._confirmed_gesture
        
        # Switch mode based on CONFIRMED gesture (after 2-second stability)
        if self._confirmed_gesture == GestureType.OPEN:
            # OPEN hand → Wrist Rotation Mode
            self.mode = DetectionMode.WRIST_ROTATION
            self._update_wrist_rotation(lms)
            
        elif self._confirmed_gesture == GestureType.ONE:
            # ONE finger → 3-Area Mode
            self.mode = DetectionMode.THREE_AREA
            self._update_three_area(lms)
            
        elif self._confirmed_gesture == GestureType.FIST:
            # FIST works in both modes (just send gesture)
            pass
        
        # Send UDP messages
        self._send_udp()
        
        # Return current state
        if self.mode == DetectionMode.WRIST_ROTATION:
            return self.mode, self._confirmed_gesture, self.wrist_position, self.wrist_angle
        elif self.mode == DetectionMode.THREE_AREA:
            return self.mode, self._confirmed_gesture, self.area_position, None
        else:
            return DetectionMode.UNKNOWN, self._confirmed_gesture, 0, None
    
    def _detect_gesture(self, lms: np.ndarray) -> GestureType:
        """
        Detect what gesture the hand is making
        Priority: ONE > OPEN > FIST
        """
        wrist = lms[WRIST]
        
        # Check each finger if extended
        def is_finger_extended(tip_idx, pip_idx):
            tip = lms[tip_idx]
            pip = lms[pip_idx]
            tip_dist = _distance(wrist, tip)
            pip_dist = _distance(wrist, pip)
            return tip_dist > pip_dist * 1.15
        
        index_ext = is_finger_extended(INDEX_TIP, INDEX_PIP)
        middle_ext = is_finger_extended(MIDDLE_TIP, MIDDLE_PIP)
        ring_ext = is_finger_extended(RING_TIP, RING_PIP)
        pinky_ext = is_finger_extended(PINKY_TIP, PINKY_PIP)
        
        # Count extended fingers (excluding thumb)
        extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])
        
        # Priority 1: ONE finger (only index extended)
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return GestureType.ONE
        
        # Priority 2: OPEN hand (2+ fingers)
        if extended_count >= 2:
            return GestureType.OPEN
        
        # Priority 3: FIST (0-1 fingers)
        return GestureType.FIST
    
    def _update_wrist_rotation(self, lms: np.ndarray):
        """Update wrist rotation (4 positions)"""
        # Calculate wrist angle
        wrist = lms[WRIST]
        middle_mcp = lms[9]  # Middle finger MCP
        
        vec = middle_mcp - wrist
        angle = np.degrees(np.arctan2(-vec[1], vec[0]))
        
        # Normalize to 0-180°
        angle = abs(angle) % 360.0
        if angle > 180.0:
            angle = 360.0 - angle
        angle = 180.0 - angle
        angle = np.clip(angle, 0.0, 180.0)
        
        self.wrist_angle = angle
        
        # Map to 4 positions
        if angle < 60.0:
            self.wrist_position = 1      # LEFT
        elif angle < 90.0:
            self.wrist_position = 2      # LEFT-CENTER
        elif angle < 120.0:
            self.wrist_position = 3      # RIGHT-CENTER
        else:
            self.wrist_position = 4      # RIGHT
    
    def _update_three_area(self, lms: np.ndarray):
        """Update 3-area pointing"""
        # Use index finger tip position
        index_tip = lms[INDEX_TIP]
        x = index_tip[0]
        
        # Divide screen into 3 equal areas
        area_width = self.width / 3
        
        if x < area_width:
            self.area_position = 1       # LEFT
        elif x < area_width * 2:
            self.area_position = 2       # CENTER
        else:
            self.area_position = 3       # RIGHT
    
    def _send_udp(self):
        """Send UDP messages based on mode and gesture"""
        if self.sock is None:
            return
        
        try:
            # 1. Send gesture message (if changed)
            gesture_msg = None
            if self.current_gesture == GestureType.FIST:
                gesture_msg = "gesture/zero"
            elif self.current_gesture == GestureType.OPEN:
                gesture_msg = "gesture/five"
            elif self.current_gesture == GestureType.ONE:
                gesture_msg = "gesture/one"
            
            if gesture_msg and gesture_msg != self._last_gesture_msg:
                self.sock.sendto(gesture_msg.encode(), (self.udp_ip, self.udp_port))
                self._last_gesture_msg = gesture_msg
                print(f"📡 UDP: {gesture_msg}")
            
            # 2. Send position message (if changed)
            position_msg = None
            if self.mode == DetectionMode.WRIST_ROTATION and self.wrist_position > 0:
                position_msg = f"area/menu/{self.wrist_position}"
            elif self.mode == DetectionMode.THREE_AREA and self.area_position > 0:
                position_msg = f"area/color/{self.area_position}"
            
            if position_msg and position_msg != self._last_position_msg:
                self.sock.sendto(position_msg.encode(), (self.udp_ip, self.udp_port))
                self._last_position_msg = position_msg
                print(f"📡 UDP: {position_msg}")
                
        except Exception as e:
            print(f"⚠ UDP error: {e}")
    
    def _send_no_hand_udp(self):
        """Send NO HAND message"""
        if self.sock is None:
            return
        
        try:
            # Send to both endpoints
            self.sock.sendto(b"area/menu/0", (self.udp_ip, self.udp_port))
            self.sock.sendto(b"area/color/0", (self.udp_ip, self.udp_port))
            print("📡 UDP: No hand detected")
        except Exception as e:
            print(f"⚠ UDP error: {e}")
    
    def _extract_landmarks(self, hand):
        """Extract landmarks from hand"""
        if hand is None:
            return None
        
        lms = None
        if isinstance(hand, dict):
            lms = hand.get('landmarks')
        elif hasattr(hand, 'landmarks'):
            lms = hand.landmarks
        
        if lms is None:
            return None
        
        lms = np.asarray(lms, dtype=float)
        if lms.shape == (21, 3):
            lms = lms[:, :2]
        if lms.shape != (21, 2):
            return None
        
        return lms
    
    def reset(self):
        """Reset detector"""
        self.mode = DetectionMode.UNKNOWN
        self.current_gesture = GestureType.UNKNOWN
        self._confirmed_gesture = GestureType.UNKNOWN
        self._gesture_history = []
        self._last_gesture_change_time = time.time()
        self.wrist_position = 0
        self.wrist_angle = None
        self.area_position = 0
        self._last_gesture_msg = None
        self._last_position_msg = None
        self._no_hand_start_time = None
        self._no_hand_sent = False
        print("🔄 Detector reset")
    
    def get_info(self):
        """Get current state info"""
        return {
            "mode": self.mode.name,
            "gesture": self.current_gesture.name,
            "wrist_position": self.wrist_position,
            "wrist_angle": self.wrist_angle,
            "area_position": self.area_position,
        }