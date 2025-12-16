#!/usr/bin/env python3
"""
3-Area Gesture Detector
Detects index finger (ONE) or fisted hand in 3 horizontal areas

Screen Layout:
┌─────────┬─────────┬─────────┐
│ Area 1  │ Area 2  │ Area 3  │
│ 33.33%  │ 33.33%  │ 33.33%  │
└─────────┴─────────┴─────────┘
"""
import numpy as np
import socket
import time
from enum import Enum, auto
from typing import Optional, Tuple, Dict


class GestureType(Enum):
    """Detected gesture types"""
    NONE = 0
    ONE = 1      # Index finger extended
    FIST = 2     # Fisted hand (all fingers closed)


class ThreeAreaDetector:
    """
    Detects index finger (ONE) or fist in 3 horizontal screen areas
    
    Detection Strategy:
    - ONE: Only index finger extended, other fingers closed
    - FIST: All fingers closed (fingertips near palm center)
    
    Reference Points:
    - ONE: Uses index fingertip position
    - FIST: Uses palm center (average of MCP joints)
    """
    
    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
    
    def __init__(self, udp_ip="192.168.0.10", udp_port=9000, debug=False):
        """
        Initialize 3-area detector
        
        Args:
            udp_ip: UDP destination IP address
            udp_port: UDP destination port
            debug: Enable debug output
        """
        # UDP setup
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.debug = debug
        
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
            print(f"✓ UDP enabled: {udp_ip}:{udp_port}")
        except Exception as e:
            print(f"⚠ UDP socket error: {e}")
            self.sock = None
        
        # State tracking
        self._last_sent_gesture = None
        self._last_sent_area = None
        self._current_gesture = GestureType.NONE
        self._current_area = 0
        self._reference_point = None
        
        # NO HAND delay (prevents false "no hand" triggers)
        self._no_hand_start_time = None
        self._no_hand_delay_seconds = 3.0
        self._no_hand_sent = False
        
        # Detection thresholds
        self.fist_threshold = 0.75     # Fingertips must be within 75% of palm size
        self.one_threshold = 1.25      # Index must extend 25% beyond MCP
        self.other_threshold = 1.1     # Other fingers must be below 10% extension
        
        # Smoothing for gesture changes
        self._gesture_history = []
        self._gesture_history_size = 3  # Require 3 consistent frames
    
    def update(self, hand, frame_shape: Tuple[int, int]) -> Tuple[GestureType, int]:
        """
        Update detector with new hand data
        
        Args:
            hand: Hand object with landmarks attribute, or None if no hand
            frame_shape: (height, width) of frame
            
        Returns:
            tuple: (gesture_type, area_number)
                - gesture_type: GestureType enum (NONE, ONE, or FIST)
                - area_number: 1-3 for detected areas, 0 for no detection
        """
        if hand is None or not hasattr(hand, 'landmarks') or hand.landmarks is None:
            return self._handle_no_hand()
        
        # HAND DETECTED - reset NO HAND timer
        if self._no_hand_start_time is not None:
            if self.debug:
                print("✓ Hand back in view")
        self._no_hand_start_time = None
        self._no_hand_sent = False
        
        # Extract landmarks as 2D array
        lms = self._extract_landmarks_2d(hand.landmarks)
        if lms is None:
            return self._handle_no_hand()
        
        # Detect gesture type
        gesture = self._detect_gesture(lms)
        
        # Add to history for smoothing
        self._gesture_history.append(gesture)
        if len(self._gesture_history) > self._gesture_history_size:
            self._gesture_history.pop(0)
        
        # Use most common gesture in history (smoothing)
        if len(self._gesture_history) >= self._gesture_history_size:
            from collections import Counter
            gesture_counts = Counter(self._gesture_history)
            gesture = gesture_counts.most_common(1)[0][0]
        
        self._current_gesture = gesture
        
        # Calculate reference point based on gesture
        if gesture == GestureType.ONE:
            # Use index fingertip position
            self._reference_point = lms[self.INDEX_TIP]
        elif gesture == GestureType.FIST:
            # Use palm center (average of MCP joints)
            palm_points = [
                lms[self.INDEX_MCP], 
                lms[self.MIDDLE_MCP],
                lms[self.RING_MCP], 
                lms[self.PINKY_MCP]
            ]
            self._reference_point = np.mean(palm_points, axis=0)
        else:
            self._reference_point = None
        
        # Determine area from reference point
        if self._reference_point is not None:
            area = self._point_to_area(self._reference_point, frame_shape)
            self._current_area = area
        else:
            area = 0
            self._current_area = 0
        
        # Send UDP messages when state changes
        self._send_udp()
        
        return gesture, area
    
    def _extract_landmarks_2d(self, landmarks) -> Optional[np.ndarray]:
        """
        Extract 2D landmark positions
        
        Args:
            landmarks: Landmark array (can be 21x2 or 21x3)
            
        Returns:
            21x2 numpy array of (x, y) positions, or None if invalid
        """
        try:
            lms = np.asarray(landmarks, dtype=float)
            
            # Handle 3D landmarks (take only x, y)
            if len(lms.shape) == 2 and lms.shape[1] >= 2:
                lms_2d = lms[:, :2]
            elif len(lms.shape) == 1:
                # Flat array, reshape
                lms_2d = lms.reshape(-1, 2)[:21]
            else:
                lms_2d = lms
            
            # Validate shape
            if lms_2d.shape != (21, 2):
                return None
            
            return lms_2d
        
        except Exception as e:
            if self.debug:
                print(f"Error extracting landmarks: {e}")
            return None
    
    def _detect_gesture(self, lms: np.ndarray) -> GestureType:
        """
        Detect if hand is showing ONE (index extended) or FIST (all closed)
        
        Strategy:
        1. Calculate extension ratio for each finger (tip distance vs MCP distance from wrist)
        2. ONE: Index extended (ratio > threshold), others not extended
        3. FIST: All fingertips close to palm center
        
        Args:
            lms: 21x2 array of landmark positions
            
        Returns:
            GestureType enum
        """
        wrist = lms[self.WRIST]
        
        # Helper: Check if finger is extended
        def is_extended(tip_idx: int, mcp_idx: int, threshold: float) -> bool:
            """Check if finger tip extends beyond MCP by threshold ratio"""
            tip_dist = np.linalg.norm(lms[tip_idx] - wrist)
            mcp_dist = np.linalg.norm(lms[mcp_idx] - wrist)
            if mcp_dist < 1e-6:
                return False
            ratio = tip_dist / mcp_dist
            return ratio > threshold
        
        # Check extension for each finger
        index_extended = is_extended(self.INDEX_TIP, self.INDEX_MCP, self.one_threshold)
        middle_extended = is_extended(self.MIDDLE_TIP, self.MIDDLE_MCP, self.other_threshold)
        ring_extended = is_extended(self.RING_TIP, self.RING_MCP, self.other_threshold)
        pinky_extended = is_extended(self.PINKY_TIP, self.PINKY_MCP, self.other_threshold)
        
        if self.debug:
            print(f"Extensions - I:{index_extended} M:{middle_extended} R:{ring_extended} P:{pinky_extended}")
        
        # ONE gesture: ONLY index extended, all others NOT extended
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return GestureType.ONE
        
        # FIST detection: Check if all fingertips are close to palm center
        palm_center = np.mean([
            lms[self.INDEX_MCP], 
            lms[self.MIDDLE_MCP], 
            lms[self.RING_MCP], 
            lms[self.PINKY_MCP]
        ], axis=0)
        
        palm_size = np.linalg.norm(lms[self.INDEX_MCP] - lms[self.PINKY_MCP])
        
        # Count fingertips near palm
        tips_near_palm = 0
        for tip_idx in [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]:
            dist = np.linalg.norm(lms[tip_idx] - palm_center)
            if dist < palm_size * self.fist_threshold:
                tips_near_palm += 1
        
        if self.debug:
            print(f"Tips near palm: {tips_near_palm}/4")
        
        # FIST: At least 3 fingertips near palm
        if tips_near_palm >= 3:
            return GestureType.FIST
        
        return GestureType.NONE
    
    def _point_to_area(self, point: np.ndarray, frame_shape: Tuple[int, int]) -> int:
        """
        Convert point to area number (1, 2, or 3)
        
        Screen is divided into 3 equal horizontal sections:
        - Area 1: 0.00 - 0.33 (left)
        - Area 2: 0.33 - 0.67 (center)
        - Area 3: 0.67 - 1.00 (right)
        
        Args:
            point: (x, y) coordinates
            frame_shape: (height, width) of frame
            
        Returns:
            Area number 1-3
        """
        h, w = frame_shape[:2]
        x, y = point
        
        # Normalize x coordinate (0.0 to 1.0)
        x_norm = np.clip(x / w, 0.0, 1.0)
        
        # Determine area (3 equal horizontal sections)
        if x_norm < 1.0 / 3.0:
            return 1
        elif x_norm < 2.0 / 3.0:
            return 2
        else:
            return 3
    
    def _handle_no_hand(self) -> Tuple[GestureType, int]:
        """
        Handle case when no hand is detected
        
        Returns:
            (GestureType.NONE, 0)
        """
        # Start timer if not already started
        if self._no_hand_start_time is None:
            self._no_hand_start_time = time.time()
            if self.debug:
                print("⏳ Hand lost, waiting 3 seconds...")
        
        # Check if delay has elapsed
        elapsed = time.time() - self._no_hand_start_time
        if elapsed >= self._no_hand_delay_seconds and not self._no_hand_sent:
            # Send NO HAND message
            self._send_udp(no_hand=True)
            self._no_hand_sent = True
        
        self._current_gesture = GestureType.NONE
        self._current_area = 0
        self._reference_point = None
        
        return GestureType.NONE, 0
    
    def _send_udp(self, no_hand=False):
        """
        Send UDP messages for gesture and area
        
        UDP Message Format:
        - Gesture: "gesture/one" or "gesture/zero"
        - Area: "area/color/1-3"
        - No Hand: "area/color/0"
        
        Messages are only sent when values change to reduce network traffic.
        
        Args:
            no_hand: If True, send "no hand" message
        """
        if self.sock is None:
            return
        
        try:
            if no_hand:
                # NO HAND detected
                msg = "area/color/0"
                self.sock.sendto(msg.encode(), (self.udp_ip, self.udp_port))
                self._last_sent_area = msg
                self._last_sent_gesture = None
                print(f"📡 UDP: {msg}")
                return
            
            # Send gesture (only when it changes)
            if self._current_gesture == GestureType.ONE:
                gesture_msg = "gesture/one"
            elif self._current_gesture == GestureType.FIST:
                gesture_msg = "gesture/zero"
            else:
                gesture_msg = None
            
            if gesture_msg and gesture_msg != self._last_sent_gesture:
                self.sock.sendto(gesture_msg.encode(), (self.udp_ip, self.udp_port))
                self._last_sent_gesture = gesture_msg
                print(f"📡 UDP: {gesture_msg}")
            
            # Send area (only when it changes)
            if self._current_area > 0:
                area_msg = f"area/color/{self._current_area}"
                if area_msg != self._last_sent_area:
                    self.sock.sendto(area_msg.encode(), (self.udp_ip, self.udp_port))
                    self._last_sent_area = area_msg
                    print(f"📡 UDP: {area_msg}")
        
        except Exception as e:
            if self.debug:
                print(f"⚠ UDP send error: {e}")
    
    def get_state_info(self) -> Dict:
        """
        Get current state information
        
        Returns:
            Dictionary with current state
        """
        return {
            "gesture": self._current_gesture.name,
            "gesture_value": self._current_gesture.value,
            "area": self._current_area,
            "reference_point": self._reference_point.tolist() if self._reference_point is not None else None,
            "no_hand_delay": self._no_hand_delay_seconds,
            "thresholds": {
                "fist": self.fist_threshold,
                "one": self.one_threshold,
                "other": self.other_threshold
            }
        }
    
    def reset(self):
        """Reset detector state"""
        self._current_gesture = GestureType.NONE
        self._current_area = 0
        self._reference_point = None
        self._no_hand_start_time = None
        self._no_hand_sent = False
        self._last_sent_gesture = None
        self._last_sent_area = None
        self._gesture_history.clear()
        print("3-Area Detector reset!")
