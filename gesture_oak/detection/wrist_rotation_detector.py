# src/gesture_oak/detection/wrist_rotation_detector.py
"""
SIMPLE & INDEPENDENT Wrist Rotation Detector + UDP
- Hand state (Fisted/Open) = for display only
- Position (1/2/3/4) = ALWAYS updates based on angle (independent!)
- UDP Messages:
  * "gesture/five" - OPEN hand
  * "gesture/zero" - FISTED hand
  * "area/menu/1" to "area/menu/4" - Position
  * "area/menu/0" - NO HAND
"""
from enum import Enum, auto
from collections import deque
from typing import Optional, Tuple, List
import numpy as np
import math
import socket
import time

__all__ = ["WristRotationDetector", "HandState", "RotationPosition", "__version__", "WRIST"]
__version__ = "3.0.0"

# MediaPipe landmarks
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


class HandState(Enum):
    UNKNOWN = auto()
    FISTED = auto()
    OPEN = auto()


class RotationPosition(Enum):
    NONE = 0
    LEFT_FAR = 1    # 0-60°
    LEFT_NEAR = 2   # 60-90°
    RIGHT_NEAR = 3  # 90-120°
    RIGHT_FAR = 4   # 120-180°


def _distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def _median(seq: List[float]):
    if not seq:
        return None
    return float(np.median(np.asarray(seq, dtype=float)))


class WristRotationDetector:
    """
    Simple detector with INDEPENDENT hand state and position
    """

    def __init__(self, udp_ip="192.168.0.10", udp_port=9000):
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
        
        self._last_sent_state = None
        self._last_sent_position = None
        
        # NO HAND delay (3 seconds before sending area/menu/0)
        self._no_hand_start_time = None
        self._no_hand_delay_seconds = 3.0
        self._no_hand_sent = False
        
        # Angle smoothing
        self.angle_buffer = deque(maxlen=5)
        
        # Hand state
        self.state = HandState.UNKNOWN
        self._open_streak = 0
        self._fist_streak = 0
        
        # Position (ALWAYS updated!)
        self._current_angle: Optional[float] = None
        self._current_position: RotationPosition = RotationPosition.NONE
        
        # Calibration
        self._neutral_angle: Optional[float] = None
        self._calib_samples = []
        self._calibrated = False
        
        # Debug
        self._finger_states = {}
        
        # Output
        self.result_file = "result.txt"

    def update(self, hand) -> Tuple[RotationPosition, Optional[float], HandState]:
        """
        Main update - TWO INDEPENDENT THINGS:
        1. Hand state (for display)
        2. Position (ALWAYS based on angle)
        """
        # Extract landmarks
        lms, confidence = self._extract_landmarks(hand)
        
        if lms is None or confidence < 0.3:
            # NO HAND detected - start timer if not already started
            if self._no_hand_start_time is None:
                self._no_hand_start_time = time.time()
                print("⏳ Hand lost, waiting 3 seconds...")
            
            # Check if 3 seconds have passed
            elapsed = time.time() - self._no_hand_start_time
            if elapsed >= self._no_hand_delay_seconds and not self._no_hand_sent:
                # Send NO HAND message after delay
                self._write_result("NO_HAND", 0, 0.0)
                self._send_udp(no_hand=True)
                self._no_hand_sent = True
            
            return RotationPosition.NONE, None, HandState.UNKNOWN
        
        # HAND DETECTED - reset NO HAND timer
        if self._no_hand_start_time is not None:
            print("✓ Hand back in view")
        self._no_hand_start_time = None
        self._no_hand_sent = False

        # ============ 1. HAND STATE (for display) ============
        measured_state = self._detect_hand_state(lms)
        self._update_hand_state_with_hysteresis(measured_state)

        # ============ 2. ANGLE & POSITION (INDEPENDENT!) ============
        raw_angle = self._calculate_wrist_angle(lms)
        if raw_angle is None:
            return self._finalize()

        # Calibration
        if not self._calibrated:
            self._calib_samples.append(raw_angle)
            if len(self._calib_samples) >= 10:
                self._neutral_angle = _median(self._calib_samples)
                self._calibrated = True
                print(f"✓ Calibrated to {self._neutral_angle:.1f}°")

        # Apply calibration
        if self._neutral_angle is not None:
            angle = raw_angle + (90.0 - self._neutral_angle)
            angle = float(np.clip(angle, 0.0, 180.0))
        else:
            angle = raw_angle

        # Smooth angle
        self.angle_buffer.append(angle)
        smoothed_angle = _median(list(self.angle_buffer)) or angle
        self._current_angle = smoothed_angle

        # ALWAYS UPDATE POSITION (independent of hand state!)
        self._current_position = self._angle_to_position(smoothed_angle)

        # Write to file and send UDP (REAL-TIME for hand present)
        self._write_result(self.state.name, self._current_position.value, smoothed_angle)
        self._send_udp()

        return self._current_position, smoothed_angle, self.state

    def _angle_to_position(self, angle: float) -> RotationPosition:
        """
        Simple direct mapping - NO hysteresis, NO conditions
        Just pure angle → position
        """
        if angle < 60.0:
            return RotationPosition.LEFT_FAR      # Position 1
        elif angle < 90.0:
            return RotationPosition.LEFT_NEAR     # Position 2
        elif angle < 120.0:
            return RotationPosition.RIGHT_NEAR    # Position 3
        else:
            return RotationPosition.RIGHT_FAR     # Position 4

    def _detect_hand_state(self, lms: np.ndarray) -> HandState:
        """
        BALANCED dual-method detection:
        Method 1: Distance ratio (stricter)
        Method 2: Finger spread (for slightly open hands)
        Both must agree for OPEN
        """
        wrist = lms[WRIST]
        
        # Get palm size
        palm_width = _distance(lms[INDEX_MCP], lms[PINKY_MCP])
        if palm_width < 1e-6:
            palm_width = 50.0
        
        # METHOD 1: Distance ratio (BALANCED threshold)
        def is_finger_extended(tip_idx, mcp_idx):
            tip_dist = _distance(wrist, lms[tip_idx])
            mcp_dist = _distance(wrist, lms[mcp_idx])
            if mcp_dist < 1e-6:
                return False
            ratio = tip_dist / mcp_dist
            # BALANCED - 1.2 (20% farther)
            return ratio > 1.2
        
        # Check all fingers
        thumb_open = is_finger_extended(THUMB_TIP, THUMB_MCP)
        index_open = is_finger_extended(INDEX_TIP, INDEX_MCP)
        middle_open = is_finger_extended(MIDDLE_TIP, MIDDLE_MCP)
        ring_open = is_finger_extended(RING_TIP, RING_MCP)
        pinky_open = is_finger_extended(PINKY_TIP, PINKY_MCP)
        
        extended_count = sum([thumb_open, index_open, middle_open, ring_open, pinky_open])
        
        # METHOD 2: Finger spread
        # When fisted: fingertips bunched together
        # When open: fingertips spread apart
        tips = [lms[INDEX_TIP], lms[MIDDLE_TIP], lms[RING_TIP], lms[PINKY_TIP]]
        total_spread = sum(_distance(tips[i], tips[i+1]) for i in range(len(tips)-1))
        spread_ratio = total_spread / max(palm_width, 1e-6)
        
        # METHOD 3: Check if all fingertips are close to palm center
        # Fisted hand: all tips near palm
        palm_center_x = (lms[INDEX_MCP][0] + lms[PINKY_MCP][0]) / 2
        palm_center_y = (lms[INDEX_MCP][1] + lms[PINKY_MCP][1]) / 2
        palm_center = np.array([palm_center_x, palm_center_y])
        
        tips_near_palm = 0
        for tip_idx in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
            dist_to_palm = _distance(palm_center, lms[tip_idx])
            if dist_to_palm < palm_width * 0.8:  # Tips very close to palm
                tips_near_palm += 1
        
        self._finger_states = {
            'thumb': thumb_open,
            'index': index_open,
            'middle': middle_open,
            'ring': ring_open,
            'pinky': pinky_open
        }
        
        # DECISION LOGIC:
        # FISTED if: ALL fingers close to palm (tight fist)
        if tips_near_palm >= 3:
            return HandState.FISTED
        
        # OPEN if: 2+ fingers extended OR good spread
        good_spread = spread_ratio > 1.0
        is_open = (extended_count >= 2) or (extended_count >= 1 and good_spread)
        
        return HandState.OPEN if is_open else HandState.FISTED

    def _update_hand_state_with_hysteresis(self, measured: HandState):
        """Minimal hysteresis for hand state"""
        if measured == HandState.OPEN:
            self._open_streak += 1
            self._fist_streak = 0
        else:
            self._fist_streak += 1
            self._open_streak = 0

        # Only 1 frame needed to change (minimal hysteresis)
        if self.state in (HandState.UNKNOWN, HandState.FISTED):
            if self._open_streak >= 1:
                self.state = HandState.OPEN

        if self.state in (HandState.UNKNOWN, HandState.OPEN):
            if self._fist_streak >= 1:
                self.state = HandState.FISTED

    def _calculate_wrist_angle(self, lms: np.ndarray) -> Optional[float]:
        """Calculate wrist rotation angle"""
        wrist = lms[WRIST]
        middle_mcp = lms[MIDDLE_MCP]
        vec = middle_mcp - wrist
        if np.linalg.norm(vec) < 1e-3:
            return None
        angle = math.degrees(math.atan2(-vec[1], vec[0]))
        angle = abs(angle) % 360.0
        if angle > 180.0:
            angle = 360.0 - angle
        angle = 180.0 - angle
        return float(np.clip(angle, 0.0, 180.0))

    def _extract_landmarks(self, hand):
        """Extract landmarks"""
        lms = None
        confidence = 1.0
        if isinstance(hand, dict):
            lms = hand.get('landmarks')
            confidence = float(hand.get('lm_score', 1.0))
        else:
            if hasattr(hand, 'landmarks'):
                lms = getattr(hand, 'landmarks')
            if hasattr(hand, 'lm_score'):
                confidence = float(getattr(hand, 'lm_score'))
        if lms is None:
            return None, 0.0
        lms = np.asarray(lms, dtype=float)
        if lms.shape == (21, 3):
            lms = lms[:, :2]
        if lms.shape != (21, 2):
            return None, 0.0
        return lms, confidence

    def _write_result(self, state_name: str, position: int, angle: float):
        """Write to result.txt"""
        try:
            with open(self.result_file, 'w', encoding='utf-8') as f:
                f.write(f"State: {state_name}\n")
                f.write(f"Position: {position}\n")
                f.write(f"Angle: {angle:.1f}°\n")
        except Exception:
            pass

    def _send_udp(self, no_hand=False):
        """
        Send UDP messages:
        - "gesture/five" for OPEN
        - "gesture/zero" for FISTED
        - "area/menu/1-4" for position
        - "area/menu/0" for NO HAND
        """
        if self.sock is None:
            return
        
        try:
            if no_hand:
                # NO HAND detected
                msg = "area/menu/0"
                self.sock.sendto(msg.encode(), (self.udp_ip, self.udp_port))
                self._last_sent_position = msg
                self._last_sent_state = None
                print(f"📡 UDP: {msg}")
                return
            
            # Send hand state (only when it changes)
            if self.state == HandState.OPEN:
                state_msg = "gesture/five"
            elif self.state == HandState.FISTED:
                state_msg = "gesture/zero"
            else:
                state_msg = None
            
            if state_msg and state_msg != self._last_sent_state:
                self.sock.sendto(state_msg.encode(), (self.udp_ip, self.udp_port))
                self._last_sent_state = state_msg
                print(f"📡 UDP: {state_msg}")
            
            # Send position (only when it changes)
            pos_msg = f"area/menu/{int(self._current_position.value)}"
            if pos_msg != self._last_sent_position:
                self.sock.sendto(pos_msg.encode(), (self.udp_ip, self.udp_port))
                self._last_sent_position = pos_msg
                print(f"📡 UDP: {pos_msg}")
                
        except Exception as e:
            print(f"⚠ UDP send error: {e}")

    def _finalize(self) -> Tuple[RotationPosition, Optional[float], HandState]:
        return self._current_position, self._current_angle, self.state

    def reset(self):
        """Reset"""
        self.angle_buffer.clear()
        self.state = HandState.UNKNOWN
        self._open_streak = 0
        self._fist_streak = 0
        self._current_angle = None
        self._current_position = RotationPosition.NONE
        self._neutral_angle = None
        self._calib_samples.clear()
        self._calibrated = False
        self._no_hand_start_time = None
        self._no_hand_sent = False
        self._last_sent_state = None
        self._last_sent_position = None
        print("Reset!")

    def get_state_info(self):
        """Get state info"""
        return {
            "state": self.state.name,
            "hand_state": self.state.name,
            "angle": self._current_angle,
            "position": int(self._current_position.value),
            "position_name": self._position_name(self._current_position),
            "calibrated": self._calibrated,
            "finger_states": self._finger_states.copy(),
        }

    def _position_name(self, pos: RotationPosition) -> str:
        names = {
            RotationPosition.NONE: "NONE",
            RotationPosition.LEFT_FAR: "Position 1 (LEFT)",
            RotationPosition.LEFT_NEAR: "Position 2 (LEFT-CENTER)",
            RotationPosition.RIGHT_NEAR: "Position 3 (RIGHT-CENTER)",
            RotationPosition.RIGHT_FAR: "Position 4 (RIGHT)",
        }
        return names.get(pos, "NONE")