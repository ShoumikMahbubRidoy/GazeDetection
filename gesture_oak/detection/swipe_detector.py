#!/usr/bin/env python3
import time
import numpy as np
from collections import deque
from enum import Enum
from typing import Optional, Tuple
import socket

__all__ = ["SwipeDetector"]


class SwipeState(Enum):
    IDLE = "idle"
    DETECTING = "detecting"
    VALIDATING = "validating"
    CONFIRMED = "confirmed"


class SwipeDetector:
    """
    Left-to-right swipe detector, robust to variable FPS and fast motion.
    - Works with a single 2D hand-center (pixels), optionally with world/3D center ignored.
    - Uses timestamps (time.time()) to compute velocities → FPS-independent.
    - Sends UDP packet "Swipe" to 192.168.10.10:6001 on confirmed swipe.
    """

    def __init__(
        self,
        buffer_size: int = 18,    # more history helps at long distance
        min_distance: int = 90,   # pixels; smaller travel at 1.5 m
        min_duration: float = 0.20,
        max_duration: float = 2.00,
        min_velocity: float = 35,  # px/s; allow slower far swipes
        max_velocity: float = 900, # px/s; still filter absurd spikes
        max_y_deviation: float = 0.35,
        cooldown: float = 0.80,
    ):
        self.buffer_size = buffer_size
        self.min_distance = float(min_distance)
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)
        self.min_velocity = float(min_velocity)
        self.max_velocity = float(max_velocity)
        self.max_y_deviation = float(max_y_deviation)

        self.state = SwipeState.IDLE
        self.position_buffer = deque(maxlen=buffer_size)  # (x, y)
        self.time_buffer = deque(maxlen=buffer_size)      # t (s)

        self.swipe_start_time: Optional[float] = None
        self.swipe_start_pos: Optional[Tuple[float, float]] = None
        self.last_confirmed_swipe = 0.0
        self.swipe_cooldown = float(cooldown)

        # Stats
        self.total_swipes_detected = 0
        self.false_positives_filtered = 0

        # UDP socket
        try:
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_target = ("192.168.10.10", 6001)
        except Exception:
            self._udp_sock = None
            self._udp_target = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(
        self,
        hand_center: Optional[Tuple[float, float]],
        _world_center: Optional[Tuple[float, float, float]] = None,  # reserved
    ) -> bool:
        """
        Feed a new measurement.
        :param hand_center: (x, y) in pixels; None if no hand this frame
        :return: True if a swipe was just confirmed
        """
        now = time.time()

        if hand_center is None:
            self._reset_detection()
            return False

        self.position_buffer.append(hand_center)
        self.time_buffer.append(now)

        if len(self.position_buffer) < 3:
            return False

        if self.state == SwipeState.IDLE:
            return self._check_start_detection()
        elif self.state == SwipeState.DETECTING:
            return self._process_detection()
        elif self.state == SwipeState.VALIDATING:
            return self._validate_swipe()
        return False

    def get_current_swipe_progress(self) -> Optional[dict]:
        if self.state == SwipeState.IDLE or not self.swipe_start_pos or not self.position_buffer:
            return None

        cur = self.position_buffer[-1]
        cur_t = self.time_buffer[-1]
        dist_x = float(cur[0] - self.swipe_start_pos[0])
        dur = max(cur_t - (self.swipe_start_time or cur_t), 1e-6)
        vel = dist_x / dur

        return {
            "state": self.state.value,
            "distance": dist_x,
            "duration": dur,
            "velocity": vel,
            "progress": min(max(dist_x / self.min_distance, 0.0), 1.0),
        }

    def get_statistics(self) -> dict:
        return {
            "total_swipes_detected": self.total_swipes_detected,
            "false_positives_filtered": self.false_positives_filtered,
            "current_state": self.state.value,
        }

    def reset_statistics(self):
        self.total_swipes_detected = 0
        self.false_positives_filtered = 0

    # ------------------------------------------------------------------ #
    # Internal state machine
    # ------------------------------------------------------------------ #
    def _check_start_detection(self) -> bool:
        # Look at the last 3 points → must be moving right consistently
        x0, x1, x2 = self.position_buffer[-3][0], self.position_buffer[-2][0], self.position_buffer[-1][0]
        if (x1 - x0) > 3 and (x2 - x1) > 3:  # tiny threshold to start tracking
            self.state = SwipeState.DETECTING
            self.swipe_start_time = self.time_buffer[-3]
            self.swipe_start_pos = self.position_buffer[-3]
        return False

    def _process_detection(self) -> bool:
        assert self.swipe_start_time is not None and self.swipe_start_pos is not None

        cur_pos = self.position_buffer[-1]
        cur_time = self.time_buffer[-1]

        # Timeout
        if (cur_time - self.swipe_start_time) > self.max_duration:
            self._reset_detection()
            return False

        # If moved left overall, abort
        if (cur_pos[0] - self.swipe_start_pos[0]) < 0:
            self._reset_detection()
            return False

        # When distance reached, go validate
        if (cur_pos[0] - self.swipe_start_pos[0]) >= self.min_distance:
            self.state = SwipeState.VALIDATING

        return False

    def _validate_swipe(self) -> bool:
        assert self.swipe_start_time is not None and self.swipe_start_pos is not None

        cur_time = self.time_buffer[-1]
        duration = cur_time - self.swipe_start_time

        if duration < self.min_duration:
            return False
        if duration > self.max_duration:
            self._reset_detection()
            return False

        if self._validate_swipe_characteristics():
            # Cooldown gate
            if (cur_time - self.last_confirmed_swipe) > self.swipe_cooldown:
                self.total_swipes_detected += 1
                self.last_confirmed_swipe = cur_time

                # UDP notify (non-blocking)
                if self._udp_sock and self._udp_target:
                    try:
                        self._udp_sock.sendto(b"Swipe", self._udp_target)
                    except Exception:
                        pass

                self._reset_detection()
                return True

        # If validation failed, reset and count as filtered
        self.false_positives_filtered += 1
        self._reset_detection()
        return False

    def _validate_swipe_characteristics(self) -> bool:
        # Use only the portion after swipe_start_time
        times = np.asarray(self.time_buffer, dtype=float)
        poses = np.asarray(self.position_buffer, dtype=float)

        # find start index
        start_idx = np.searchsorted(times, self.swipe_start_time, side="left")
        poses = poses[start_idx:]
        times = times[start_idx:]
        if poses.shape[0] < 3:
            return False

        total_dx = poses[-1, 0] - poses[0, 0]
        total_dy = poses[-1, 1] - poses[0, 1]
        if total_dx < self.min_distance:
            return False

        # Y deviation constraint
        y_dev = abs(total_dy) / max(total_dx, 1e-6)
        if y_dev > self.max_y_deviation:
            return False

        # Average velocity
        duration = max(times[-1] - times[0], 1e-6)
        v = total_dx / duration
        if not (self.min_velocity <= v <= self.max_velocity):
            return False

        # Consistent rightward movement (allow small jitters)
        diffs = np.diff(poses[:, 0])
        if np.sum(diffs < -12.0) > 0:  # any strong backtrack
            return False

        return True

    def _reset_detection(self):
        self.state = SwipeState.IDLE
        self.swipe_start_time = None
        self.swipe_start_pos = None
