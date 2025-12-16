# src/gesture_oak/utils/FPS.py
import time
from collections import deque

# Add the missing 'now' function that HandTracker.py needs
def now():
    """Return current time in milliseconds"""
    return time.perf_counter() * 1000


class FPS:
    """
    Robust FPS tracker.
    - Call start() once at the beginning.
    - Call update() once per processed frame.
    - stop() is optional; if not called, elapsed() uses 'now'.
    Backward-compat shims: get() -> current FPS, get_global() -> average FPS.
    """

    def __init__(self, avg_window: int = 50):
        self._times = deque(maxlen=avg_window)  # per-frame deltas
        self._start_ts = None
        self._first_update_ts = None
        self._last_update_ts = None
        self._stop_ts = None
        self._frame_count = 0

    def start(self):
        now_ts = time.perf_counter()
        self._start_ts = now_ts
        self._first_update_ts = None
        self._last_update_ts = None
        self._stop_ts = None
        self._frame_count = 0
        self._times.clear()
        return self

    def stop(self):
        self._stop_ts = time.perf_counter()
        return self

    def update(self, n: int = 1):
        """
        Call once per successfully processed/displayed frame.
        """
        now_ts = time.perf_counter()
        for _ in range(n):
            if self._first_update_ts is None:
                self._first_update_ts = now_ts
            if self._last_update_ts is not None:
                dt = now_ts - self._last_update_ts
                if dt > 0:
                    self._times.append(dt)
            self._last_update_ts = now_ts
            self._frame_count += 1

    def nb_frames(self) -> int:
        """Alias for frames() - for compatibility"""
        return self._frame_count

    def frames(self) -> int:
        return self._frame_count

    def elapsed(self) -> float:
        """
        Elapsed time between first and last update (not from start()).
        This avoids huge FPS when no frames were processed.
        """
        if self._first_update_ts is None:
            return 0.0
        end = self._stop_ts or self._last_update_ts or time.perf_counter()
        return max(0.0, end - self._first_update_ts)

    def fps(self) -> float:
        """
        Instant/rolling FPS based on recent inter-frame deltas.
        """
        if not self._times:
            return self.avg_fps()  # fall back to avg
        avg_dt = sum(self._times) / len(self._times)
        return (1.0 / avg_dt) if avg_dt > 0 else 0.0

    def avg_fps(self) -> float:
        """
        Global average FPS = total frames / elapsed (first->last update).
        """
        elapsed = self.elapsed()
        return (self._frame_count / elapsed) if elapsed > 0 else 0.0

    def draw(self, img, orig=(10, 30), size=1, color=(0, 255, 0), thickness=2):
        """Draw FPS on image - for compatibility with HandTracker"""
        import cv2
        fps_text = f"FPS: {self.get_global():.1f}"
        cv2.putText(img, fps_text, orig, cv2.FONT_HERSHEY_SIMPLEX, 
                   size, color, thickness)

    # ---- Backward compatibility (keep old calls working) ----
    def get(self) -> float:
        return self.fps()

    def get_global(self) -> float:
        return self.avg_fps()