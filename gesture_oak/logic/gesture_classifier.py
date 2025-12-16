# src/gesture_oak/logic/gesture_classifier.py
from __future__ import annotations
import math
from typing import Dict, List, Any, Optional
import numpy as np

# MediaPipe indices (your detector follows this convention)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = {
    "thumb": THUMB_TIP,
    "index": INDEX_TIP,
    "middle": MIDDLE_TIP,
    "ring": RING_TIP,
    "pinky": PINKY_TIP,
}
FINGER_PIPS = {
    "thumb": THUMB_IP,   # thumb has IP (no PIP)
    "index": INDEX_PIP,
    "middle": MIDDLE_PIP,
    "ring": RING_PIP,
    "pinky": PINKY_PIP,
}
FINGER_MCPS = {
    "thumb": THUMB_MCP,
    "index": INDEX_MCP,
    "middle": MIDDLE_MCP,
    "ring": RING_MCP,
    "pinky": PINKY_MCP,
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def _np(v) -> np.ndarray:
    return np.asarray(v, dtype=float)

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle ABC in radians. a,b,c are 2D or 3D points.
    """
    ab = _np(a) - _np(b)
    cb = _np(c) - _np(b)
    nab = np.linalg.norm(ab) + 1e-9
    ncb = np.linalg.norm(cb) + 1e-9
    cosang = np.clip(np.dot(ab, cb) / (nab * ncb), -1.0, 1.0)
    return float(math.acos(cosang))

def _choose_landmarks(hand) -> Optional[np.ndarray]:
    """
    Prefer normalized landmarks; fallback to pixel-space.
    Return shape (21, 3).
    """
    if hasattr(hand, "norm_landmarks") and hand.norm_landmarks is not None:
        arr = np.asarray(hand.norm_landmarks)
        if arr.ndim == 2 and arr.shape[0] == 21:
            if arr.shape[1] == 2:
                arr = np.concatenate([arr, np.zeros((21, 1))], axis=1)
            return arr
    if hasattr(hand, "landmarks") and hand.landmarks is not None:
        arr = np.asarray(hand.landmarks)
        if arr.ndim == 2 and arr.shape[0] == 21:
            if arr.shape[1] == 2:
                arr = np.concatenate([arr, np.zeros((21, 1))], axis=1)
            return arr
    return None

def _is_thumb_extended(lm: np.ndarray) -> bool:
    wrist = lm[WRIST]
    tip   = lm[THUMB_TIP]
    ip    = lm[THUMB_IP]
    mcp   = lm[THUMB_MCP]
    cmc   = lm[THUMB_CMC]
    ang = _angle(cmc, mcp, tip)  # bigger ~ straighter
    tip_w = _dist(tip, wrist)
    ip_w  = _dist(ip, wrist)
    return (ang > math.radians(35)) and (tip_w > ip_w * 1.07)

def _is_finger_extended(lm: np.ndarray, finger: str) -> bool:
    wrist = lm[WRIST]
    tip   = lm[FINGER_TIPS[finger]]
    pip   = lm[FINGER_PIPS[finger]]
    mcp   = lm[FINGER_MCPS[finger]]
    tip_w = _dist(tip, wrist)
    pip_w = _dist(pip, wrist)
    dist_ok = tip_w > pip_w * 1.06
    bend = _angle(mcp, pip, tip)
    straight_ok = bend > math.radians(25)
    return dist_ok and straight_ok

def classify_one(hand) -> Dict[str, Any]:
    lm = _choose_landmarks(hand)
    if lm is None:
        return {
            "fingers": {f: False for f in FINGER_ORDER},
            "count": 0,
            "gesture": "unknown",
            "fingers_up_list": [],
            "handedness": getattr(hand, "label", None),
            "confidence": getattr(hand, "lm_score", None),
        }

    fingers = {
        "thumb": _is_thumb_extended(lm),
        "index": _is_finger_extended(lm, "index"),
        "middle": _is_finger_extended(lm, "middle"),
        "ring": _is_finger_extended(lm, "ring"),
        "pinky": _is_finger_extended(lm, "pinky"),
    }
    up = [f for f in FINGER_ORDER if fingers[f]]
    count = len(up)

    # simple set of names
    if count == 0:
        gesture = "fist"
    elif count == 1:
        gesture = f"one ({up[0]})"
    elif count == 2:
        gesture = "two (peace)" if set(up) == {"index", "middle"} else f"two ({'+'.join(up)})"
    elif count == 3:
        gesture = "three (index+middle+ring)" if set(up) == {"index", "middle", "ring"} else "three"
    elif count == 4:
        gesture = "four"
    else:
        gesture = "five (open palm)"

    return {
        "fingers": fingers,
        "count": count,
        "gesture": gesture,
        "fingers_up_list": up,
        "handedness": getattr(hand, "label", None),
        "confidence": getattr(hand, "lm_score", None),
    }

def classify_many(hands: List[Any]) -> List[Dict[str, Any]]:
    return [classify_one(h) for h in (hands or [])]
