# src/gesture_oak/detection/hand_detector.py
#!/usr/bin/env python3
import os
import sys
import re
from pathlib import Path
import numpy as np
import depthai as dai
import cv2
import marshal
from string import Template

from ..utils import mediapipe_utils as mpu
from ..utils.FPS import FPS


def _find_asset(relpath: str) -> str:
    """
    Return a filesystem path to an asset whether running from source or PyInstaller onefile.
    relpath examples:
      - "models/palm_detection_sh4.blob"
      - "src/gesture_oak/utils/template_manager_script_solo.py"
    """
    # 1) If running under PyInstaller onefile, files are extracted under _MEIPASS
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        p = Path(meipass) / relpath
        if p.exists():
            return str(p)

    # 2) Try relative to repo root guessed from this file
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        p = parent.parent.parent.parent / relpath  # step up to project root then into rel
        if p.exists():
            return str(p)

    # 3) Finally try relative to CWD
    p = Path(relpath)
    if p.exists():
        return str(p)

    # If we reach here, we return best-effort string (caller may still handle)
    return str(Path(relpath))


class HandDetector:
    """
    MediaPipe-based hand detector for OAK-D using IR mono cameras.

    Key points:
      - LEFT/RIGHT mono cameras -> StereoDepth for mm depth
      - Light image enhancement for IR
      - Non-blocking host queues
      - Palm & Landmark NNs + postproc script (Script node orchestrated)
      - Depth-aware filtering (distance-aware variance tolerance)
    """

    def __init__(
        self,
        fps: int = 30,
        resolution=(640, 480),
        pd_score_thresh: float = 0.15,
        pd_nms_thresh: float = 0.3,
        use_gesture: bool = True,
        use_rgb: bool = True  # not used in IR path; kept for API compat
    ):
        self.fps_target = fps
        self.resolution = resolution
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_gesture = use_gesture

        # Model/template assets (paths resolved to work in source and PyInstaller)
        self.pd_model = _find_asset("models/palm_detection_sh4.blob")
        self.lm_model = _find_asset("models/hand_landmark_lite_sh4.blob")
        self.pp_model = _find_asset("models/PDPostProcessing_top2_sh1.blob")

        # Pipeline objects / queues
        self.device = None
        self.pipeline = None
        self.q_video = None
        self.q_manager_out = None
        self.q_depth = None

        # Geometry / sizes
        self.pd_input_length = 128
        self.lm_input_length = 224
        self.frame_size = max(resolution)
        self.img_w, self.img_h = resolution
        self.pad_h = (self.frame_size - self.img_h) // 2 if self.frame_size > self.img_h else 0
        self.pad_w = (self.frame_size - self.img_w) // 2 if self.frame_size > self.img_w else 0

        # Runtime stats
        self.fps_counter = FPS()

        # Continuity helpers
        self.last_hand_positions = []
        self.frames_without_detection = 0
        self.max_frames_without_detection = 5

    # ---------------------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------------------
    def setup_pipeline(self) -> dai.Pipeline:
        """Create DepthAI pipeline for hand detection."""
        print("Creating hand detection pipeline...")
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

        # IR mono cameras (OV9282). 400p is native; Depth will produce mm map.
        print("Setting up IR mono cameras for dark environment detection...")
        cam_left = pipeline.createMonoCamera()
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_left.setFps(min(self.fps_target, 30))

        cam_right = pipeline.createMonoCamera()
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_right.setFps(min(self.fps_target, 30))

        # Stereo depth (mm)
        depth = pipeline.createStereoDepth()
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(True)
        depth.setSubpixel(True)  # better precision at range
        cam_left.out.link(depth.left)
        cam_right.out.link(depth.right)

        # Convert mono to RGB888p so downstream matches expected layout
        mono_to_rgb = pipeline.createImageManip()
        mono_to_rgb.initialConfig.setResize(self.img_w, self.img_h)
        mono_to_rgb.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
        cam_left.out.link(mono_to_rgb.inputImage)

        # Depth XLink out
        depth_out = pipeline.createXLinkOut()
        depth_out.setStreamName("depth_out")
        depth.depth.link(depth_out.input)

        # Camera (RGB888p) XLink out
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam_out.input.setQueueSize(2)
        cam_out.input.setBlocking(False)
        mono_to_rgb.out.link(cam_out.input)

        # Manager script node (postproc coordinator)
        manager_script = pipeline.create(dai.node.Script)
        manager_script.setScript(self.build_manager_script())

        # Palm detection preproc
        pre_pd_manip = pipeline.create(dai.node.ImageManip)
        pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length * self.pd_input_length * 3)
        pre_pd_manip.setWaitForConfigInput(True)
        pre_pd_manip.inputImage.setQueueSize(1)
        pre_pd_manip.inputImage.setBlocking(False)
        mono_to_rgb.out.link(pre_pd_manip.inputImage)
        manager_script.outputs['pre_pd_manip_cfg'].link(pre_pd_manip.inputConfig)

        # Palm NN
        pd_nn = pipeline.create(dai.node.NeuralNetwork)
        pd_nn.setBlobPath(self.pd_model)
        try:
            pd_nn.setNumInferenceThreads(2)
        except Exception:
            pass
        pre_pd_manip.out.link(pd_nn.input)

        # Palm postproc NN -> to script
        post_pd_nn = pipeline.create(dai.node.NeuralNetwork)
        post_pd_nn.setBlobPath(self.pp_model)
        pd_nn.out.link(post_pd_nn.input)
        post_pd_nn.out.link(manager_script.inputs['from_post_pd_nn'])

        # Landmark preproc
        pre_lm_manip = pipeline.create(dai.node.ImageManip)
        pre_lm_manip.setMaxOutputFrameSize(self.lm_input_length * self.lm_input_length * 3)
        pre_lm_manip.setWaitForConfigInput(True)
        pre_lm_manip.inputImage.setQueueSize(1)
        pre_lm_manip.inputImage.setBlocking(False)
        mono_to_rgb.out.link(pre_lm_manip.inputImage)
        manager_script.outputs['pre_lm_manip_cfg'].link(pre_lm_manip.inputConfig)

        # Landmark NN -> back to script
        lm_nn = pipeline.create(dai.node.NeuralNetwork)
        lm_nn.setBlobPath(self.lm_model)
        try:
            lm_nn.setNumInferenceThreads(2)
        except Exception:
            pass
        pre_lm_manip.out.link(lm_nn.input)
        lm_nn.out.link(manager_script.inputs['from_lm_nn'])

        # Script -> host
        manager_out = pipeline.create(dai.node.XLinkOut)
        manager_out.setStreamName("manager_out")
        manager_script.outputs['host'].link(manager_out.input)

        print("Pipeline created successfully.")
        return pipeline

    # ---------------------------------------------------------------------
    # Resource-safe Script template (PyInstaller friendly)
    # ---------------------------------------------------------------------
    def build_manager_script(self) -> str:
        """
        Build the Script node code from the embedded template.
        Works from source AND from PyInstaller onefile (reads as package data).
        """
        # Try using importlib.resources (package data) first
        try:
            import importlib.resources as ir
            pkg = "src.gesture_oak.utils"
            with ir.files(pkg).joinpath("template_manager_script_solo.py").open("r", encoding="utf-8") as f:
                raw = f.read()
        except Exception:
            # Fallback to filesystem lookup
            tpl_path = _find_asset("src/gesture_oak/utils/template_manager_script_solo.py")
            with open(tpl_path, "r", encoding="utf-8") as f:
                raw = f.read()

        template = Template(raw)
        code = template.substitute(
            _TRACE1="#",
            _TRACE2="#",
            _pd_score_thresh=self.pd_score_thresh,
            _lm_score_thresh=0.10,  # lenient to keep hands at distance
            _pad_h=self.pad_h,
            _img_h=self.img_h,
            _img_w=self.img_w,
            _frame_size=self.frame_size,
            _crop_w=0,
            _IF_XYZ='"""',
            _IF_USE_HANDEDNESS_AVERAGE='"""',
            _single_hand_tolerance_thresh=15,
            _IF_USE_SAME_IMAGE='"""',
            _IF_USE_WORLD_LANDMARKS='"""',
        )

        # Strip template comment blocks and blank lines
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'\n\s*\n', '\n', code)
        return code

    def connect(self) -> bool:
        """Connect to device and create output queues."""
        try:
            self.pipeline = self.setup_pipeline()
            self.device = dai.Device(self.pipeline)
            print(f"Connected to device: {self.device.getDeviceName()}")
            print(f"USB Speed: {self.device.getUsbSpeed()}")

            # Non-blocking host queues (small sizes to reduce latency)
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=2, blocking=False)
            self.q_manager_out = self.device.getOutputQueue(name="manager_out", maxSize=2, blocking=False)
            self.q_depth = self.device.getOutputQueue(name="depth_out", maxSize=2, blocking=False)
            return True
        except Exception as e:
            print(f"Failed to connect to OAK-D: {e}")
            return False

    # ---------------------------------------------------------------------
    # Result extraction / enhancement
    # ---------------------------------------------------------------------
    def extract_hand_data(self, res, hand_idx):
        """Turn script output into a HandRegion with 2D/3D info if available."""
        hand = mpu.HandRegion()
        hand.rect_x_center_a = res["rect_center_x"][hand_idx] * self.frame_size
        hand.rect_y_center_a = res["rect_center_y"][hand_idx] * self.frame_size
        hand.rect_w_a = hand.rect_h_a = res["rect_size"][hand_idx] * self.frame_size
        hand.rotation = res["rotation"][hand_idx]
        hand.rect_points = mpu.rotated_rect_to_points(
            hand.rect_x_center_a, hand.rect_y_center_a,
            hand.rect_w_a, hand.rect_h_a, hand.rotation
        )
        hand.lm_score = res["lm_score"][hand_idx]
        hand.handedness = res["handedness"][hand_idx]
        hand.label = "right" if hand.handedness > 0.5 else "left"

        # landmarks
        hand.norm_landmarks = np.array(res['rrn_lms'][hand_idx]).reshape(-1, 3)
        hand.landmarks = (np.array(res["sqn_lms"][hand_idx]) * self.frame_size).reshape(-1, 2).astype(np.int32)

        # remove padding if any
        if self.pad_h > 0:
            hand.landmarks[:, 1] -= self.pad_h
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][1] -= self.pad_h
        if self.pad_w > 0:
            hand.landmarks[:, 0] -= self.pad_w
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][0] -= self.pad_w

        if self.use_gesture:
            mpu.recognize_gesture(hand)

        return hand

    def filter_hands_by_depth(self, hands, depth_frame):
        """
        Filter hands using depth with *distance-aware tolerance*.

        Keeps far valid hands (300–2000 mm) and relaxes std tolerance with distance,
        so mid/far hands aren't dropped too aggressively.
        """
        if depth_frame is None or len(hands) == 0:
            return hands

        filtered = []
        dh, dw = depth_frame.shape

        for hand in hands:
            if not hasattr(hand, 'landmarks') or hand.landmarks is None:
                continue

            cx = int(hand.rect_x_center_a * dw / self.img_w)
            cy = int(hand.rect_y_center_a * dh / self.img_h)
            if not (0 <= cx < dw and 0 <= cy < dh):
                continue

            d_center = int(depth_frame[cy, cx])
            if d_center <= 0:
                continue

            # Accept wider working range
            if not (300 <= d_center <= 2000):
                continue

            # ROI grows a bit for far hands to stabilize the mean
            half = 18 if d_center < 1000 else 26
            y1 = max(0, cy - half)
            y2 = min(dh, cy + half)
            x1 = max(0, cx - half)
            x2 = min(dw, cx + half)

            roi = depth_frame[y1:y2, x1:x2]
            vals = roi[roi > 0]
            if len(vals) < 30:
                continue

            avg = float(np.mean(vals))
            std = float(np.std(vals))

            # Distance-aware variance limit:
            # ~80 mm allowed at ~0.8 m → up to ~180+ mm near 2.0 m
            std_limit = 80.0 + 0.08 * max(0.0, (avg - 800.0))
            if std <= std_limit:
                hand.depth = avg
                hand.depth_confidence = max(0.0, 1.0 - (std / max(std_limit, 1.0)))
                filtered.append(hand)

        return filtered

    def enhance_ir_frame(self, frame):
        """Light IR enhancement to keep edges and reduce noise."""
        # input is RGB888p from ImageManip; convert to gray
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # CLAHE + light bilateral keeps features for palm/LM models
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

        # back to 3-channel RGB for any consumers expecting it
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return enhanced_rgb

    # ---------------------------------------------------------------------
    # Main host API
    # ---------------------------------------------------------------------
    def get_frame_and_hands(self):
        """
        Returns: frame (RGB888), hands (list of HandRegion), depth_frame (mm) or None.
        Non-blocking reads; if something is missing this frame, we continue gracefully.
        """
        try:
            self.fps_counter.update()

            # Video frame (prefer non-blocking)
            in_video = self.q_video.tryGet() if self.q_video else None
            if in_video is None:
                return None, [], None
            raw_frame = in_video.getCvFrame()

            frame = self.enhance_ir_frame(raw_frame)

            # Depth (non-blocking)
            depth_frame = None
            in_depth = self.q_depth.tryGet() if self.q_depth else None
            if in_depth is not None:
                depth_frame = in_depth.getFrame()

            # Manager/script results (non-blocking)
            in_res = self.q_manager_out.tryGet() if self.q_manager_out else None
            if in_res is None:
                return frame, [], depth_frame

            res = marshal.loads(in_res.getData());
            hands = []
            lm_scores = res.get("lm_score", [])
            for i in range(len(lm_scores)):
                hand = self.extract_hand_data(res, i)
                hands.append(hand)

            # Depth-based filtering (can disable via env OAK_DEPTH_FILTER=0)
            if depth_frame is not None and len(hands) > 0 and os.environ.get("OAK_DEPTH_FILTER", "1") == "1":
                hands = self.filter_hands_by_depth(hands, depth_frame)



            # continuity bookkeeping (optional)
            if hands:
                self.frames_without_detection = 0
                self.last_hand_positions = [(h.rect_x_center_a, h.rect_y_center_a) for h in hands]
            else:
                self.frames_without_detection += 1

            return frame, hands, depth_frame

        except Exception as e:
            print(f"Error getting frame and hands: {e}")
            return None, [], None

    def close(self):
        """Close the device cleanly."""
        if self.device:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None
        print("Hand detector closed.")
