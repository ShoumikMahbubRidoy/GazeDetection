from HandTracker import HandTracker
from HandTrackerRenderer import HandTrackerRenderer
import cv2
import numpy as np

def get_pinch_area(hand, frame_shape, cols=3, rows=2):
    """
    ピンチ位置（親指と人差し指の加重平均）からエリア番号を返す
    人差し指の重み: 0.8、親指の重み: 0.2
    """
    h, w = frame_shape[:2]
    thumb_tip = hand.landmarks[4]      # 親指先端
    index_tip = hand.landmarks[8]      # 人差し指先端
    
    # 人差し指寄りの加重平均を計算
    pinch_center_x = thumb_tip[0] * 0.2 + index_tip[0] * 0.8
    pinch_center_y = thumb_tip[1] * 0.2 + index_tip[1] * 0.8
    
    x = pinch_center_x / w
    y = pinch_center_y / h
    col = min(int(x * cols), cols - 1)
    row = min(int(y * rows), rows - 1)
    area = row * cols + col + 1
    return area, (col, row), (pinch_center_x, pinch_center_y)

def is_pinching(hand, threshold=50):
    """
    親指と人差し指がつまんでいる状態かを判定
    """
    thumb_tip = hand.landmarks[4]      # 親指先端
    index_tip = hand.landmarks[8]      # 人差し指先端
    
    # 2点間の距離を計算
    dist_thumb_index = np.linalg.norm(thumb_tip - index_tip)
    
    # 距離が閾値以下ならつまんでいる
    return dist_thumb_index < threshold 

tracker = HandTracker(
    solo=False,
    lm_model="full",              
    lm_nb_threads=2,
    internal_fps=15,
    resolution="ultra",
    internal_frame_height=720,    
    pd_score_thresh=0.6,          
    lm_score_thresh=0.6,          
    single_hand_tolerance_thresh=10
)
renderer = HandTrackerRenderer(tracker=tracker)

print("Hand area detection started. Press 'q' to quit.\n")
print("Screen divided into 6 areas (3 cols × 2 rows)")
print("┌─────┬─────┬─────┐")
print("│  1  │  2  │  3  │")
print("├─────┼─────┼─────┤")
print("│  4  │  5  │  6  │")
print("└─────┴─────┴─────┘\n")

COLS = 3
ROWS = 2

# 前フレームの状態を記録（両手分）
prev_pinch_state = {0: False, 1: False}
prev_area = {0: None, 1: None}

while True:
    frame, hands, bag = tracker.next_frame()
    if frame is None:
        break
    
    h, w = frame.shape[:2]
    col_width = w // COLS
    row_height = h // ROWS
    
    # グリッド線を描画
    for i in range(1, COLS):
        cv2.line(frame, (col_width * i, 0), (col_width * i, h), (0, 255, 0), 2)
    for i in range(1, ROWS):
        cv2.line(frame, (0, row_height * i), (w, row_height * i), (0, 255, 0), 2)
    
    # エリア番号を描画
    for row in range(ROWS):
        for col in range(COLS):
            area_num = row * COLS + col + 1
            x_pos = col * col_width + col_width // 2
            y_pos = row * row_height + row_height // 2
            cv2.putText(frame, str(area_num), (x_pos - 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    # 現在のフレームの状態
    current_pinch_state = {0: False, 1: False}
    current_area = {0: None, 1: None}
    
    # 手が検出された場合
    hand_info_y = 80
    for idx, hand in enumerate(hands):
        if hand.landmarks is not None:
            hand_label = "Right" if hand.handedness > 0.5 else "Left"
            color = (0, 0, 255) if hand_label == "Right" else (255, 0, 0)
            
            # 常にピンチ位置（親指と人差し指の加重平均）でエリア判定
            area, (col, row), (center_x, center_y) = get_pinch_area(hand, frame.shape, COLS, ROWS)
            current_area[idx] = area
            
            # ピンチジェスチャー検出
            current_pinch = is_pinching(hand, threshold=80)
            current_pinch_state[idx] = current_pinch
            
            # 中心位置を丸で描画
            center_x_int = int(center_x)
            center_y_int = int(center_y)
            cv2.circle(frame, (center_x_int, center_y_int), 15, color, -1)
            cv2.putText(frame, hand_label, (center_x_int - 30, center_y_int - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # イベント検出
            # 1. ピンチイベント（False → True）
            if not prev_pinch_state.get(idx, False) and current_pinch:
                print(f"🎯 {hand_label} Hand PINCHED!")
            
            # 2. エリア変更イベント
            if prev_area.get(idx) is not None and prev_area[idx] != area:
                print(f"📍 {hand_label} Hand moved to Area {area}")
            
            # 画面に表示
            pinch_status = "PINCH!" if current_pinch else ""
            cv2.putText(frame, f"{hand_label}: AREA {area} {pinch_status}", (50, hand_info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # 親指と人差し指を常に可視化
            thumb_tip = hand.landmarks[4].astype(int)
            index_tip = hand.landmarks[8].astype(int)
            cv2.circle(frame, tuple(thumb_tip), 6, (255, 255, 0), -1)
            cv2.circle(frame, tuple(index_tip), 6, (255, 255, 0), -1)
            
            # ピンチしている時は中点を強調表示
            if current_pinch:
                cv2.circle(frame, (center_x_int, center_y_int), 12, (0, 255, 255), 3)
            
            hand_info_y += 60
    
    # 前フレームの状態を更新
    prev_pinch_state = current_pinch_state.copy()
    prev_area = current_area.copy()
    
    # レンダリング
    frame = renderer.draw(frame, hands, bag)
    key = renderer.waitKey(delay=1)
    
    if key == ord('q') or key == 27:
        break

renderer.exit()
tracker.exit()
print("\nTracking stopped.")