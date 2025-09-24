import cv2
import numpy as np


def center(x, y, w, h):
    return (x + w // 2, y + h // 2)


# 初始化参数
Vehicles_L = 0
Vehicles_R = 0
height_Lline = 550  # 左车道检测线y坐标
height_Rline = 400  # 右车道检测线y坐标
min_w_L, min_h_L = 90, 90
min_w_R, min_h_R = 25, 25
min_area_L = 5000
min_area_R = 900
tracking_dist_L = 25
tracking_dist_R = 40
aspect_ratio_range_L = (0.8, 3.2)
aspect_ratio_range_R = (0.7, 4.0)

# 跨帧跟踪列表（拆分为左右两个）
prev_cars_L = []  # 左车道前一帧车辆
prev_cars_R = []  # 右车道前一帧车辆

cap = cv2.VideoCapture(r'D:\Desktop\Vision\vision\learnopencv\vid\video.mp4')

# 检测区域x坐标范围（根据实际画面调整）
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
x_left_start = 50
x_left_end = frame_width - 700
x_right_start = frame_width - 600
x_right_end = frame_width - 380

bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理流程保持不变...
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 3)
    fg_mask = bg_sub.apply(blur)
    eroded = cv2.erode(fg_mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    morph = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_cars_L = []  # 当前帧左车道车辆
    current_cars_R = []  # 当前帧右车道车辆

    # 绘制检测线（保持原坐标）
    cv2.line(frame, (x_left_start, height_Lline), (x_left_end, height_Lline), (0, 255, 0), 3)
    cv2.line(frame, (x_right_start, height_Rline), (x_right_end, height_Rline), (0, 255, 0), 3)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = w / max(h, 1)
            # 根据x坐标分配到对应车道
        if x_left_start <= x <= x_left_end:
            if (w >= min_w_L and h >= min_h_L and
                    area >= min_area_L and
                    aspect_ratio_range_L[0] < aspect < aspect_ratio_range_L[1]):
                cx_L, cy_L = center(x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                current_cars_L.append((cx_L, cy_L))
        elif x_right_start <= x <= x_right_end:
            if (w >= min_w_R and h >= min_h_R and
                    area >= min_area_R and
                    aspect_ratio_range_R[0] < aspect < aspect_ratio_range_R[1]):
                cx_R, cy_R = center(x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                current_cars_R.append((cx_R, cy_R))

    # 左车道计数（从上方移动到下方）
    for (cx_L, cy_L) in current_cars_L:
        if cy_L > height_Lline:  # 当前在检测线下方
            for (px_L, py_L) in prev_cars_L:
                if np.hypot(cx_L - px_L, cy_L - py_L) < tracking_dist_L and py_L <= height_Lline:
                    Vehicles_L += 1
                    break

    # 右车道计数（从下方移动到上方）
    for (cx_R, cy_R) in current_cars_R:
        if cy_R < height_Rline:  # 当前在检测线上方
            for (px_R, py_R) in prev_cars_R:
                if np.hypot(cx_R - px_R, cy_R - py_R) < tracking_dist_R and py_R >= height_Rline:
                    Vehicles_R += 1
                    break

    # 更新跟踪列表
    prev_cars_L = current_cars_L.copy()
    prev_cars_R = current_cars_R.copy()

    # 显示统计信息
    cv2.putText(frame, f"Vehicle counts_L: {Vehicles_L}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(frame, f"Vehicle counts_R: {Vehicles_R}", (frame_width - 500, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow('Traffic Monitoring', frame)
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()