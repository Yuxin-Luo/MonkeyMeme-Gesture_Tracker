# 架构

## 数据流

```
摄像头
  │
  │  cv2.VideoCapture(0).read()
  ▼
BGR 帧
  │
  │  cv2.flip(frame, 1)         ← 镜像,符合"自拍"直觉
  │  cv2.cvtColor(..., BGR2RGB) ← MediaPipe 期望 RGB
  ▼
RGB 帧
  │
  │  mp.Image(SRGB, rgb_frame)  ← 包成 MediaPipe 输入
  │  HandLandmarker.detect_for_video(mp_image, ts_ms)
  │  FaceLandmarker.detect_for_video(mp_image, ts_ms)
  ▼
hand_landmarks: list[NormalizedLandmark] | None  (21 个点 / 0 或 1 只手)
face_landmarks: list[NormalizedLandmark] | None  (478 个点 / 0 或 1 张脸)
  │
  │  if hand_landmarks and face_landmarks:
  │      check_thinking_gesture(...)     ← 独立判定,优先级最高
  │  if raw_gesture == "NEUTRAL":
  │      classify_gesture(hand_landmarks) ← 7 种手势规则
  ▼
raw_gesture: str
  │
  │  GestureSmoother.update(raw_gesture)  ← 滑动窗口多数投票
  ▼
current_gesture: str  (稳定的输出标签)
  │
  │  draw_gesture_label(frame, current_gesture, ...)  ← 中央彩色文字
  │  draw_hand_landmarks(frame, hand_landmarks)       ← 骨骼叠加
  │  if current_gesture in IMAGE_PATHS:
  │      拼接右侧 meme 图片
  ▼
cv2.imshow('Gesture & Image Pairing', output_frame)
```

## 模块布局(`gesture-tracker.py` 自上而下)

| 行段 | 内容 | 备注 |
|---|---|---|
| 1-22 | 导入 + 模型 URL | 模块级常量 |
| 24-72 | MediaPipe Tasks 初始化 | `BaseOptions` / `VisionRunningMode` 等符号 |
| 75-105 | 模型下载 (`ensure_models`) | 首次运行从 Google Storage 拉 |
| 110-138 | 关节索引常量 + 颜色表 | 21 个 landmark 索引集中维护 |
| 143-163 | `IMAGE_PATHS` + `load_and_resize_image` | 配图加载 |
| 168-185 | `_finger_curled` / `_finger_extended` / `_middle_ring_pinky_curled` | 单指判定基础 |
| 190-296 | `classify_gesture` + `check_thinking_gesture` | **核心** |
| 301-345 | `draw_hand_landmarks` + `draw_gesture_label` | OpenCV 绘图 |
| 350-388 | `GestureSmoother` | 时序平滑 |
| 393-470 | `main` | 主循环 |

## 关键依赖

| 包 | 版本 | 作用 |
|---|---|---|
| `mediapipe` | `>=0.10` | **必须**。`mp.tasks.vision.HandLandmarker` / `FaceLandmarker` |
| `opencv-python` | 任意 | 摄像头 + 图像处理 + 显示 |
| `numpy` | 任意 | 距离计算 + 图像拼接 |

## 几个不容易看出来的设计选择

1. **模型路径写死到 `models/` 子目录**,不放进 pip 包。原因:`.task` 文件太大(加起来 11MB),不适合每次 `pip install` 都重下。代码里 `urllib.request.urlretrieve` 在缺文件时静默拉取,首次运行联网、之后离线可用。

2. **`RunningMode.VIDEO` 而不是 `LIVE_STREAM`**。`LIVE_STREAM` 要传 result callback、要起新线程、还要在 Live Stream 模式下单独处理时间戳回退;`VIDEO` 直接 `detect_for_video(image, ts_ms)` 同步调用,代码短一半。我们喂的是从摄像头连续读出的帧,语义上和视频帧没有差别。

3. **手势分类 = 纯规则,不上 ML**。理由:7 种手势用 ~50 行 if/else 就能讲清楚,引入模型反而要标注数据、训练、部署,投入产出比低。规则也容易调。

4. **彩色文字 + 配图 = 双通道反馈**。中央彩色大字永远显示当前识别结果(覆盖所有 7 种),右侧配图只对原来 4 种老手势保留。原因见 [development-history.md](development-history.md#v5-7-种手势--彩色中央文字)。

5. **THINKING 检查独立于 `classify_gesture`**,在主循环里优先判定。原因:THINKING 依赖人脸 landmark(鼻子),逻辑跟"手指形状"完全不同,强行塞进 7 分类器会让代码难读。