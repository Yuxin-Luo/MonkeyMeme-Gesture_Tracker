# Gesture Tracker — 开发文档

基于 MediaPipe 的实时手势识别项目:从摄像头读帧 → 用 MediaPipe Tasks API 检测手部 / 面部关键点 → 用规则化分类器判定 7 种手势 → 在画面中央用对应颜色的大字显示,并对其中 4 种手势配以 meme 图片。

## 快速开始

```bash
# 1. 装依赖(mediapipe>=0.10 是新 Tasks API 必须)
pip install -r requirements.txt

# 2. 首次运行若 models/ 下没有 .task 模型,脚本会自动从 Google 官方
#    模型库下载;也可以手动预下载(见 models/README.md)。
python gesture-tracker.py
```

按 `q` 或 `ESC` 退出。

## 文件结构

```
MonkeyMeme-Gesture_Tracker/
├── gesture-tracker.py         # 主程序(单文件,~430 行)
├── requirements.txt           # 依赖:mediapipe>=0.10 / opencv-python / numpy
├── README.md                  # 用户文档
├── models/                    # MediaPipe .task 模型(首次运行自动下载)
│   ├── hand_landmarker.task   # ~7.5 MB
│   └── face_landmarker.task   # ~3.8 MB
├── *.jpg                      # 四张手势配图(thumbs_up / pointing / thinking / neutral)
└── dev_doc/                   # ← 本目录
    ├── README.md              # 本文件 — 总览与索引
    ├── architecture.md        # 数据流与模块划分
    ├── gesture-classifier.md  # 7 种手势的判定规则
    ├── smoothing.md           # 滑动窗口多数投票平滑器
    ├── development-history.md # 6 轮迭代的来龙去脉
    └── testing.md             # 合成姿态测试策略
```

## 文档索引

| 文档 | 适合什么时候读 |
|---|---|
| [architecture.md](architecture.md) | 第一次看代码,想知道数据怎么从摄像头流到屏幕 |
| [gesture-classifier.md](gesture-classifier.md) | 想调阈值或加新手势 |
| [smoothing.md](smoothing.md) | 发现识别结果抖动 / 想换响应速度 |
| [development-history.md](development-history.md) | 接手这个项目,想了解前因后果 |
| [testing.md](testing.md) | 想加新测试或验证改动 |

## 一句话总结项目

单文件 Python 程序,核心 ≈ 250 行(去掉注释、draw helper、main 模板)。两个外部依赖是大头:MediaPipe(检测)和 OpenCV(显示)。手势分类全部用规则判定 — 没有训练任何模型。