# 开发历史

从最初的"项目跑不起来"到"7 种手势稳定识别 + 彩色大字显示"经过了 6 轮迭代。每轮记录:**触发原因 / 改了什么 / 验证手段 / 留下的后遗症**(如有)。

## v1 — 原始版本(继承自原项目)

代码使用 `mp.solutions.drawing_utils / mp.solutions.hands / mp.solutions.face_mesh`,即 MediaPipe < 0.10 的旧 Python API。

```python
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
```

**状态**:不能跑 — `AttributeError: module 'mediapipe' has no attribute 'solutions'`。

## v2 — 切换到 MediaPipe Tasks API

**触发**:环境里的 `mediapipe` 是 0.10.35,Google 在 0.10 把 `mp.solutions` 砍了,改用 `mp.tasks.vision.*`。

**改动**:
- 用 `mp.tasks.vision.HandLandmarker` / `FaceLandmarker` 替换 `Hands` / `FaceMesh`
- `RunningMode.VIDEO` + `detect_for_video(mp_image, ts_ms)`
- 模型从 `.tflite` 换成 `.task`(`hand_landmarker.task` / `face_landmarker.task`),首次运行自动下载
- 绘制用 `mp.tasks.vision.drawing_utils.draw_landmarks`,签名略变 — `landmark_list` 现在直接是 `list[NormalizedLandmark]`

**验证**:11 项无头冒烟测试 — 模型加载、合成帧推理、4 个 gesture helper 函数全部通过。

**留下**:`requirements.txt` 加 `mediapipe>=0.10`。

## v3 — 改进手势分类规则

**触发**:用户反馈"动作识别不太准,尤其是点赞"。

**分析**(代码逐行):旧分类器有两个结构性 bug:
- 用**中指 PIP** 当万能参考判 4 指是否弯曲,逻辑错位
- 点赞判定只看"拇指尖 y < 中指 PIP y",非常宽松

**改动**:
- 加 `_finger_curled(landmarks, pip, tip)` 用**每根手指自己的 PIP**
- 点赞要求"拇指尖 y < 拇指 IP y AND < 拇指 MCP y"
- thinking 加"食指必须伸出"约束,排除握拳贴近脸误触发

**验证**:11 项合成姿态测试。

## v4 — 修向上指不触发

**触发**:用户反馈"向上指识别不到"(点赞和 thinking 都稳定)。

**分析**:向上指时拇指常常**斜向上**伸出,虽然不是严格垂直,但 TIP y < IP y < MCP y 也成立 — 旧代码的 `not thumb_extended_up` 直接把指向手势卡住。

**改动**:把"thumb 是否向上"的反向判断换成**对称的"topmost"判别**:
- `thumb_is_topmost` = 拇指尖明显高于食指尖
- `index_is_topmost` = 食指尖明显高于拇指尖

二者互斥,各自对应不同手势。`TOPMOST_TOLERANCE = 0.01` 防止"差不多高"时左右横跳。

**验证**:13 项测试,新增"拇指斜向上 45°"和"拇指水平"两个真实失败姿态。

## v5 — 扩展到 7 种手势 + 彩色中央文字

**触发**:用户希望看到所有识别结果(原来只对配图的 4 种有视觉反馈)。

**改动**:
- 加 `GESTURE_COLORS` 字典(8 种颜色)
- 加 `draw_gesture_label()` 在画面中央画大号彩色文字 + 半透明黑底
- `classify_gesture` 扩展到 7 种:`FIST` / `OPEN_PALM` / `PEACE` / `THUMBS_DOWN` 新加入
- main() 改成总是画中央标签,只在 `IMAGE_PATHS` 里的 4 种才显示配图

**验证**:12 项测试 + 8 种颜色的渲染冒烟测试。

**留下**:判定顺序变成"FIST 先于 THUMBS_UP" — 这埋了 v6 的坑。

## v6 — 稳定性三轮修复

**触发**:用户反馈三个稳定性问题:
1. 拇指向下误识为向上指
2. 握拳不稳且误识为点赞
3. thinking 和 pointing 互窜

### v6.1 — THUMBS_DOWN 不再要求食指弯曲

旧:THUMBS_DOWN 需要 `thumb_pointing_down AND middle_ring_pinky_curled AND index_curled`。真实场景做拇指向下时食指常常半伸出,`index_curled` 失败 → THUMBS_DOWN 不命中 → 落到 POINTING(因为食指伸出且成为最高点)。

修:删掉 `index_curled`。

### v6.2 — THUMBS_UP 要求"显著高于"+ 重排判定顺序

旧顺序:FIST → OPEN_PALM → PEACE → THUMBS_UP → ...。问题:真实握拳时拇指搭在蜷起的手指上,`tip.y < ip.y < mcp.y` 也成立,**但因为 FIST 排前面**,FIST 先匹配。所以 v5 反而没爆出 v6 的 bug —— 等用户开始连续握拳(v6 测试时)才发现。

修:
- 把 THUMBS_DOWN / THUMBS_UP 提到 FIST **前面**
- THUMBS_UP 加 `THUMBS_UP_GAP = 0.10`(归一化坐标) — 拇指尖必须比食指尖**高 ≥ 0.10** 才算真点赞

### v6.3 — 平滑器换成滑动窗口多数投票

旧:`GestureSmoother(hold_frames=5)` 要求**连续** 5 帧同一标签才切换。任何一帧不一致 → 计数清零 → thinking ↔ POINTING 边界抖动永远切不过去。

修:6 帧窗口 + ≥ 4 帧一致就切(67% 多数)。`check_thinking_gesture` 的距离阈值也从 6% 放宽到 8%,食指判定改成"tip 高于 MCP"(允许钩形)。

**验证**:16 项测试,覆盖三个修复点的所有失败姿态。

---

## 时间线小结

| 版本 | 关键改动 | 测试数 |
|---|---|---|
| v1 | 原始(不可用) | — |
| v2 | mp.solutions → mp.tasks.vision | 11 |
| v3 | 用每根手指自己的 PIP | 11 |
| v4 | 对称 topmost 判别 | 13 |
| v5 | 7 种手势 + 彩色文字 | 12 |
| v6 | 三个稳定性修复 | 16 |