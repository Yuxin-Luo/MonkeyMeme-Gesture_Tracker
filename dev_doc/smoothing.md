# 时序平滑(`GestureSmoother`)

> 代码位置:[`gesture-tracker.py:350-388`](../../gesture-tracker.py#L350-L388)

## 为什么需要平滑

`classify_gesture` 逐帧独立判定,没有任何前后关联。两个直接后果:

1. **模型偶尔丢帧**:MediaPipe 在某些帧会检测不到手,`hand_landmarks` 为 None,直接 `NEUTRAL`。一帧 NEUTRAL 就让画面跳一下。
2. **边界抖动**:thinking 的指尖到鼻子的距离在阈值附近抖动,某帧过线下一帧不过线,thinking 和 POINTING 来回跳。

不解决这两点,识别结果看起来就像抽风。

## v1 实现(被弃):连续帧计数

```python
class GestureSmoother:
    def __init__(self, hold_frames=5):
        self.count = 0
        self.candidate = "NEUTRAL"
    def update(self, raw):
        if raw == self.candidate: self.count += 1
        else: self.candidate, self.count = raw, 1
        if self.count >= self.hold_frames: self.current = self.candidate
```

**问题**:只要中间有任何一帧不一致,计数就清零重来。如果用户做 thinking 时每 10 帧里有 1 帧指尖距离抖到阈值外,这个 smoother 永远切不过去 — 它要求**连续** 5 帧同一标签。

## 当前实现:滑动窗口多数投票

```python
class GestureSmoother:
    def __init__(self, window_size=6, min_count=4):
        self.history = deque(maxlen=6)   # 最近 6 帧
        self.current = "NEUTRAL"
    def update(self, raw):
        self.history.append(raw)
        most_common = argmax(counts)    # 出现最多的标签
        if counts[most_common] >= 4:     # 6 帧中至少出现 4 次
            self.current = most_common
```

**核心差别**:

| 维度 | 连续计数 | 滑动窗口 |
|---|---|---|
| 容忍单帧噪声 | ❌ 一帧就清零 | ✅ 最多 2/6 帧错仍能维持 |
| 初始响应速度 | ✅ 5 帧(167ms)就能切 | ⚠️ 4 帧(~130ms)切,接近 |
| 阈值边界抖动 | ❌ 永远切不过去 | ✅ 多数帧正确即可 |
| 代价 | O(1) | O(N) 但 N=6 完全无所谓 |

## 调参建议

- `window_size` 调大 → 更稳定但更迟钝(7-8 也行,10+ 没意义)
- `min_count / window_size` 比例:
  - **0.5**:基本只看相对多数,噪声大时容易左右横跳
  - **0.6-0.7** ← 当前位置(4/6 = 67%),平衡
  - **0.8**:更严格,适合需要"绝对确认"的场景(代价是切换慢)

如果想更灵敏:`GestureSmoother(window_size=5, min_count=3)` — 5 帧里 3 帧一致就切,约 100ms。

如果想更稳定:`GestureSmoother(window_size=8, min_count=6)` — 8 帧里 6 帧一致才切,约 200ms。

## 它处理不了的事

- **快速切换手势**:用户在 100ms 内从点赞切到指向,平滑器可能输出两者之间的"混合"窗口结果。这是**设计上**的取舍 — 想要绝对实时响应就别用平滑器。
- **完全脱离画面**:用户把手移开,`raw_gesture` 持续 NEUTRAL,经过窗口后切换到 NEUTRAL,这部分是符合预期的。