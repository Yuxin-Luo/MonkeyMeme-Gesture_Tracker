# 手势分类器(`classify_gesture`)

> 代码位置:[`gesture-tracker.py:190-296`](../../gesture-tracker.py#L190-L296)
>
> 单独的判定函数,被 `main()` 调用一次/帧。

## 目标标签(7 种)

`THUMBS_UP` · `THUMBS_DOWN` · `POINTING` · `PEACE` · `OPEN_PALM` · `FIST` · `NEUTRAL`

THINKING 不在这个函数里 — 它在 [check_thinking_gesture](../../gesture-tracker.py#L248-L274) 单独判,优先级最高。

## 判定顺序(为什么是这个顺序)

按"信号强度从强到弱"排列。每个手势需要的特征越独特、越不可能和其它手势混淆,就排在前面。

```
1. THUMBS_DOWN   ← 拇指向下是最强信号(拇指尖明显低于腕关节和 MCP)
2. THUMBS_UP     ← 拇指明显是手的最高点(比食指尖高出 0.10 以上)
3. FIST          ← 4 指弯曲 + 拇指既不上也不下
4. OPEN_PALM     ← 4 指全部伸出
5. PEACE         ← 食指 + 中指伸出,无名 / 小指弯曲
6. POINTING      ← 食指伸出 + 拇指不是最高点 + 其余弯曲
7. NEUTRAL       ← 兜底
```

### 为什么不是先 FIST 再 THUMBS_UP?

这是 v6 之前的 bug 之一 — 旧顺序里 FIST 排第一,只要 4 指弯曲 + 拇指"没向下"就 FIST。但真实握拳时拇指常常稍微搭在蜷起的食指上方,`tip.y < ip.y < mcp.y` 也成立,会被后续的 THUMBS_UP 检查**反过来**匹配 — 因为那时 THUMBS_UP 的"topmost"判定太宽松。

修正方法:把 THUMBS_UP / THUMBS_DOWN 提到 FIST **之前**,并把 THUMBS_UP 的"topmost"判定加一个**最小间距**(`THUMBS_UP_GAP = 0.10`)— 拇指尖必须比食指尖高 0.10(归一化坐标,约几十像素)才算真正"topmost"。

## 关键阈值与容差

```python
CURL_TOLERANCE       = 0.00   # 弯曲判定无容差(tip.y 必须严格 > pip.y)
THUMB_DOWN_TOLERANCE = 0.02   # 拇指尖必须明显低于腕关节和 MCP
THUMBS_UP_TOLERANCE  = 0.03   # 拇指尖必须明显低于自己的 IP/MCP
THUMBS_UP_GAP        = 0.10   # 拇指尖必须明显低于食指尖
```

调整这些值的效果(经验值):

| 阈值 | 调小 → | 调大 → |
|---|---|---|
| `THUMBS_UP_GAP` | 握拳误识为点赞变多 | 真点赞也被漏掉 |
| `THUMB_DOWN_TOLERANCE` | 真拇指向下被漏掉 | 拇指自然下垂也被误识 |
| `THUMBS_UP_TOLERANCE` | 边缘姿态误识为点赞 | 真点赞漏掉 |

## 三个"手势易混"的边界条件

### THUMBS_DOWN vs POINTING

> 用户反馈:做拇指向下经常被误识为向上指。

**原因**:旧版要求 `index_curled`(食指也弯曲),但做拇指向下时食指常常不是紧握的。

**修正**:THUMBS_DOWN **不再检查食指状态**。只要 `thumb_pointing_down + middle_ring_pinky_curled` 就匹配。

### FIST vs THUMBS_UP

> 用户反馈:握拳容易误识为点赞。

**原因**:FIST 排在 THUMBS_UP 前面,且 THUMBS_UP 的"topmost"判定只看拇指尖是否高于食指尖,容差太小,握拳时拇指轻微抬起就过线。

**修正**:重新排序 + 引入 `THUMBS_UP_GAP = 0.10`。拇指尖和食指尖的高度差必须 ≥ 0.10 才算真点赞。

### THINKING vs POINTING(跨函数)

> 用户反馈:thinking 和 pointing 互相误识。

**原因**:`check_thinking_gesture` 的距离阈值太紧(6%),边界抖动 → THINKING 漏检 → 落到 `classify_gesture` → 因为食指伸出被判 POINTING。

**修正**(在 [check_thinking_gesture](../../gesture-tracker.py#L248-L274) 里,不在这个函数里):
1. 距离阈值放宽到 8%
2. 食指判定从"严格伸出"改成"tip 高于 MCP" — 允许"钩形"食指
3. 让滑动窗口平滑器(见 [smoothing.md](smoothing.md))吞掉剩余抖动

## 加新手势的 checklist

1. **起名**:`SCREAMING` / `WAVE` / `ROCK_ON` / 任意大写 + 下划线
2. **加颜色**:在 [`GESTURE_COLORS`](../../gesture-tracker.py#L82-L93) 加一行,BGR 三元组
3. **加图**(可选):放 `xxx.jpg` 到项目根 + 在 [`IMAGE_PATHS`](../../gesture-tracker.py#L143-L148) 加映射 — 不加也可以,代码会自动跳过图片面板
4. **写规则**:在 `classify_gesture` 里挑一个合适的顺序位置加一个 `if ...: return "XXX"`,规则尽量复用现有的 `_finger_curled` / `middle_ring_pinky_curled` 等 helper
5. **加测试**:在 [testing.md](testing.md) 的合成姿态构造器上加一个 `xxx_pose()` 函数,然后在主测试体里 `check("XXX", mod.classify_gesture(p), "XXX")`
6. **跑测试**:跑 `python dev_doc/testing.md` 里写的命令

## 已知的"故意不做"的手势

- **OK 手势**(拇指 + 食指成圈):`middle_ring_pinky_curled` False(中/无/小指伸出)→ 进不到 THUMBS_*;`index_curled` 也难以精确刻画成圈状态。结论:`NEUTRAL`。要做的话得加 `index_curled AND thumb_index_touching` 判定。
- **摇滚手势**(食指 + 小指伸出):`middle_ring_pinky_curled` False → NEUTRAL。单独加的话 `if index_extended_up and pinky_extended_up and not middle_extended_up and not ring_extended_up` 就行。
- **打电话手势**(拇指 + 小指伸出):和摇滚类似,加规则即可。