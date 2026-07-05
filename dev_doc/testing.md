# 测试策略

## 总体思路:合成姿态 + 无头冒烟

项目没有正式的 pytest suite — 用一个**合成姿态构造器**写一次性脚本,直接 import 主模块然后调函数,不依赖摄像头。

为什么这样做:
- 主程序要摄像头,CI / 服务器跑不了
- 分类逻辑是纯函数(给定 landmarks → 给定 label),理论上和摄像头无关
- 合成姿态可控、可重复、能精准覆盖边界情况

## 怎么跑

合成测试脚本不放在 repo 里(避免污染项目),开发时临时写一个,大致长这样(详见 [development-history.md](development-history.md) 里每次迭代的验证段):

```python
import importlib.util
spec = importlib.util.spec_from_file_location("gt", "gesture-tracker.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

class L:
    __slots__ = ("x","y","z")
    def __init__(self, x=0.5, y=0.5, z=0.0): self.x, self.y, self.z = x, y, z

def hand_at(...): ...  # 21 个 landmark 的 list
def thumb_up(p): ...   # 改拇指的几个关节
# ... 构造目标姿态 ...

assert mod.classify_gesture(p) == "EXPECTED"
```

直接 `python test_synthetic.py`,无任何额外依赖。

## 当前的 16 个合成姿态覆盖

按"姿态 → 期望标签"分组(对应 [development-history.md](development-history.md) 里每次迭代):

| 姿态 | 期望 | 第一次加这个测试的版本 |
|---|---|---|
| THUMBS_UP(经典) | THUMBS_UP | v3 |
| POINTING(拇指斜 45°) | POINTING | v4 |
| POINTING(拇指蜷缩) | POINTING | v4 |
| PEACE (V 手势) | PEACE | v5 |
| OPEN PALM | OPEN_PALM | v5 |
| FIST(拇指蜷缩) | FIST | v5 |
| THUMBS_DOWN(经典) | THUMBS_DOWN | v5 |
| THUMBS_DOWN + 食指伸出 | THUMBS_DOWN | v6 |
| THUMBS_DOWN + 食指钩形 | THUMBS_DOWN | v6 |
| FIST + 拇指轻抬 | FIST(不是 THUMBS_UP) | v6 |
| FIST + 拇指中度抬 | FIST | v6 |
| THINKING(指尖贴鼻) | True | v3 |
| THINKING(指尖~50px 边界) | True | v6 |
| 握拳贴近鼻子 | False | v3 |
| Smoother 处理 THINKING 抖动 | 最终输出 THINKING | v6 |
| Smoother 容忍 1 帧噪声 | 保持当前标签 | v6 |

## 怎么加新测试

举例:想验证加了一个 `ROCK_ON` 手势。

1. 在测试脚本里加一个 `rock_on_pose()` 函数:
   ```python
   def rock_on(p):
       extend_finger(p, INDEX, PIP, TIP, x=0.45)   # 食指伸出
       extend_finger(p, PINKY, PIP, TIP, x=0.60)   # 小指伸出
       # 中指、无名指保持默认(蜷缩)
       return p
   ```

2. 在主测试体里:
   ```python
   p = hand_at(); rock_on(p)
   check("ROCK_ON", mod.classify_gesture(p), "ROCK_ON")
   ```

3. 跑一下,期望 `OK  ROCK_ON ...`。

## 测试覆盖的盲区

合成姿态能覆盖分类器的**逻辑边界**,但覆盖不了:
- **MediaPipe 真实输出**:真实摄像头帧的 landmark 噪声、抖动、误检
- **多人 / 多手**:代码里 `num_hands=1` / `num_faces=1`,只取第一只/张,这个限制在 `main()` 里,合成测试看不到
- **摄像头不存在的环境**:能不能优雅地报错 — 现实里你只能拿真机试

要补这些盲区,最直接的方法是录一段"标准手势视频",然后在测试脚本里把每帧喂给分类器,统计标签分布。

## 一段可以复用的"最小测试骨架"

```python
import importlib.util
spec = importlib.util.spec_from_file_location("gt", "gesture-tracker.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class L:
    __slots__ = ("x", "y", "z")
    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x, self.y, self.z = x, y, z

def hand_at(wrist_y=0.70, curled_tip_y=0.55, curled_pip_y=0.45, curled_mcp_y=0.50):
    """21-landmark hand with middle/ring/pinky curled and thumb tucked."""
    lm = [L(0.5, wrist_y) for _ in range(21)]
    # ... 填每个手指的 MCP/PIP/TIP + 拇指 CMC/MCP/IP/TIP
    return lm

def extend_finger(lm, mcp, pip, tip, x, mcp_y=0.50, pip_y=0.35, tip_y=0.20):
    lm[mcp] = L(x, mcp_y); lm[pip] = L(x, pip_y); lm[tip] = L(x, tip_y)
    return lm

# 你的测试加在下面:
p = hand_at()
assert mod.classify_gesture(p) == "NEUTRAL"  # 默认状态就是 NEUTRAL(握拳)
print("OK default NEUTRAL")
```