# DSRL-SAC 适配到 RoboTwin + GAP：实验与结果完整记录

2026-10-19。覆盖 2026-10-13 起全部 DSRL 相关工作。逐日原始推理过程见 `daily_logs/2026-10-13 / 15 / 16 / 18 / 19.md`。

---

## 1. 结论

**噪声引导在当前这个冻结的 GAP checkpoint 上没有上升空间，瓶颈不在 RL 调参。**

> ### ⚠ outcome-sensitivity 的数字已撤回（2026-10-19 晚）
>
> 补做确定性对照后（`--fixed_noise`：每次抽样喂**完全相同**的动作噪声，且
> `fixed_track_noise=True`），翻转率**不是 0**：steps=10 为 **0.10**、steps=1 为
> **0.33**。即管线不是 `(场景, 动作噪声)` 的确定性函数，**任何翻转率都不能归因于噪声**。
> 同配置重跑 steps=10 还从 0/15 变成 6/15。残留的不确定性来源尚未定位
> （track RNG 已排除），诊断在跑。详见 §6.3 与 §7.7。
>
> **仍然成立的是 §6.2 的 authority 测量**（动作层面，1/370 @10 步，13.4× @1 步）
> 和 §5 的全部训练结果 —— 它们不依赖这个翻转计数。

| 配置 | authority（动作层，仍有效） | outcome 层（**已撤回**） |
|---|---|---|
| `num_inference_steps=10`（部署值） | **0.0035 ≈ 1/370** | ~~0 翻转 / headroom 0~~ |
| `num_inference_steps=1` | **0.0463（13.4×）** | ~~9/15 翻转 / +0.289~~ |

1 步的成功率代价则是配对测量、不受此影响：条件化 0.667 → **0.467**，合并不一致对 15:3（p=0.0075）。

**根因**：checkpoint 的 `prediction_type` 是 `'sample'` 而非 `'epsilon'`，且是刻意选择（`GAP.yaml:82` 注释 `# predict clean sample (not noise!)`）。`'sample'` 下每个 DDIM 步是朝条件均值的**收缩**，迭代 10 次收敛到与 x_T 无关的不动点。DSRL 的前提（x_T → x_0 近似可逆）被结构性破坏。

**另一个独立结论**：此前追了多日的「40% vs 75.33%」35pp 差距**不存在**，是条件成功率 vs 无条件成功率的比较错误（§6.1）。

继续调 `utd` / `target_ent` / `n_critics` / `action_magnitude` 不会改变结果。剩余有上升空间的方向见 §8。

---

## 2. 用的是哪个 DSRL 版本

**全部实验用 `DSRL-reference`。**

| | `DSRL-reference/` ← **本项目使用** | `DSRL-pi0-reference/` |
|---|---|---|
| 上游 | `github.com/ajwagen/dsrl` | `github.com/nakamotoo/dsrl_pi0` |
| 论文 | DSRL 原始论文 arXiv:2506.15799 | DSRL on pi0（RLinf 文档引用的那个） |
| 基座 | 扩散策略，`predict_epsilon: True`（`cfg/robomimic/dsrl_square.yaml:81`） | pi0，flow matching |
| RL | 自带 SB3 fork 子模块 `ajwagen/stable-baselines3-dsrl` | JAX |
| 本项目 | **实际运行的全部训练与评测** | **仅阅读对比超参，从未运行** |

选它：论文原始实现，且基座同为 DDIM 扩散策略 + SB3 SAC，与 GAP 最同构，迁移面最小。实际依赖其子模块里的 SB3 fork：

```python
# train_dsrl_scan_object.py:63
sys.path.insert(0, os.path.join(base_path, "..", "DSRL-reference", "stable-baselines3"))
from stable_baselines3 import SAC
```

**`dsrl_na` 从未实现、从未运行。** `train_dsrl_scan_object.py:301` 硬断言 `cfg.algorithm == "dsrl_sac"`；全仓库搜 `dsrl_na` 只有 `cfg` 里一行「以后值得试」的注释。按 §1 的结论也不值得再实现——它换的是 RL 算法，而瓶颈在策略的噪声不敏感性，换算法不触及这一点。

---

## 3. 系统结构

```
RoboTwin SAPIEN (scan_object, demo_clean_depth)
    │  head_camera rgb/depth + robot state
    ▼  encode_obs + extract_dino_features
state[16], dino_features[391,1024], patch_centers[391,3]
    ▼
SAC actor (SB3 fork)   obs: state(16) + mean-pool dino(1024) = 1040
                       action: 256 = horizon 16 × action_dim 16  ← 就是 x_T
    ▼  x_T
GAPBasePolicyWrapper → GAPPolicy.predict_action_from_noise   （权重冻结，DDIM 10 步）
    ▼
action chunk [16,16]（原始 EEF 位姿）
    ├──► TASK_ENV.take_action × 16 → 环境推进
    └──► reward：dynamics model 预测 track → MVTrackSVMReward
                 或 MVActionSVMReward 直接打分 (state, action)
```

两个约定：DSRL 的 action **就是** DDIM 初始噪声，不是机器人动作；奖励在 chunk 的**前动作观测**上计算（与 mv_svm 自身「每决策步一个奖励、在 chunk 锚帧打分」一致）。

---

## 4. 代码改动

### 4.1 GAP 侧（commit `43fe3ce`）— **未改任何权重或训练代码**

| 位置 | 改动 |
|---|---|
| `gap.py:1697` `conditional_sample_from_noise` | DSRL 唯一接入点：初始动作噪声**外部传入**，其余 DDIM 过程与 `conditional_sample` 完全相同，故对任何 i.i.d. N(0,I) 输入同分布。track 分支仍自己抽样并正常去噪（DSRL 只引导动作噪声）。 |
| `gap.py:2099` `predict_action_from_noise` | 对外接口：obs → memory/memory_pos → 外部噪声进 → 未归一化动作出，返回**完整 horizon**（不截到 `n_action_steps`，DSRL 执行整个 chunk）。 |
| `deploy_policy.py` `_collect_multiview_obs` | opt-in `CHAIN_COLLECT_MULTIVIEW`：在 chunk 锚帧额外采集 head + 双 wrist 的 rgb/depth/K/E 供奖励模型打分。**不改变策略自身观测**（仍只有 head_camera）。 |

### 4.2 `DSRL-reference` 侧 — 仅 1 个文件，+40/−9

`stable-baselines3/stable_baselines3/sac/policies.py`：新增 `squash_actions`（默认 `True`，现有行为不变），`False` 时用无界 `DiagGaussianDistribution` 替代 `SquashedDiagGaussianDistribution`。

需要它的原因：**DSRL 的 action 就是喂给扩散模型的噪声**，tanh 让 `|a| > action_magnitude` 结构性不可达，于是压缩后的 actor 无论怎么训练都发不出 N(0,1) 中超过 ±1.5 的那 13.3% 质量。实测 std 匹配（0.98 vs 1.00）但 kurtosis **1.61 vs 3.01**、`P(|a|>1.5)` **恰好 0 vs 0.133**。压缩还给 `log_prob` 加 tanh 的 Jacobian 项，饱和时爆炸——正是熵调节器被骗的机制。

三个坑（均已写进注释）：

1. **`squash_output=True` 故意保留**，即使 `squash_actions=False`。它只选 `predict()` 分支：True → `unscale_action`（仿射映到边界），False → `np.clip`。无界高斯要仿射那条，**不能要 clip**——clip 会把这个选项想去掉的硬边界加回来，还会破坏 replay buffer（`_sample_action` 用 `scale_action` 填充）与 actor loss（直接把 `action_log_prob()` 原始输出喂 critic）之间的尺度一致性。故必须同时 `action_magnitude=1.0`，让 `unscale_action` 成为恒等。
2. **`_get_constructor_parameters` 原本漏了 `standard_gauss_init`** —— **既存隐藏 bug**：该参数改变前向行为（是否 clamp `log_std`），漏掉意味着 `SAC.load()` 会重建一个**不同的** actor 再把权重装进去，每次 eval 和 resume 静默错误。两个 flag 均已补上并验证序列化往返。
3. **`log_std` clamp 条件放宽**为 `if not standard_gauss_init or not squash_actions`。

> `squash_actions=False` **不是** 35pp 的解（§7.1），但改动本身正确，且逼出了第 2 条真 bug。

### 4.3 新建模块 `dsrl-scan-object/`（适配层主体）

**核心运行时**

| 文件 | 作用 |
|---|---|
| `env.py` | `DSRLScanObjectEnv`：把 `eval_policy.py` 的单集生命周期（正常一次调用跑到底）拆成 `reset()`（建/重置场景）与 `step()`（推进**恰好一个 action chunk**）。**故意不调** `deploy_policy.eval()`（它把动作生成与执行合在一次调用里），因为 DSRL 必须分开——噪声来自 SAC，不是策略自采样。 |
| `gap_base_policy.py` | `GAPBasePolicyWrapper`：满足 `DSRL-reference/utils.py` 的 `DPPOBasePolicyWrapper.__call__` 接口（obs dict + 外部噪声 → 原始动作 chunk）。 |
| `gym_env.py` | gymnasium 接口。`action_space = Box(±action_magnitude, (256,))`；`observation_space` 1040 维（state 16 + dino 391 patch mean-pool 1024）。**mean-pool 是临时选择**，丢掉了空间细节，换更强编码器是将来可做的事。 |
| `reward.py` | 两个奖励源的加载与 obs 构造。 |
| `dynamics_scorer_client.py` / `_worker.py` | dynamics 打分跑**独立子进程**。必须如此：dynamics 仓库自带一份 `gap_policy` 包，与 GAP 本体命名空间冲突，同进程会互相覆盖。 |
| `train_dsrl_scan_object.py` | hydra 训练入口。 |
| `dsrl_final_eval.py` | 最终确定性评测（actor 均值动作，无探索噪声）。 |

**必须做对但都不显然的点**

- **`hydra.job.chdir: False`** —— RoboTwin 的 `envs` 模块按相对 cwd 加载资产，cwd 必须留在 `ROBOTWIN_ROOT`。
- **跳过 `play_once()`/`expert_check`** —— 原因是 GAP 不消费语言。**但这个决定是 35pp「差距」的全部来源**（§6.1）。现已加 opt-in 的 `expert_check=True`。
- **`swap_rgb_channels=False`** —— 对齐 baseline **实际启动配置**（`eval.sh:31` 显式传 False，覆盖 `deploy_policy.py:655` 过期的 True 默认）。实测 False → 0.400，True → 0.233（**不要重试**）。
- **`reward_source`** —— `"track"`（dynamics → MVTrackSVMReward，子进程）/ `"action"`（MVActionSVMReward，进程内）/ `"none"`（常数 0，两个模型都不构造）。
- **`"action"` 复用 GAP 自己的 DINOv2 实例** —— 训练 env 与 eval env 各持一个完整 `GAPPolicyWrapper` + `MVActionSVMReward`，两份独立 DINOv2 会把 12GB 卡打爆（job 4029232 实际发生）。

**`train_dsrl_scan_object.py` 的改动**

| 改动 | 原因 |
|---|---|
| `collect_init_rollouts(...)`，`init_rollout_steps=1500` | 用**未引导**扩散策略 transition 预填 replay buffer（对齐 DSRL-reference 的 1501）。 |
| `collect_init_rollouts(..., clip=)` | 预填原本把噪声 clip 到 ±`action_magnitude`，于是那 1500 条「未引导扩散策略」transition **缺了同样 13.3% 的尾部**，根本不是文档声称的基线分布。现在与 `squash_actions` 挂钩。 |
| `DeterministicEvalCallback` + `reset_seed_counter()` | 每轮 eval 开始同步回**同一批固定 seed**。之前每轮静默漂移到全新场景，「策略变好了没」与「这批更容易吗」混在一起。 |
| 日志用 `num_timesteps` 而非 `n_calls` | 后者每次进程启动归零，与 resume 不兼容（历史 10k 续训的 step 值曾需手工 +2000）。 |
| step=0 训练前 baseline eval | 对齐 DSRL-reference 自己的 `train_dsrl.py`。 |
| `squash_actions=False` + `action_magnitude != 1.0` 直接报错 | 见 §4.2 第 1 条。 |
| `_WandbOutputFormat` | 把 SB3 的 `train/*`（`ent_coef`、`ent_coef_loss`、actor/critic loss、`n_updates`）转发到 wandb。没有它这些只进 stdout，而它们恰是诊断熵崩溃的关键量。 |
| `n_eval_episodes` 3 → 10 | DSRL-reference 自己每轮评 200 集，这里不可行；10 是折中（可用 `DSRL_EVAL_EPISODES` 调）。 |

**2026-10-19 新增**

| 改动 | 原因 |
|---|---|
| `expert_check=True` + 磁盘缓存 `results/expert_check_cache.json` | 复刻 RoboTwin 自己的专家 seed 筛选，让 eval 与已发表数字同口径（§6.1）。专家结果按 `(task,config,seed)` 确定故可缓存，固定 batch 只付一次代价。已对独立测量验证：跳过的 seed 完全一致。 |
| `reward_source="none"` | 纯成功率评测变便宜（无 dynamics、无 SVM、无子进程），并把整个奖励路径从 `step()` 移除——它是唯一触碰 `_collect_multiview_obs` / `_verify_action_track_setup` 的代码。 |
| `num_inference_steps` 贯通 env→gym_env→train→cfg（默认 0 = 用 checkpoint 的 10） | 唯一经测量能改变噪声影响力的旋钮（§6.2）。 |
| `reset(seed=, video_label=)` | 支持重放同一场景（§6.3 必需）与录像命名。默认行为不变。 |
| `video_dir` 录像 | 接 RoboTwin 自己的 ffmpeg。**坑**：`setup_demo` 一旦看到 `eval_video_save_dir` 就设 `eval_video_path`，之后 `take_action` 每步往 ffmpeg stdin 写帧——每次 `setup_demo` 后、任何 `take_action` 前都必须挂上 ffmpeg，否则抛异常；专家检查那遍（`play_once` 也调 `take_action`）必须显式屏蔽。 |
| `video_substep_freq` / `video_fps` | **修视频卡顿**（见 §4.4）。默认 0 = 保持 RoboTwin 原粒度，不影响任何已有流程。 |
| `fixed_track_noise` / `GAPBasePolicyWrapper(track_seed=)` | **修真缺陷**：原先传 `generator=None`，`joint_diffusion` 的 track 分支从**全局 torch RNG** 抽样，而 track 影响动作 → 策略输出依赖进程历史，且不是 DSRL agent 能控制的量。GAP 自己的 eval 从无此问题（`deploy_policy.py:507/:561` 显式传 seeded generator）。详见 §7.4。 |
| resume 修复（`DSRL_RESUME_REPLAY_BUFFER`） | 预填原在 `else` 分支，resume 带**空 buffer** 在 `learning_starts=1`、`utd=20` 下开始——正是熵崩溃的起始条件，预训练权重挡不住。更糟：`dsrl_train_continue_10k.sbatch` 传了 `save_replay_buffer=true`，buffer 一直在写但 resume **从未调 `load_replay_buffer`**。现可恢复，没给则按新鲜 run 预填。历史 `scan_object_explore_10k_continued` 仍不能当干净续训读。 |

### 4.4 录像粒度：为什么手臂看起来在瞬移

**实测**：每 chunk **15.2-16.1 帧**，正好是 GAP 的 action horizon 16。所以是**每个 `take_action` 一帧**，不是每个 chunk 一帧。

机制（`envs/_base_task.py:1480-1666`）：

1. 帧在 `take_action` **开头**写，用的是 `self.now_obs` —— **该动作执行之前**的观测。
2. 随后 `left_plan_path` 规划出 `left_n_step` 个插值 waypoint（规划失败时固定 50），进入 **control loop**：每 waypoint 一次 `scene.step()` + `_update_render()`。
3. **循环里不写任何帧**（只在 `check_success()` 变真时补写一帧）。

所以相邻帧之间跳过了几十个物理步的真实运动。这不是适配层的问题——RoboTwin 自己的 eval 视频同粒度（`eval_video_freq = 1  # fixed`，但那个 1 的单位是 action step，不是物理步）。

**修法**：在 `env.py` 里 wrap `TASK_ENV._update_render`（它恰在 control loop 中每物理步后被调一次），按 `video_substep_freq` 额外写帧；RGB 走 RoboTwin 自己单帧保存的便宜路径（`save_camera_rgb:503` 的 `cameras.update_picture()` + `get_rgb()`，不碰 depth/点云）。**不动 `_base_task.py`**——该文件被所有 eval 路径与数据采集共用。第一帧会校验 shape 与 ffmpeg 的 `-video_size` 一致，不一致直接报错而非静默写出损坏视频。代价是每 N 个物理步多一次 RGB 渲染，故仅用于诊断视频，默认关闭。

### 4.5 诊断脚本（`tests/`）

| 脚本 | 测什么 |
|---|---|
| `compare_noise_distribution.py` | actor 噪声边缘分布 vs 真 N(0,1) 对照组，走完全相同的测量代码：std、kurtosis、pooled KS、尾部质量、跨维相关、边界占比。 |
| `compare_policy_paths.py` | 两条策略调用路径是否同分布 + authority ratio。**故意比分布而非精确匹配噪声**：`joint_diffusion` 下 track 从同一生成器抽样且影响动作，复现一条路径的动作噪声仍留下不同的 track 噪声。 |
| `sweep_inference_steps_authority.py` | authority vs 去噪步数，**同一批固定 state 上配对**。 |
| `measure_expert_pass_rate.py` | RoboTwin 的 seed 筛选通过率。只跑专家，不加载 GAP/DINOv2/扩散。 |
| `eval_unsteered_baseline.py` | 未引导策略成功率，**逐 seed 记录结果**，于是同一批 rollout 同时给出条件化与无条件两个口径。 |
| `measure_outcome_sensitivity.py` | **固定场景重放 K 次只换噪声，统计结果是否翻转**；`--fixed_noise` 是确定性对照。不经过任何奖励模型。 |
| `record_rollout_videos.py` | 复现测量时的动作噪声流并录像，文件名按结果改名。 |
| `inspect_actor.py` / `analyze_ent_coef.py` / `eval_checkpoint_curve.py` | actor 状态、熵系数轨迹、checkpoint 曲线。 |
| `svm_sanity_check.py` / `action_svm_sanity_check.py` | 两个奖励模型的 sanity。 |
| `measure_noise_authority.py` | **已被取代**（见文件头注释），保留仅为可追溯。 |

---

## 5. 实验与结果

### 5.1 管线验证与 smoke test

| 实验 | 结果 |
|---|---|
| env pilot（job 4028526）：2 集、`step_lim=40`、随机噪声 | rc=0，obs 形状正确（state (16,)、dino (391,1024)、patch_centers (391,3)），奖励有限且在预期区间，按 `step_lim` 正常终止。两集都失败——随机噪声本应如此。**管线通，不说明任何 RL 性能。** |
| 预填机制 smoke（job 4031075） | rc=0，形状/`scale_action` 无误。 |
| resume + eval smoke（job 4029042） | rc=0，resume 路径可跑（但当时未发现 buffer 未恢复，见 §4.3）。 |
| 随机（非确定性）eval 对照（job 4034043） | step-0，n=30，`deterministic=False` → **0.400**，与确定性 eval 相同。 |

### 5.2 奖励模型 sanity：不是问题

| 模型 | per-step AUC | 成功 step logit | 失败 step logit |
|---|---|---|---|
| track-SVM（经 dynamics 预测 track） | **0.9771** | — | — |
| action-SVM（直接 (state, action)） | **0.9693** | +0.1575 ± 0.0956 | −0.4013 ± 0.2105 |

695 chunk / 50 集（41 成功、9 失败）。两个 track 来源给出的奖励几乎一致（`mean |reward_dyn − reward_pol| = 0.0091`）。

奖励与最终成败的关系（10-19 新测）：point-biserial **+0.609 ~ +0.675**；成功集累积奖励 **+2.1 ± 0.9** vs 失败集 **−10.3 ± 8.8**。

**已知局限**（action-SVM 自己的 README 也记载）：它能很好区分**不同状态**，但**同状态内**区分候选动作很弱——候选间 logit 差只有 ~0.004，而跨状态 ~3。作为稠密 RL 奖励可用，作为测试时候选排序无用。

> 这条**不能**用来论证策略的噪声不敏感性——它是那个**奖励模型**的性质。我曾一度误引，已更正（§7.6）。

### 5.3 主结果：DSRL 训练让策略变差

最终确定性评测（actor 均值动作、无探索噪声），50 集/seed × 3 seed，seeds 5000/5001/5002（训练与其他 eval 从未触碰），**与 baseline 同口径**（都走专家筛选）：

| seed | baseline | DSRL track 奖励（10k 步） | DSRL action 奖励（2k 步） |
|---|---|---|---|
| 5000 | 74.0% | 44.0% | 48.0% |
| 5001 | 80.0% | 44.0% | 44.0% |
| 5002 | 72.0% | 46.0% | 50.0% |
| **均值** | **75.33%** | **44.67%** | **47.33%** |

三 seed 紧密一致，不是噪声。2k 和 10k 两个预算都明显变差。

### 5.4 超参诊断（全部失败）

| 实验 | 结果 |
|---|---|
| **熵崩溃**（10-13 诊断，10-19 重测更正） | 崩溃本身是真的且严重：entropy **+120（未训练）→ +6.9 / −40.0**（`explore_2k` / `action_explore_2k`，各 39,980 次更新），`ent_coef` 0.94 → 0.02。**但 10-13 记录的机制细节不成立**，见 §7.5。驱动量是 `sigma` **1.094 → 0.705** 加小幅均值漂移，不是 tanh 饱和。 |
| **预填 1500 + eval seed 固定**，3k 训练（job 4031196） | eval 50/60/50/50/50/50/40%；`ent_coef` 仍塌到 0.005-0.02（同一个崩溃渐近值，只是到达更缓）。**预填单独不解决熵崩溃。** |
| **`action_magnitude` 扫描**（未训练 actor，step-0，n=30/点） | 0.3 → 40.0%，0.5 → 46.7%，1.0 → 50.0%，1.5 → 40.0%。**窄带 40-50%，无单调趋势。** |
| **`target_ent` 根因**（10-18） | 它是**绝对 nats、对全部 256 维求和**，依赖参数化、不能跨 setup 照搬。从 28 维 robomimic 配置逐字继承的 `0.0` 使平衡项 `(log_prob + target_ent)` 永远是 −174.5，`ent_coef` 塌到 ~0.01。 |
| **`target_ent=174.5` + `utd=5`**（10-18/19，全程 10k） | 熵约束**完美维持**（435 个记录点，平衡项never 离 0 超 ±2，`log_prob` 钉在 −174.5），**但 critic 仍发散**：`critic_loss` 2.3e4 → 4.1e7 单调约 9000 步，Q ≈ 3.2e4。`utd` 20→5 只改**速率**（step 5000 时 8e6 vs 1.44e8，α 峰值 34 而非 85），不改**方向**。 |

**配对 10k 实验在 ~5000 步的对照**（发生的是 ~5% 的尾部情形，不是 85% 的中心预测）：

| | control（`target_ent=0`） | experiment（`target_ent=174.5`） |
|---|---|---|
| `ent_coef` | 0.837 → **0.00595** | 1.01 → **峰值 ~85**，45-85 振荡 |
| 平衡项 | −174 → −37.7 | **均值 +0.19**（−0.30..+1.93） |
| 重建 `log_prob` | −60.7 → −37.7 | **−174.6**，205 点全程钉住 |
| `critic_loss` | 5.38e3 → **0.67** | 1.34e4 → **1.44e8** |
| Q 估计 | 423 → 396 | 100 → **61,873** |
| `eval` @0/2500/5000 | 0.400 / 0.400 / 0.400 | 0.400 / 0.400 / — |

**Q 的理论上界**：每步奖励硬界 [−2, 2]（`reward.py` 直接返回 `0.1 * clip(logit, ±20)`，不加结果奖励），每集平均 23.2 chunk 步，γ=0.99 → `|Q| ≤ 41`；算上 SB3 的 timeout bootstrapping 最多 200。实测 **61,873 = 最松上界的 300 倍**。

`alpha` 爬到 85 是**结果而非病因**：它是在发散的 critic 面前把熵按在 target 上所需的力。`ent_coef` 与 `critic_loss` **同步振荡**（85→45→78→60 对 2.1e8→0.7e8→1.9e8→1.4e8），是追移动目标的活跃反馈环，不是单调失控。

**提出的机制**：熵按高 → 注入噪声保持多样 → state-action 分布保持宽 → `utd=20` 在 `n_envs=1` 上对每条独特 transition 做 20 次梯度步，在宽支撑上 TD target 自举跑飞。control 只是**偶然**避开：`alpha → 0` 把策略收窄，critic 要拟合的区域变窄所以收敛。**两个病理一直在互相掩盖——熵崩溃反而在维持 critic 稳定。**

### 5.5 四个机制状态，同一个数字

| run | 机制状态 | eval@0 / 2500 / 5000 / 7500 / 10000 |
|---|---|---|
| control | 熵崩溃，critic 收敛 | 0.400 / 0.400 / 0.400 / 0.500 / **0.400** |
| utd=5 | 熵维持，critic 发散 | 0.400 / 0.400 / 0.500 / 0.400 / — |
| utd=20（已杀） | 熵维持，critic 快速发散 | 0.400 / 0.400 / 0.400 / — / — |

训练 rollout 成功率同样一致：**0.398 / 0.428 / 0.406** —— 随机探索策略也从未找到更好的区域。

**四个机制状态落在同一个数字上**，是「问题不在 RL 内部」的强信号——被 §6 证实。

### 5.6 Best-of-N（不是 RL，但测同一件事）

3 seed × 50 集：

| seed | baseline | best-of-N (track) | best-of-N (action) |
|---|---|---|---|
| 0 / 1 / 2 | 80.0 / 66.0 / 66.0% | 88.0 / 64.0 / 62.0% | 76.0 / 56.0 / 74.0% |
| **均值** | **70.67%** | **71.33%**（+0.67pp，wash） | **68.67%**（−2pp） |

§6.3 回溯解释了原因：**根本没有可选择的东西。**

---

## 6. 三个决定性测量（10-19）

### 6.1 「35pp 差距」不存在

RoboTwin 评测协议**硬编码**专家 seed 筛选。GAP 自己的 `scripts/eval_policy.py:254`（`eval.sh` 真正调用、产出 75.33% 的那份）设 `expert_check = True` 且无覆盖路径：

```python
if expert_check:                                  # :277
    TASK_ENV.setup_demo(..., seed=now_seed, ...)
    episode_info = TASK_ENV.play_once()           # 脚本专家
if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
    succ_seed += 1                                # :301 seed 被接受
else:
    now_seed += 1; continue                       # :305 从不交给策略
```

而 `env.py` 的 `reset()` 完全跳过它（其 docstring 原文："Every seed is used as-is"）。

**专家通过率**（前三行零成本：已完成评测早把逐集 seed 列表写进了各自的 `_rollouts.json`，通过率 = `50 / seed 跨度`）：

| seed 段 | 集数 | 消耗 seed | 专家通过率 |
|---|---|---|---|
| 100000+ / 200000+ / 300000+（已存） | 50 | 180 / 179 / 177 | 0.2778 / 0.2793 / 0.2825 |
| 500100000+（75.33% 那批） | — | 30 | 0.2667 |
| 100200000+（**所有 DSRL eval 那批**） | — | 30 | **0.4333** |

每评 1 集烧约 3.6 个 seed。所以 **75.33% 是 `P(成功 | 专家可解)`**，而 0.400 是无条件。

**确证**（未引导冻结策略，`reward_source="none"`，n=30/arm）：

| 测法 | 结果 | 95% Wilson CI | 覆盖 0.7533 |
|---|---|---|---|
| 开专家筛选直接测 | 20/30 = 0.6667 | [0.488, 0.808] | 是 |
| raw arm 按专家 mask 事后条件化 | 11/13 = 0.8462 | [0.578, 0.957] | 是 |
| **两者合并** | 31/43 = **0.7209** | [0.573, 0.833] | 是 |
| raw arm **无条件** | 17/30 = 0.5667 | [0.392, 0.726] | — |
| baseline | 0.7533 | — | — |

三个内部一致性检验通过：raw arm 精确分解为 `13/30 × 0.846 + 17/30 × 0.353 = 0.5667`；expert arm 为 30 集烧 70 个候选（通过率 0.4286 对独立测得的 0.4333）；expert arm 共享 seed 与新 seed 两半一致（9/13 = 0.692，11/17 = 0.647）。

**所以没有差距。** 专家失败子集 6/17 = 0.353 —— 低但明确非零（`plan_success=False` 是手写专家的 IK/规划失败，不是场景不可解）。

### 6.2 噪声对动作的权威（authority）vs 去噪步数

两条路径：**A** = `GAPPolicyWrapper.get_action`（GAP **自己的** eval 入口 + 自己的 `torch.randn`），**B** = DSRL env 路径。6 个真实状态 × 每状态 32 抽样。路径一致性 `gap / max(se)` = **1.12**（~1-3 表示同分布），所以结果不是 DSRL 管线的伪影。

| `num_inference_steps` | authority A | authority B | 跨噪声散度 A | 跨状态散度 A |
|---|---|---|---|---|
| 10（部署） | 0.00345 | 0.00341 | 0.000466 | 0.135064 |
| 5 | 0.00263 | 0.00276 | 0.000355 | 0.135062 |
| 3 | 0.00252 | 0.00236 | 0.000340 | 0.134985 |
| 2 | 0.00202 | 0.00206 | 0.000273 | 0.134872 |
| **1** | **0.04626** | **0.04479** | **0.006035** | 0.130459 |

部署值下跨噪声散度 0.00038 vs 跨状态 0.135 ≈ **1/370** —— **冻结策略实际上是观测的确定性函数**。steps=10 一行复现了独立的 `compare_policy_paths.py`（0.00282 / 0.00263，不同 state、不同 seed），仪器自洽。

**我预测的方向和理由都错了**：以为步数少则对 x_T 冲刷少、authority 升。实测 10→2 反而**降**，且跨状态分母几乎不动（0.1351→0.1349），是分子的真实变化。DDIM 对 x_T 的线性贯穿系数 `sqrt((1-ā_0)/(1-ā_T))` **与步数无关**，所以「步数少则保留更多 x_T」从来不是一个机制。真实机制是**每步朝条件均值收缩**（网络被训练成把任意带噪输入映到干净动作），步数越多越收敛到与 x_T 无关的不动点；恰好 1 步时只有一次前向、无重锚定，输出就是 `net(x_T, T, obs)`。

**但 1 步要付 20pp**（同一批 30 场景，逐场景配对）：

| 口径 | steps=10 | steps=1 | 配对（both/neither/只10赢/只1赢） | McNemar 精确 p |
|---|---|---|---|---|
| 条件化（可比 75.33%） | 20/30 = 0.667 | 14/30 = **0.467** | 11 / 7 / 9 / 3 | 0.146 |
| 无条件 | 17/30 = 0.567 | 11/30 = **0.367** | 11 / 13 / 6 / **0** | **0.031** |
| 合并不一致对 | — | — | 只10赢 **15**，只1赢 **3** | **0.0075** |

两口径都正好掉 20pp，方向基本单边。无条件 arm 里**没有**任何场景是 1 步解出而 10 步解不出。

### 6.3 噪声能否改变**结果**（最关键）

前两个测的是「噪声把**动作数值**推动多少」。但 RL 学的是**奖励**（此处是终局成败）：若固定场景下每个噪声都给相同结果，回报就是动作的常函数，**无论 authority 多大都没有梯度可爬**。

做法：每个场景重放 K=3 次——**相同 seed、相同 SAPIEN 初始状态**——只换噪声流。**不涉及任何奖励模型**，所以 action-SVM 自身的候选不敏感性（§5.2）污染不到结论。

| | **steps=10** | **steps=1** |
|---|---|---|
| 结果不一致的场景 | **0/15 = 0.000** | **9/15 = 0.600** |
| 单次抽样 | 0.7333 | 0.4444 |
| **oracle best-of-3（上限）** | 0.7333 | **0.7333** |
| worst-of-3（下限） | 0.7333 | **0.1333** |
| **headroom** | **测不出** | **+0.2889** |

逐场景（1 = 成功）：

```
steps=10:  全部 15 个场景一致 —— [0,0,0] ×4，[1,1,1] ×11
steps=1 :  100200001 [0,0,1]F  100200002 [0,0,0]   100200005 [0,0,0]
           100200008 [0,1,1]F  100200009 [1,0,1]F  100200012 [1,1,1]
           100200017 [0,1,1]F  100200019 [1,1,1]   100200021 [0,1,1]F
           100200023 [0,0,0]   100200024 [0,0,0]   100200025 [1,0,0]F
           100200026 [0,0,1]F  100200031 [0,0,1]F  100200034 [1,1,0]F
```

> **以下读法已撤回**，因为确定性对照不为零（见本节末）。原文保留以便追溯。

- **steps=10**：测不出收益。回溯解释了 best-of-N 是 wash（§5.6）、eval 在四个机制状态下都停在 0.400（§5.5）、squashed 与 unsquashed 都是 0.400 且 22/26 场景一致。
- **steps=1**：杠杆是真的（60% 场景会翻），但**完美引导的天花板 0.7333 恰好等于普通 10 步策略已有的 0.7333**。给 DSRL 一个完美 critic，1 步下最好也只是打平那个不用 RL 的 10 步策略；而 worst-of-3 仅 0.1333——选错是灾难性的。
- **统计边界**：0/15 的单侧 95% 上界是 **18.1%**，所以排除的是**大**敏感性，不能排除小的。K=3 对 p=0.5 场景检出率 75%，对 p=0.1 只有 27%。据此决策的依据不是这一个测量很紧，而是**三个独立测量一致**：authority 1/370（动作层）、best-of-N n=150 只 +0.67pp（结果层，经奖励模型）、0/15 翻转（结果层，不经奖励模型）。

### 6.3.1 确定性对照：翻转率不能归因于噪声

补做的对照（`--fixed_noise`，每次抽样**同一条**动作噪声；`fixed_track_noise=True`）：

| run | 动作噪声 | 翻转率 | single | oracle | headroom |
|---|---|---|---|---|---|
| steps=10 | 变化 | 0.4000 | 0.7556 | 0.9333 | +0.1778 |
| **steps=10** | **相同（对照）** | **0.1000** | 0.6333 | 0.7000 | **+0.0667** |
| steps=1 | 变化 | 0.7333 | 0.5111 | 0.8000 | +0.2889 |
| **steps=1** | **相同（对照）** | **0.3333** | 0.3556 | 0.5333 | **+0.1778** |

**对照不为零**，所以：

1. **「steps=10 翻转 0/15、headroom 恰好 0」撤回。** 同配置重跑得到 6/15（文件同名被覆盖，原数字见 §1 与本节上文）。一个测量给出 0/15 与 6/15，说明不确定性主导。
2. **「steps=1 headroom +0.2889」不能归因于噪声** —— 对照自身就有 +0.1778。
3. 只能勉强保留**同 run 内的配对差**：两种步数下「变化」都比「相同」高约 **+0.30** 翻转率。但 n=10–15，该差仅 1.5–2σ，而且它暗示两种步数下噪声贡献相当，**与 authority 的 13.4 倍不一致**。

track RNG 已排除（四个 run 都 `fixed_track=True`）。候选来源：GPU 浮点不确定性（cuDNN/TF32）造成动作块微小差异、被接触动力学放大（scan_object 含抓取接触，混沌）；或 `reset(seed=)` 未完全复位 SAPIEN。`tests/diagnose_nondeterminism.py` 通过比较**动作块本身**而非成败来判别：动作块逐位相同而观测发散 → 物理引擎；第一步动作就不同 → 策略/GPU 侧。

**要让这个测量成立，需要**：先定位并消除（或量化）该不确定性，然后用「变化噪声 − 相同噪声」的配对差作为估计量，并把场景数提到能分辨 0.1 量级差异的规模。

---

## 7. 被推翻的假设与更正

| 假设 | 结果 |
|---|---|
| **35pp 差距本身** | **不存在**，条件 vs 无条件的比较错误（§6.1）。代价最大。 |
| tanh 压缩（§7.1） | **推翻**：0.400 vs 0.400（n=30 各），26 配对场景 22 个结果一致，不一致 2:2。 |
| RGB 通道序（§7.2） | **推翻**：0.233 vs 0.400（更差）。烧了 ~40 分钟 GPU。 |
| tensor 形状（§7.3） | **读代码即推翻**，未跑。 |
| 降去噪步数 | **方向和机制都反了**（§6.2）。线索对，推理错。 |

### 7.1 tanh 压缩

预测解除压缩会补上 35pp。实测 **0.400 vs 0.400**。该节的**测量本身成立**（kurtosis 1.61 vs 3.01、`P(|x|>1.5)` 精确为 0），站不住的是从它推出的因果主张。代码改动保留（§4.2）。

### 7.2 RGB 通道序 —— 方法上最贵的一次

我从 `deploy_policy.py:655` 的 `True` 默认推断 baseline 做了通道交换，改了 `env.py`，烧掉 ~40 分钟 GPU，实测 **0.233（更差）**。而 `eval.sh:31` 显式传 `False`，注释写明默认在 2026-08-01 改为 False，因为 peilin 的 26 任务批次在真彩色上训练。**错的是顺序，不是假设**：baseline 的实际启动配置在花任何 GPU 时间之前就能回答。已回滚，两个实测数字写进代码注释防重试。

### 7.3 tensor 形状

DSRL 传 `dino_features` 是 3-D `[1,391,1024]`，`get_action` 是 4-D `[1,1,391,1024]`。`gap.py:1170` 把 4-D reshape 成 `(B, N_views*N_patches, D)`——在 `N_views=1` 时是 no-op——而 3-D 走 `else` 原样通过。**两者张量完全相同。**

### 7.4 track 噪声不受控 —— 适配层的真缺陷

为了**看**翻转而不只是数它，录了几段 rollout。steps=10 对照组在场景 100200009 录出 `[1,0,1]`——而 §6.3 表里同一场景记的是 `[1,1,1]`。

查因（读代码即可）：`gap_base_policy.py` 调 `predict_action_from_noise` 时**没传 generator**，于是 `conditional_sample_from_noise` 里 `joint_diffusion` 的 track 分支 `torch.randn(..., generator=None)` 走**全局 torch RNG**。track 影响动作，所以：

1. 「场景 S 第 k 次抽样」在不同脚本/顺序下**不是同一件事**——还取决于进程里之前消耗了多少全局 randn。两结果矛盾不需要任何 bug 就能解释，录像脚本 docstring 里「复现精确噪声流」的说法对 track 分支是错的。
2. 翻转计数**混淆了动作噪声与 track 噪声**。而 steps=10 下动作 authority 仅 1/370，那点结果变化**可能完全由 track 噪声驱动，与 DSRL agent 的动作无关**。
3. **对 DSRL 训练本身是此前未识别的独立障碍**：回报方差里有一部分 agent 既观测不到也控制不了。在 1/370 下它易压过 agent 自身影响，直接毒化 credit assignment。

**影响有多大**：录像与 sensitivity 测量在同一 (场景, 动作噪声) 上的逐集一致性只有 **8/12**：

| 场景 | 测得 | 录像 | 一致 |
|---|---|---|---|
| 100200009 @1步 | `[1,0,1]` | `[1,0,1]` | 3/3 |
| 100200025 @1步 | `[1,0,0]` | `[1,0,1]` | 2/3 |
| 100200034 @1步 | `[1,1,0]` | `[0,0,0]` | 1/3 |
| 100200009 @10步 | `[1,1,1]` | `[1,0,1]` | 2/3 |

（录像还差了 `video_dir` 与 GPU，故这不是纯对照；干净的对照是 `--fixed_noise` arm。）

GAP 自己的 eval 从无此问题：`deploy_policy.py:507/:561` 显式传 `self._sample_generator`，:158-164 注释写明「没有它，重跑同一个 eval 会纯因采样噪声落到不同成功率上」。**DSRL 路径只是从未接上。**

已修（`fixed_track_noise`，默认开），并补上**本该一开始就做的确定性对照**（`--fixed_noise`：每次抽样喂完全相同的动作噪声，此时任何不一致都是管线不确定性；非零就意味着翻转率不能归因于噪声）。

**仍成立的部分**：10 步的 0/15 与 1 步的 9/15 在**同一套（同样有缺陷的）条件**下测得，共享同一额外变化源，故「1 步远比 10 步敏感」不受影响且差异很大。受影响的是 10 步那列的精确值、以及把它读成「恰好零」。**已录视频是修复前跑的**，故其中 1 步的翻转也可能部分由 track 噪声驱动。

### 7.5 熵崩溃的机制描述是错的（饱和从未复现）

2026-10-13 记录：*"actor 噪声每维精确饱和在 ±1.5，跨抽样近零方差，范数 22.6"*，并由此推出「饱和 → tanh Jacobian 项抬高 `log_prob` → 骗过熵调节器 → 自我强化」。

2026-10-19 用 `tests/inspect_actor.py`（扩展了 mean/std/min/max 列并加了真 N(0,1) 对照）重测**磁盘上全部** SAC checkpoint：

| checkpoint | a_mag | mean | std | L2 norm | sat_frac | sigma | draw_std | entropy |
|---|---|---|---|---|---|---|---|---|
| [对照] 真 N(0,1) | — | −0.010 | 1.003 | **16.04** | 0.136 | — | — | — |
| 未训练（squashed） | 1.5 | 0.023 | 0.975 | 15.60 | 0.035 | 1.094 | 0.903 | +120.3 |
| `explore_2k` final | 1.5 | 0.011 | 1.031 | 16.46 | 0.016 | 0.712 | 0.592 | **+6.9** |
| `action_explore_2k` final | 1.5 | 0.066 | 1.091 | **17.47** | 0.028 | 0.705 | 0.528 | **−40.0** |
| `explore_10k_continued` final | 1.5 | 0.071 | 0.969 | 15.54 | 0.010 | 0.680 | 0.611 | +50.8 |
| `targetent_diag` final | 1.5 | 0.012 | 0.882 | 14.11 | 0.003 | 0.849 | 0.848 | +165.2 |

- **norm 22.6 复现不出来**：最高 17.5，全部在对照 16.0 的 ±10% 内，离完全饱和的 24.0 很远。
- **sat_frac 0.003–0.028**，而真 N(0,1) 在 ±1.5 界外本来就有 **0.136** 的质量。
- **draw_std 0.53–0.61**，不是「近零」。`|mean_z|` 0.03–0.93，远低于饱和所需的 >>3。

所以 actor 是**偏窄**，不是撞边界。「饱和 → Jacobian → 骗过调节器」这条链在 `sat_frac ≈ 0.02` 下不可能是主因。真实机制更简单，且与 §7.6 的 `target_ent` 更正一致：**`target_ent=0.0` 被满足得过于宽松 → 调节器从不回推 → α → 0 → actor 自由收缩 `sigma` 去利用 critic。** 不需要饱和参与。

**附带的方法教训**：mean / std / min / max / L2 norm **检测不到这个故障**。它们是边缘统计（每个观测抽一次后汇总），而不同观测产生不同均值，跨状态的变化会冒充离散度 —— 崩溃后边缘 std 仍是 1.03/1.09。只有**固定观测下**的量（`sigma`、`draw_std`）才暴露它。这与 authority 的分子是同一个量。

同时修了该脚本的一个真 bug：`action_magnitude` 原先从 obs cache 取并套用到所有 checkpoint，于是 `unsquashed_init_check`（实际跑 `1.0`）被按 `1.5` 报成 std 1.500 / norm 23.98 —— 一行看起来像严重饱和、实则纯属缩放错误的数据。现按各 checkpoint 自己的 hydra 配置读。

### 7.6 三处对自己先前判断的更正

- **「steps=10 的 headroom 恰好 +0.0000」说过头了**。点估计是 0，但 n=15、K=3，95% 上界 18.1%，且录像直接录到一个 10 步下的翻转。应表述为「低且在当前样本下测不出」。
- **`target_ent = 174.5` 是过度矫正不是修复**。压缩高斯每维熵上界 `log 2 = 0.693`，174.5/256 = 0.682 ≈ **理论最大值**（DSRL 论文用 0/28 = 0；RLinf 用 −16/32 = −0.5）。所以 `0.0` 从来不是**不可达**，而是被满足得过宽、α 正确地 → 0 从而失去熵正则（真问题，但是另一个问题）。我自己的数据显示设到上限的后果：`critic_loss` 2.3e4 → 4.1e7。
- **「0.004 vs 3 的候选差」被我误引**。那是 **action-SVM 自己记载的局限**——奖励模型的性质，不是策略的性质。

### 7.7 outcome-sensitivity 缺了确定性对照（最新一次同类错误）

见 §6.3.1。我在没有确定性对照的情况下，把一次 0/15 的观测当成了「headroom 恰好为零」这一结构性结论，并据此宣布 DSRL 在 10 步下不可能工作。补做对照后，相同噪声本身就有 0.10/0.33 的翻转率，整个归因失效。

这与 §7.2、§7.5 是**同一个方法错误的第三次出现**：拿一个点估计当结构性事实，而本来有一个更便宜的检验可以先做。这次那个检验就是「把同一个输入跑两遍」。

### 7.8 同一个方法错误产生了上面多数失误

每次都在**没先读清对照数字是怎么产生的**情况下就去比较。通道那次答案在 `eval.sh:31`；35pp 那次在 `eval_policy.py:254`，更便宜的话在 `_rollouts.json` 里早已躺着的 seed 列表；§7.4 那次缺的是一个本该先跑的确定性对照。

---

## 8. 剩余选项

### 8.0 `prediction_type='epsilon'` 的支持现状（2026-10-19 已接好）

**采样路径本来就支持**：diffusers 的 `DDIMScheduler.step` 内部按
`config.prediction_type` 分支（epsilon / sample / v_prediction 都处理），所以
`conditional_sample_from_noise` 跑 epsilon checkpoint **不需要改一行**。

缺的不是采样，而是适配层不知道自己在跟哪种参数化对话。已补上：

| 改动 | 作用 |
|---|---|
| `GAPBasePolicyWrapper.prediction_type` / `.track_prediction_type` | 从 `noise_scheduler.config` 读出并暴露 |
| `env.py` 启动时打印 | 每个 run 都记录基座参数化与去噪步数 |
| `train_dsrl_scan_object.py` 的 `_validate_noise_parameterization()` | 按 checkpoint 的 `prediction_type` 校验噪声设置 |

校验规则：

- **epsilon + `squash_actions=True` → 直接报错**。这是唯一一个会**恰好浪费掉 epsilon 所恢复性质**的组合：tanh 把噪声硬界在 ±`action_magnitude`，于是发不出 N(0,1) 中超过 ±1.5 的那 13.3%（实测 `P(|a|>1.5)` 精确为 0 vs 0.133，kurtosis 1.61 vs 3.01）。此前它能静默跑通。
- **epsilon + `standard_gauss_init=False` → 报错**：未训练 actor 就不是精确 N(0,1)，预填和 step-0 基线也就不是它们被文档声称的「未引导参考」。
- **epsilon + `target_ent` 偏离 363.2 → 警告**。363.2 = `256 × 0.5·log(2πe)`，即 N(0,1)^256 的熵。它在 epsilon 下是**有原则的**目标：把**每状态**的离散度按在训练分布的水平，而**均值**仍可随状态变化 —— 而那个均值就是引导本身。这与 squashed 设定下 0.0 / 174.5 两边都错的情形不同。
- **sample → 打印 authority 警示**（0.0035 @10 步、0.0463 @1 步），提醒不要把该 run 的结果当作 RL 结论读。

`DSRL_ALLOW_PARAM_MISMATCH=1` 可把报错降为警告。

**GAP 侧要训一个 epsilon checkpoint**，改两处（两条流必须一致，`gap.py:2276` 对 `v_prediction` + `joint_diffusion` 会直接报错）：

```yaml
# gap_policy/config/GAP.yaml:82 与 :101
prediction_type: epsilon
```

或命令行覆盖：

```
policy.noise_scheduler.prediction_type=epsilon policy.track_noise_scheduler.prediction_type=epsilon
```

`clip_sample` 两处都不用动：action 的 normalizer 是 `limits` 模式 → [−1,1]，所以 range 1.0 正好等于数据范围；track 的 `clip_sample_range=10.0` 裁的是同一个归一化 track 空间里的量，epsilon 下裁的是导出的 x̂₀，理由不变。

---

1. **用 `prediction_type='epsilon'` 重训 GAP。** 在全步数下恢复 DSRL 前提：x_T → x_0 变回近似可逆的学习流，而非朝条件均值的收缩。训练代码**已支持**（`gap.py:2227` action target 与 `:2276` joint_diffusion track target 都有 epsilon 分支，只有 `v_prediction` 对 joint_diffusion 不支持），所以是配置改动 + 一次完整重训，可能同时改善 GAP 本身。
2. **引导别的东西**：冻结策略输出上的残差动作 RL，或直接 RL 微调权重。放弃 DSRL「不碰权重」的卖点，但不受噪声不敏感性限制。
3. **转回策略本身**：条件化基线 0.72-0.75，无条件 0.57；提升策略比引导它更有价值。

若在 epsilon 重训后再试 DSRL，有参考依据的配置差异（**但在当前 checkpoint 上都不解决问题**）：

| 参数 | 本项目 | `DSRL-reference`（robomimic） | `DSRL-pi0-reference` |
|---|---|---|---|
| 基座参数化 | **`sample`** | `epsilon` | flow matching |
| 去噪步数 | 10 | 8 | — |
| `n_envs` | **1** | 4 | 16 |
| `n_critics` | 2 | 2 | **10** |
| actor 延迟 | 无 | 无 | **每 10 个 critic 步** |
| `discount` | **0.99** | — | 0.999 |
| 噪声维度 | **256** | 28 | 32/步 |
| `target_ent` | 0.0 → 174.5 | 0.0 | −16 |
| 预填步数 | 1500 | 2001 | — |

已确认 **RLinf 没有 RoboTwin + DSRL 的样例**（那里 RoboTwin 是 PPO/GRPO on VLA，DSRL 在 LIBERO-Spatial），没有现成配方可抄。

---

## 9. 复现

必须在 **RoboTwin conda 环境**、从 **`ROBOTWIN_ROOT`**、且在 **tmux 里**运行（否则 session 结束进程被杀）。

```bash
PY=/home/veraxiao/miniconda3/envs/RoboTwin/bin/python3
cd /data/vxiao/bimanual_manipulation/RoboTwin
D=/data/vxiao/bimanual_manipulation/dsrl-scan-object

$PY $D/tests/measure_expert_pass_rate.py --start_seed 100200000 --n 30  # 专家通过率，最便宜
bash $D/scripts/run_unsteered_comparison.sh    # 未引导基线，两个口径
bash $D/scripts/run_inference_steps_sweep.sh   # authority vs 去噪步数
bash $D/scripts/run_outcome_sensitivity.sh     # 结果敏感性（最关键）
bash $D/scripts/run_sensitivity_control.sh     # 确定性对照 + 修复后重测
bash $D/scripts/run_record_videos.sh           # 录像
```

产物：`dsrl-scan-object/results/*.json`、`results/videos/*.mp4`、`outputs/*.out`。

> 读 `.out` 必须过滤：`tr -d '\000' < f.out | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*m//g'`。`take_action` 用 `end="\r"` 打印 `step: N / 500`，不处理的话一个 episode 就是一行几十 KB 的 ANSI 刷屏。
