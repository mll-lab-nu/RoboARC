# RoboARC 外观种子与 9 个新底档：交接说明

2026-10-21 · jianwen（GitHub LUIGIVAMPER）

我们今天分三批推到 RoboARC 的 `dev/jianwen`，每批都接在上一批之后，不改写历史：

| 批次 | 提交号 | 内容 |
|---|---|---|
| 第一批 | `d6f2f23` | 接在 `f5a3d6f` 之后，19 个提交：外观种子、runner、派生工具、目标图，以及 9 个新底档 |
| 第二批（WIP） | `ae3be16` | 5 个提交：同步 TopoGym `3eed5d8b33`；重新生成 9 个新底档的题面；补测试文件；推送说明和证据；本文档初稿 |
| 第三批（WIP） | 推完后另行通知 | 同步 TopoGym `a849170272`：颜色对照表引出的颜色修正，以及 Claws 06–10 的题面修改；颜色对照表；本文档定稿 |

**请等我们通知第三批的提交号后再开始集成。** 原因见第 3 节：第二、三批都改动 `source/topogym/`，如果提前开工，有一部分工作要重做。

---

## 1. 你们回复里要求我们做的，现在的状态

| 要求 | 状态 | 位置 |
|---|---|---|
| 外观种子代码：`scene_variation.appearance`、各族外观参数、S3 的物理固定 | 第一批已推，第二批同步到 TopoGym `3eed5d8b33` | `source/topogym/` |
| `ci_checks/test-appearance-seeds.py` | 第二批已加，连同 `test-release-seeds.py` | `source/ci_checks/`，运行方法见 `docs/handover/jianwen_push_20261021.md` |
| 派生工具 `derive_seed_instance`、release_rebuild 的种子支持 | 第一批已推 | `source/scripts/robotangle/derive_seed_instance.py`、`source/scripts/robotangle/release_rebuild.py` |
| 目标图脚本：`fold_target_images.py`、种子版 `region_images.py` | 第一批已推 | `scripts/roboarc/goal_images/` |
| runner、`task_seed.py`、`write_task_results.py`（R1–R4） | 第一批已推 | `scripts/run_agent_batch.sh`、`scripts/task_seed.py`、`scripts/write_task_results.py`。D2 的结果目录分层和负责人的决定不一致，见第 4 节 |
| 9 个新底档的参考轨迹、录制工具、证据 | 第一批已推，第二批补上换电池和卡尺的最新外观 | 参考解：`scripts/roboarc/oracles/<底档>/privileged.npz`，每个种子的重新盖章版是 `privileged.s1.npz`、`privileged.s2.npz`。录制工具：`source/scripts/robotangle/references/<族>/`。证据见 `CHANGELOG.md` 和 `docs/handover/evidence_20261021/` |
| 推完停掉 ws2、ws3 上的派生、回放、Harbor 作业 | 已确认 | 两台机器上都没有我们的 RoboARC 作业。ws2、ws3 上只在跑 TopoGym 这边的开发作业，比如接线任务的验收轮和它的 Harbor 测试，都用自己的镜像名，不碰 RoboARC |
| 颜色对照表（底档 × 种子） | 已做完，第三批交 | 结论见第 4 节 |
| F4：新底档题面里的旧文档路径 | 第二批已进 | 9 个新底档 18 个文件改为 `docs/PROTOCOL.md`、`docs/STANDARD_MODE.md` |
| D11：cast-coil-06 保持 0.047 s | 不用改 | RoboARC 树里 06 两种模式本来就是 0.047。0.039 只出现在 TopoGym 的旧文本里，不会同步过去 |
| 新底档先给负责人看演示 | 演示页已做好 | 九个新底档各一段 `cam_high` 参考解回放，单机位、原始画面，链接由 jianwen 单独发给负责人 |

## 2. 已推内容里要注意的两点

- **我们在 1.0.x 上自己试做的"1.2.0"**：`images/roboarc/VERSION`、按它重新生成的任务目录，以及 `CHANGELOG.md` 里的 1.2.0 条目，都标为 PRE-RELEASE，不是正式版。正式 1.2.0 以你们在 1.1.0 上做的为准。这些提交不改写，只在推送说明里注明。
  - CHANGELOG 里写的镜像 `ghcr.io/embodied-first-exam/roboarc:1.2.0`（`sha256:301adcb64b06`）只存在于我们本地，没有推到 GHCR。为免和你们的正式版混淆，已在三台机器上改名为 `roboarc:1.2.0-jianwen-pre`。
- **换电池、卡尺的冻结实例已经过时，种子 0 也包括在内**：
  - 第二轮外观改了这两个底档在种子 0 下写出的文件：换电池的 `controller.xml`、`stick.xml`；卡尺的 11 个 GLB 贴图，分别是 4 个料盒、支架、4 个套筒和套筒架。
  - 所以第一批里这两个底档的冻结实例，种子 0、1、2 都算，以及 `assets.json` 里对应的固定哈希，都要重做：先重新生成种子 0，再派生种子 1、2。
  - 出生状态和参考解不变：两族第二轮外观的自查显示出生哈希、参考解终态都与改动前相同。
  - 改动前后的文件对照见 `docs/handover/evidence_20261021/seed0_declared_files_before_after_2b.md`。

## 3. 第二、三批（WIP）的内容，以及它们对你们集成和验证的影响

WIP 是 work in progress（进行中）的缩写，表示这个提交还不是最终版。你们要求"没做完的也推，注明 WIP"，所以这两批都这样标。

| 内容 | 批次 | 改到哪里 | 如果你们提前开工，要重做什么 |
|---|---|---|---|
| `source/topogym/` 同步到 TopoGym `3eed5d8b33`，包括换电池、卡尺第二轮外观和 F4；重新生成 9 个新底档的题面；补两个测试文件 | 第二批 | `source/topogym/` 的字节变了，所以 SOURCE.json、binding 和任务目录都要跟着变 | 合并源码、建镜像、重新生成任务目录、重新派生，都要再来一遍。换电池、卡尺两个底档要像第 2 节说的那样从种子 0 重做 |
| 外观修正，种子 0 都逐字节不变：<br>• 卡尺种子 1、2 的料盒改成深红、深绿；<br>• 换电池的撬棒只在尼龙色（黑、橙、蓝）里变，种子 1 是白色手柄，种子 2 是 Pulse Red 配色（白色背面朝上、红色肩键），红色 "+" 在两个种子下都落在白色背面上；<br>• Cast Trinity 新增只影响外观的参数 `third_rgb`，三件在每个种子下颜色各不相同；<br>• 穿绳族种子 1、2 的绳子改成蓝色、黑色，和所有环、扣眼拉开 | 第三批 | 这几个族的外观参数；`scene_variation.py`、`scene_visuals.py`（新增 `blue_rope`、`black_rope`）、interlocked_claws 的上色代码 | 卡尺、换电池、Trinity、rope-threading 01–08 的种子 1、2 场景要重新派生。其他 12 个 Claws 底档和 slotted-post-ring 在每个种子下都不变 |
| Claws 06–10 的题面去掉 "teal"（详见第 4 节） | 第三批 | 5 个已发布底档的题面，两种模式都改 | 这 5 个底档的任务目录要重新生成 |
| 颜色对照表 | 第三批 | 只是一份文档 | 没有影响 |

**物理验证不受这三批影响。** 这些改动只动可视几何、颜色和题面，101 个已发布底档的物理代码路径没有变。

**结论**：只要你们等第三批的提交号再开始集成，就不会有白做的部分。我们推完第三批会马上通知。

## 4. 需要你们集成的

你们回复第 5 节的路径对照表照常使用，`source/topogym/` 以外的文件按那张表移动。另外几点：

1. **D2，结果目录的分层。** 我们现在的实现按种子分层：`s<S>`，同一种子多局时是 `s<S>-p<n>`，每局另有 `seed.json`。负责人定的是按局分层、种子写进 `config.json`，需要在集成时改。结果网站、HF 读取程序、regrade、`/validate` 也要在同一版里跟着改。
2. **`stage_instance.py`。** 在 1.1.0 里它归 `source/images/roboarc/Dockerfile` 管。我们在 TopoGym 侧的 `--seed-file` 改动要合到那里。
3. **T5、T1。** 每个任务目录放一个显式的 `environment/seed`，默认是种子 0，materialize 时覆盖。`oracle` 只列出跑过 Harbor oracle 的种子。
4. **网关、`__main__`、F3。** 你们那边已经收了，变基时去掉即可。我们没有再改这几处。
5. **本地试做的 1.2.0 任务目录和 VERSION**（PRE-RELEASE 提交）：集成时以你们重新生成的为准。
6. **自 `e3ea480` 以来的 TopoGym 源码差异。** 你们复核物理用的 `e3ea480` 同步的是 TopoGym `497a9e9d7f`。之后进来的有：
   - 换电池规则 C 和手柄外壳：用到一处公共改动 `3307d319a1`，`WeakHoldGraspTarget` 的弱抓持改为每个目标整局共用一份移动额度。只有换电池用它；
   - 钟表 v3；
   - 外观种子代码本身；
   - 换电池、卡尺的第二轮外观；
   - F4 的题面路径修正；
   - 第三批的颜色修正和 Claws 06–10 的题面修改。

   公共代码改动逐项列在 TopoGym 仓库的 `docs/rcb/handover/jianwen_push_20261019.md`，第 2 节，外观种子单列在 2.7 节。我们自己在 TopoGym 侧的中立性核对比较的正是 `497a9e9d7f` 和 `9115da07d8`，见第 5 节。
7. **颜色对照表的结论**：110 个底档里有 79 个的题面点名了颜色或材质，整理成 35 行；另外 31 个完全不提颜色。
   - **题面说错的一处：Claws 06–10 的 "the other piece is a teal lock"。** 种子 1 把锁画成蓝色，种子 2 画成绿色。用户决定改题面、不改颜色：两种模式的 instruction.md、README.md、task.toml 改成 "the other piece is a lock"，release 记录里的 design 改成 "a brown key and a lock"。棕色的钥匙在每个种子下都还是棕色，那句保留。**这是对 5 个已发布底档题面的修改，请审阅。**
   - **说法模棱两可的一处：换电池的 "a nylon spudger"。** 种子 1 曾把撬棒画成木色，第三批改为撬棒只在尼龙色里变。
   - **颜色太像、但题面没说错的三处，第三批都已修正**：换电池种子 2 的红色外壳盖住了电池仓边框上的红色 "+"；Cast Trinity 种子 1、2 有两件同色；穿绳种子 1 的橙绳接近红色扣眼，种子 2 的金绳接近琥珀色环。做法见第 3 节。
   - **顺带发现、和种子无关的一处：fiber-bundling 05–10 有 16 根光纤，颜色只有 8 种循环**，所以 07–10 题面里的 "the red fiber" 其实对得上两根，三个种子都一样。要不要改，请你们定。
   - 你们点名的其余颜色在种子 1、2 下都还成立：fiber 的铜环和红色光纤不随种子变；claws 的棕色件只变深浅；cast 的金、银只变亮度、色相不变；orbits、buzz 的木色只变亮度或不变；各族的绿色目标区域不变。

## 5. 需要你们验证的，以及我们已有的证据

我们已有的证据都在本地试做的 1.2.0 上（建在 1.0.x 上），详见 `CHANGELOG.md` 1.2.0 一节的 Evidence 和 `docs/handover/evidence_20261021/`。正式 1.2.0 要在你们的镜像上重做。

| 证据（你们回复第 3、4、5 节） | 我们已有的 | 还要你们做的 |
|---|---|---|
| 种子 0 不变：instance_id、出生、终态、渲染帧，101 个底档 | 外观种子分支自查：种子 0 写出的实体文件与改动前相同。`test-appearance-seeds.py`：物理内容在各种子间不变，种子 1、2 互不相同。中立性核对：14 个底档（12 个已发布族各一个，加 bed-bolt-01、caliper-sorting-01）在 TopoGym `497a9e9d7f` 和 `9115da07d8` 上的出生和严格回放终态逐位相同 | 101 个底档全量证明，包括渲染帧 |
| blanket 14 个底档单独证明种子 0 不变（布料身份不再计入外观） | 无 | 全部 |
| 每个派生种子的物理部分与种子 0 逐位相同 | 220 个派生种子实例（110 个底档 × 种子 1、2）都过了三道检查：物理字段等于种子 0 的；出生时的 tick 记录只有 `task.instance_id` 与种子 0 不同；出生时的 world 哈希等于种子 0 的 | 终态的物理部分，按你们的四项检查；换电池、卡尺按新字节重做 |
| 每个派生种子严格两次新进程回放 | 23 个底档（每族一个，加 9 个新底档）在种子 0 和 1 上各回放一次：46/46 成功。换电池、钟表在新源码上种子 0 和 1：4/4 成功，种子 0 停在参考解终态，两个种子的 world 哈希相同 | 全部种子，每个两次新进程 |
| 派生场景通过严格场景检查 | 在镜像里用 `stage_instance.py --seed-file` 部署换电池 01、horseshoe-ring-01、blanket-fold-01 的种子 0、1、2，再开严格 Session：9/9 instance_exact 和 assets_exact，出生与记录相同 | 其余底档 |
| 种子 1、2 的 Harbor：两种模式 oracle 1 / nop 0，每族至少一个底档，加上全部新底档 | 9 个新底档（种子 0），两种模式 oracle 和 nop：36/36。种子副本（`task_seed.py materialize`）：interlocked-claws-03、hedgehog-in-cage-02 的种子 1、2 共 16/16，blanket-fold-01 的种子 1、2 共 8/8，rope-threading-04 种子 1 的 privileged oracle 为 1 | 每族至少一个底档、全部新底档的种子 1、2，在正式镜像上 |
| 9 个新底档在 Zen 4 和 Intel 两类 CPU 上回放 | 只有 Zen 4：我们的三台机器都是 Ryzen 9 7950X。每个新底档都有确定的参考解和 1000 局压力验收轮 PASS（判据 v2.13b） | Intel 上的回放 |
| 颜色对照表 | 第三批交，表头注明对应的 TopoGym 提交；结论见第 4 节第 7 条 | 审阅，尤其是 Claws 06–10 的题面修改 |

## 6. 新底档入库

- 9 个新底档在负责人看过演示之前保持 experimental。
- 演示页里每段是参考解整局的严格回放，用 Harbor 评分录像的同一机位 `cam_high`。钟表、换电池、卡尺三段是 10-21 按最新外观重新渲染的，出生和终态都核对过。
- 换电池的一个已知限制：电池仓边框上的红色 "+" 只有 3.6 mm，在 `cam_high` 原分辨率下大约 1 个像素，种子 0 也一样。从 `cam_high` 判断电池方向，看的是地面上 6 mm 的红 "+" 和电池本身的 "+"；腕部相机里所有标记都清楚。要把边框标记做大就会改动种子 0，这次没有改。
- 新族的问题直接问 jianwen。
