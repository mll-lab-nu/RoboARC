# RoboARC 外观种子与 9 个新底档：交接说明

2026-10-21 · jianwen（GitHub LUIGIVAMPER）

第一批已推到 RoboARC 的 `dev/jianwen`，提交号 `d6f2f23`。它接在 `f5a3d6f` 之后，共 19 个提交，没有改写历史。

今天还会补推第二批，提交信息里标 WIP。**请以第二批推完后的提交号为准开始集成。** 原因见第 3 节：第二批改动了 `source/topogym/`，如果在它之前开工，有一部分工作要重做。

---

## 1. 你们回复里要求我们做的，现在的状态

| 要求 | 状态 | 位置 |
|---|---|---|
| 外观种子代码：`scene_variation.appearance`、各族外观参数、S3 的物理固定 | 已推 | `source/topogym/`：同步自 TopoGym `738231f5cb`，即 dev `9115da07d8` 加外观种子 |
| `ci_checks/test-appearance-seeds.py` | **第二批补** | 第一批漏了，RoboARC 树里只有你们的 `ci_checks/check-seeds.py`。第二批一并补上 `test-release-seeds.py` |
| 派生工具 `derive_seed_instance`、release_rebuild 的种子支持 | 已推 | `source/scripts/robotangle/derive_seed_instance.py`、`source/scripts/robotangle/release_rebuild.py` |
| 目标图脚本：`fold_target_images.py`、种子版 `region_images.py` | 已推 | `scripts/roboarc/goal_images/` |
| runner、`task_seed.py`、`write_task_results.py`（R1–R4） | 已推 | `scripts/run_agent_batch.sh`、`scripts/task_seed.py`、`scripts/write_task_results.py`。D2 的结果目录分层和负责人的决定不一致，见第 4 节 |
| 9 个新底档的参考轨迹、录制工具、证据 | 已推；换电池和卡尺的最新外观在第二批 | 参考解：`scripts/roboarc/oracles/<底档>/privileged.npz`，每个种子的重新盖章版是 `privileged.s1.npz`、`privileged.s2.npz`。录制工具：`source/scripts/robotangle/references/<族>/`。Harbor 证据见 `CHANGELOG.md` |
| 推完停掉 ws2、ws3 上的派生、回放、Harbor 作业 | 已确认 | 两台机器上都没有我们的 RoboARC 派生、回放或 Harbor 作业。ws2 上只有 TopoGym 这边的开发作业，不碰 RoboARC |
| 颜色对照表（底档 × 种子） | 在做，第二批 | 见第 3 节 |
| F4：新底档题面里的旧文档路径 | TopoGym 里已改，随第二批的同步进来 | TopoGym `9e36ecf429`：9 个新底档 18 个文件改为 `docs/PROTOCOL.md`、`docs/STANDARD_MODE.md` |
| D11：cast-coil-06 保持 0.047 s | 不用改 | RoboARC 树里 06 两种模式本来就是 0.047。0.039 只出现在 TopoGym 的旧文本里，不会同步过去 |
| 新底档先给负责人看演示 | 演示页已做好 | 九个新底档各一段 `cam_high` 参考解回放，单机位、原始画面，链接由 jianwen 单独发给负责人 |

## 2. 第一批（`d6f2f23`）里有什么

- 外观种子：K = 3。种子 0 就是已发布的场景，种子 1、2 只改颜色和材质，物理与种子 0 相同。
- runner：同一任务的 K 局不放回地分配种子（D1），每局的种子写在 `<job>/seed.json`。
- 每个种子一份重新盖章的参考解（D3 选 (a)）：`solve.sh` 按种子挑选。
- 派生工具：种子 1、2 的场景从种子 0 的冻结实例派生，不重新生成。
- 外观改动能真正画出来：hedgehog、buzz 重设计档、orbits 的颜色改动现在会改到 Nyx 实际读取的文件。
- 目标图按种子各出一份，例如 blanket-fold、knot-untying 07/09。
- 9 个新底档：bed-bolt-01～05、clock-assembling-01、battery-replacement-01、caliper-sorting-01、interlocked-claws-13（Cast Trinity）。
- **我们在 1.0.x 上自己试做的"1.2.0"**：`images/roboarc/VERSION`、按它重新生成的任务目录，以及 `CHANGELOG.md` 里的 1.2.0 条目，标为 PRE-RELEASE。它不是正式版，正式 1.2.0 以你们在 1.1.0 上做的为准。这些提交不改写，只在推送说明里注明。
  - CHANGELOG 里写的镜像 `ghcr.io/embodied-first-exam/roboarc:1.2.0`（`sha256:301adcb64b06`）只存在于我们本地，没有推到 GHCR。为免和你们的正式版混淆，已在三台机器上改名为 `roboarc:1.2.0-jianwen-pre`。
- **已经过时的部分**：第一批里换电池、卡尺两个底档的种子场景和 `assets.json` 里对应的固定哈希，是在两族第二轮外观之前做的。第二批同步后需要按新字节重新派生。

## 3. 今天补推的第二批（WIP），以及它对你们集成和验证的影响

WIP 是 work in progress（进行中）的缩写，表示这个提交还不是最终版。你们要求"没做完的也推，注明 WIP"，所以这批会这样标。

| 第二批的内容 | 改到哪里 | 如果你们已经开工，要重做什么 |
|---|---|---|
| 把 `source/topogym/` 同步到 TopoGym 最新的 dev，并重新生成任务目录。包括换电池和卡尺的第二轮写实外观（都已接上外观种子）、F4 的题面修正，以及上面漏推的两个测试文件 | `source/topogym/` 的字节变了，所以 SOURCE.json、binding 和任务目录都要跟着变 | **机械部分要重来**：合并源码、建镜像、重新生成任务目录、重新派生。**物理验证不受影响**：这些改动只动换电池、卡尺的可视几何和颜色，以及新底档的题面，101 个已发布底档的代码路径没动。换电池、卡尺两个底档的派生种子和 Harbor 要按新字节重跑 |
| 卡尺种子 1、2 的料盒改成深色（种子 0 不变） | 卡尺的外观参数 | 只影响 caliper-sorting-01 的种子 1、2 |
| 颜色对照表 | 只是一份文档 | **没有影响**。只有当它查出某段题面在种子 1、2 下说错了颜色，修正题面或外观参数才会再动代码，那样我们会在这批里一起改好再交 |

**结论**：三件事今天都会推完。只要你们从第二批的提交号开始集成，就不会有白做的部分。我们推完会马上通知提交号。

## 4. 需要你们集成的

你们回复第 5 节的路径对照表照常使用，`source/topogym/` 以外的文件按那张表移动。另外几点：

1. **D2，结果目录的分层。** 我们现在的实现按种子分层：`s<S>`，同一种子多局时是 `s<S>-p<n>`，每局另有 `seed.json`。负责人定的是按局分层、种子写进 `config.json`，需要在集成时改。结果网站、HF 读取程序、regrade、`/validate` 也要在同一版里跟着改。
2. **`stage_instance.py`。** 在 1.1.0 里它归 `source/images/roboarc/Dockerfile` 管。我们在 TopoGym 侧的 `--seed-file` 改动要合到那里。
3. **T5、T1。** 每个任务目录放一个显式的 `environment/seed`，默认是种子 0，materialize 时覆盖。`oracle` 只列出跑过 Harbor oracle 的种子。
4. **网关、`__main__`、F3。** 你们那边已经收了，变基时去掉即可。我们没有再改这几处。
5. **本地试做的 1.2.0 任务目录和 VERSION**（PRE-RELEASE 提交）：集成时以你们重新生成的为准。
6. **自 `e3ea4803` 以来的 TopoGym 源码差异。** 你们复核物理用的是 `e3ea4803`。之后进来的有：
   - 钟表 v3；
   - 换电池规则 C：用到一处公共改动，`WeakHoldGraspTarget` 的弱抓持改为每个目标整局共用一份移动额度。只有换电池用它；
   - 卡尺判据 v2；
   - 床架外观，只动可视几何；
   - 床架和 Trinity 的题面修正；
   - 外观种子代码本身；
   - 第二批的两族外观。

   公共代码改动逐项列在 TopoGym 仓库的 `docs/rcb/handover/jianwen_push_20261019.md`，第 2 节，外观种子单列在 2.7 节。

## 5. 需要你们验证的，以及我们已有的证据

我们已有的证据都在本地试做的 1.2.0 上（建在 1.0.x 上），详见 `CHANGELOG.md` 1.2.0 一节的 Evidence。正式 1.2.0 要在你们的镜像上重做。

| 证据（你们回复第 3、4、5 节） | 我们已有的 | 还要你们做的 |
|---|---|---|
| 种子 0 不变：instance_id、出生、终态、渲染帧，101 个底档 | 外观种子分支自查：种子 0 写出的实体文件与改动前相同。`test-appearance-seeds.py`：物理内容在各种子间不变，种子 1、2 互不相同。中立性核对：14 个底档（12 个已发布族各一个，加 bed-bolt-01、caliper-sorting-01）在 TopoGym `497a9e9d7f` 和 `9115da07d8` 上的出生和严格回放终态逐位相同 | 101 个底档全量证明，包括渲染帧 |
| blanket 14 个底档单独证明种子 0 不变（布料身份不再计入外观） | 无 | 全部 |
| 每个派生种子的物理部分与种子 0 逐位相同 | 220 个派生种子实例（110 个底档 × 种子 1、2）都过了三道检查：物理字段等于种子 0 的；出生时的 tick 记录只有 `task.instance_id` 与种子 0 不同；出生时的 world 哈希等于种子 0 的 | 终态的物理部分，按你们的四项检查 |
| 每个派生种子严格两次新进程回放 | 23 个底档（每族一个，加 9 个新底档）在种子 0 和 1 上各回放一次：46/46 成功。换电池、钟表在新源码上种子 0 和 1：4/4 成功，种子 0 停在参考解终态，两个种子的 world 哈希相同 | 全部种子，每个两次新进程 |
| 派生场景通过严格场景检查 | 在镜像里用 `stage_instance.py --seed-file` 部署换电池 01、horseshoe-ring-01、blanket-fold-01 的种子 0、1、2，再开严格 Session：9/9 instance_exact 和 assets_exact，出生与记录相同 | 其余底档 |
| 种子 1、2 的 Harbor：两种模式 oracle 1 / nop 0，每族至少一个底档，加上全部新底档 | 9 个新底档（种子 0），两种模式 oracle 和 nop：36/36。种子副本（`task_seed.py materialize`）：interlocked-claws-03、hedgehog-in-cage-02 的种子 1、2 共 16/16，blanket-fold-01 的种子 1、2 共 8/8，rope-threading-04 种子 1 的 privileged oracle 为 1 | 每族至少一个底档、全部新底档的种子 1、2，在正式镜像上 |
| 9 个新底档在 Zen 4 和 Intel 两类 CPU 上回放 | 只有 Zen 4：我们的三台机器都是 Ryzen 9 7950X。每个新底档都有确定的参考解和 1000 局压力验收轮 PASS（判据 v2.13b） | Intel 上的回放 |
| 颜色对照表 | 第二批交 | 审阅 |

## 6. 新底档入库

- 9 个新底档在负责人看过演示之前保持 experimental。
- 演示页里每段是参考解整局的严格回放，用 Harbor 评分录像的同一机位 `cam_high`。钟表、换电池、卡尺三段是 10-21 按最新外观重新渲染的，出生和终态都核对过。
- 新族的问题直接问 jianwen。
