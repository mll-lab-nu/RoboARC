# 实例反推生成器（instance_optimizer）

2026-10-09：新增链式锁（Lock Picking v2 五档，`tasks/lock_picking/chain.py`）的生成器 `lock_picking_chain.py`。它从 oracle 要走的钥匙路线出发，先确认机械臂能完整走完这条路线，再由路线反推锁的槽位、限位和诱饵槽。五档现有参数都由它逐项复现（`ci_checks/test-lock-picking-chain.py` 检查）。

## 物理图景

桌上沿钥匙轴排着 N 片可转锁片，片距 120 mm，最后是固定的 END 板。钥匙杆尖有长齿，后方一个片距处有短齿；长齿只过深槽，短齿深浅槽都能过。钥匙插到"第 L 层"时长齿在第 L 片里、短齿在第 L−1 片里，转钥匙就带着第 1 片（L = 1）或这一对相邻片一起转。每片只能在限位行程内转。

## 生成顺序（`optimize()` 固定按此执行）

1. **路线**：每档的 `ChainSpec.steps` 给出钥匙的完整动作：进到第几层、在第几层把锁片转到多少度。棘轮类路线由 `ratchet_steps` 生成：末片要转的角度由前一片分几次带过去，前一片转到行程端点后，钥匙退回低层把它转回另一端，再回来继续带。very_easy 的路线直接写出。由路线算出每次进层时各片的角度，再给每片定一条长齿槽（代码里叫 passage）：第 L 层时钥匙角度 = 第 L 片角度 + 这条槽的角度。钥匙全程的插入深度和转角在这一步就全部确定。
2. **IK**：在一把只开了这些长齿槽、没有短齿槽和诱饵槽的锁上（设计用变体 `lock_picking_v2_design`，任务参数可给任意链式锁），把整条路线（默认 home → 拾取 → 全部插拔转动 → END）按 ≤ 2 mm、≤ 1° 采样逐点求只读 IK，检查关节限位和相邻采样连续，并用 FCL 算机器人与场景的最小间隙，要求不小于 5 mm。槽都在离轴 45 mm 以内，机器人到不了，所以这一步的结果与槽无关。
   - 5 mm 的来源：物理执行实测的间隙与 IK 算出的相差约 1 mm（easier 在原托架上：IK 0.19 mm，执行 1.06 mm），要给机械臂跟踪误差留出余量。
3. **锁**：IK 通过后才生成其余部分：
   - 短齿槽：每次进层时短齿要经过的位置；
   - 限位：路线用到的角度两侧各加 `stop_pad_deg`；
   - 诱饵槽：按 `ChainSpec.decoys` 的顺序逐条用 `chain_route.add_decoys` 检查（最短步数、倾斜上界下的步数、所有最短解用到的槽对都不变，且钥匙确实能经它进入某层），有一条不合格就不接受。

   最后用精确搜索给这把锁出证书。要求搜索得到的最短解与第 1 步的路线逐步相同、唯一，并且在倾斜上界下步数不变；全部满足记为 `accepted`。

## 运行

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. .venv-gs121-nyx/bin/python scripts/optimize_lock_chain_instance.py \
  --output runs/<新目录> [--specs v2_easy v2_hard]
```

每档写 `optimized.json`（路线、IK 报告与间隙、锁参数、保留的诱饵、证书、`accepted`，以及与已登记同名档是否一致的 `matches_tier`）和 `ik_samples.json`。hard 的精确搜索和诱饵检查要几分钟。2026-10-09 五档的运行记录在 `runs/test_results/lock_chain_v2_20261009/instance_optimizer/`。

## 新增一档

在 `SPECS` 里加一条 `ChainSpec`，运行上面的脚本。`accepted` 为真后，把 `optimized.json` 里的 `parameters` 写进 `tasks/lock_picking/chain.py` 登记为新档，再用 `scripts/plan_chain_lock_oracle.py --ik`（完整锁上的证书与 IK）和 `scripts/oracle_runners/run_lock_picking_policy.py --route-mode chain`（物理执行）复核。
