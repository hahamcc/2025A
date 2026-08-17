# Q4补：分类种子 + 全域 LHS + 联合 DE

本目录是对 `Q4/` 的独立补充实验。它只读取原 Q4 已验证的物理常量、运动模型和联合完整遮蔽评价器；不会改写 `Q4/` 下的任何代码、CSV 或 Excel。

分类标签仅用于初始种群构造：全程接力、交叠接力、空间分工探索和冗余集中。进入差分进化后，所有个体都在完整 12 维可行域中最大化同一个联合完整遮蔽时长 \(T_{\rm joint}\)，不存在模式硬约束。

## 正式运行

```powershell
D:\anaconda\python.exe Q4补\q4_mode_hybrid_main.py `
  --profile standard `
  --init-strategy mode-hybrid `
  --output-dir Q4补\runs\mode_hybrid_standard
```

## 与原初始化方式对照

```powershell
D:\anaconda\python.exe Q4补\q4_mode_hybrid_main.py `
  --profile standard `
  --init-strategy baseline `
  --output-dir Q4补\runs\baseline_compare
```

如需覆盖同名实验输出，显式加入 `--overwrite`。先检查约束和初始种群而不运行 DE：

```powershell
D:\anaconda\python.exe Q4补\q4_mode_hybrid_main.py --profile quick --check-only
```

每次正式运行在指定输出目录生成初始种群、分类摘要、DE 搜索记录、收敛记录、最终方案 CSV、`result2.xlsx` 副本和运行摘要。最终结论只应以高密度完整圆柱复核结果为准。

默认会以 8 个时间点为一批完成表面距离计算，避免最终 `720×21×15` 加密复核一次申请数 GB 的内存。若机器内存仍紧张，可进一步减小批大小，例如加入 `--time-batch-size 4`；这不会改变模型、采样点或数值结果，只会增加运行时间。

如需只增加 DE 代数、而保持标准档其他参数不变，可加入例如 `--maxiter 180`。建议将该实验写入新的输出目录，便于与 90 代版本比较。
