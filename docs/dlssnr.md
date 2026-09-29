# DLSS-NR 310.8.0 训练适配

## English Summary

This is an experimental, standalone supervised-training path for DLSS-NR 310.8.0,
not a diffusion trainer or an NVIDIA DLL replacement. It provides canonical
weight conversion, FP32 full/LoRA training, still/closed-loop inference, adapter
merge, and checked optimizer-state resume. The three example configurations use
512 x 512 fixed buckets; inputs and targets must already match the configured size.

Users must provide their own weights, paired RGB data, encoded control lanes and,
for non-reset temporal frames, motion and validity masks. No weights, DLLs or game
assets are included. See [source attribution](../src/musubi_tuner/dlssnr/NOTICE.md).

Normal training requires a forward-validation report bound to the weights and
implementation. That validation is not complete: current experiments require
the explicit `--development_smoke` switch. The surrogate uses half/E4M3 forward
publications with surrogate gradients and FP32 master weights; it is not ordinary
FP32 SiLU/softmax and is not bit-identical to the DLL. Mixed precision,
distributed training and gradient checkpointing are rejected. Outputs precede
the DLL's Natural/Cinematic color grading.

Training writes all artifacts under `output_dir/output_name/`. A new run refuses
an existing run directory; use `--resume` for a matching checkpoint. Sample IDs
must be portable filenames, unique without regard to case. The commands and
manifest contract below apply to both full and LoRA training.

这条路径使用 FP32 主权重和矩阵累加，用来做配对监督。中间激活、注意力和归约包含明确的 half/E4M3 fake quant 与替代梯度，不能再视作普通 FP32 SiLU/softmax 网络。它仍不是原版 DLL 的逐字节推理，输出也还在 proxy 颜色空间里，没有 DLL 后面的 Natural / Cinematic 调色。

## 当前范围

根目录脚本保持 thin entry point。模型、数值、数据、训练损失和评估位于 `src/musubi_tuner/dlssnr/`，LoRA 位于 `networks/lora_dlssnr.py`，生命周期位于 `training/dlssnr_trainer.py`。全量和 LoRA 共用 `NRTrainModule`、Accelerate 包装及同一更新循环，不调用 diffusion trainer。

- 单进程 FP32，默认自动选择 CUDA，缺少 CUDA 时使用 CPU；`training.device` 可设为 `auto`、`cpu` 或 `cuda`。显式请求不可用的 CUDA 会报错。
- 单帧及一个有限时序段，支持全量/LoRA 梯度累积、定期保存与恢复。累积按有效元素总数归一化，计步单位是 optimizer update。
- 配置逐表校验，未知字段报错。配置路径以 TOML 所在目录为基准；最终配置和来源指纹写入 `run_config.json`。
- `data.validation_manifest` 和 `evaluation.sequence_manifest` 会实际执行，使用独立学生 history。默认比较初始基座，写入 `evaluation/stepNNNNNN.json`；评估不会改变训练 RNG 或 dropout 状态。
- manifest 按需解码，不将整个训练集常驻内存；推理和评估按帧回放。

原生参考前向、真实片段上的兼容性和质量验收尚未完成。布局字节 round-trip、浮点训练测试和原生兼容是不同结论。

2026-09-24 已完成第一轮 Steam 截图与本地 310.8.0 显卡适配版 DLL 的对照，结论是**当前 FP32 前向不一致**。4 张截图、19 组图片/参数组合覆盖 style、intensity、tone、structure、skin 和 auto mask；默认参数的输出与 DLL 之间 MAE 为 0.0294 至 0.0420。相同 packed 权重不代表前向计算正确，不建议据此开始正式训练。详见 [截图前向实验记录](dlssnr_forward_experiment_2026-09-24.md)。

同日后续已排除主要默认参数因素，并修复 cubic 激活及关键数值边界。默认参数下四图 MAE 降至 0.00262 至 0.00520，19 组中 9 组达到原有三项显示指标门槛（含一组零强度直通），**尚未完成全部前向验收**。全量测试为 172 passed，包含真实权重 CUDA 全量/LoRA 更新及合并对照。详见 [可训练前向对齐记录](dlssnr_forward_alignment_2026-09-24.md)。数值实现版本已更新，旧训练状态不能跨版本 exact resume。

## 正式运行与实验运行

正常训练必须提供 `model.model_dir`、完整 canonical 来源文件、通过 round-trip 的转换报告，以及绑定当前权重和数值实现的 forward validation report。缺少这些证据时，入口会停止，而不会隐式随机初始化。

当前开发实验须显式传 `--development_smoke`，或设置 `training.development_smoke = true`。这允许尚未通过参考验收的 canonical 权重；只有这种模式允许省略 `model_dir` 使用随机权重。输出会标记 `experimental_surrogate`，不应作为兼容模型发布。

验证报告默认读取 `model_dir/forward_validation_report.json`，也可通过 `model.forward_validation_report` 指定。契约为 `dlssnr_forward_validation_v1`，包含 `profile`、`numerics_profile`、当前 `model_sha256`、`implementation_sha256`、固定的 `reference_identity`、`float_validated` 和 `checks`。检查项为 `raw_head`、`neural_preclamp`、`rendered_proxy`；全部必须来自实际参考验证。实现指纹由 `json_sha256(implementation_identity())` 计算。普通训练 evaluation 不能代替这份报告，也不要手工填入通过标记。

## 命令

把自行准备的 OpenDLSS-NR `models/nr` 目录转成 canonical 权重：

```bash
python dlssnr_convert_model.py --source_dir ../OpenDLSS-NR/models/nr --output_dir models/canonical_dlssnr --verify_roundtrip
```

单帧全量、时序全量、ViT LoRA：

```bash
python dlssnr_train.py --config_file configs/dlssnr_full_single.toml --development_smoke
python dlssnr_train.py --config_file configs/dlssnr_full_temporal.toml --development_smoke
python dlssnr_train_network.py --config_file configs/dlssnr_lora_vit.toml --development_smoke
```

从某个 optimizer 更新边界继续。只拷权重、不带 optimizer 状态的加载是重新开跑，不是 resume：

```bash
python dlssnr_train.py --config_file configs/dlssnr_full_single.toml --development_smoke --resume output/dlssnr_full_single/dlssnr_310_8_0_full/state-step000100
python dlssnr_train_network.py --config_file configs/dlssnr_lora_vit.toml --development_smoke --resume output/dlssnr_lora/dlssnr_310_8_0_lora_vit/state-step000100
```

静止图和闭环片段。推理清单可以不写 target。片段里非 reset 帧必须有 motion 和 history mask：

```bash
python dlssnr_generate_image.py --model_dir models/canonical_dlssnr --sample_manifest data/inference_single.jsonl --bucket_width 512 --bucket_height 512 --numerics_profile train_surrogate --output_dir output/preview
python dlssnr_generate_video.py --model_dir models/canonical_dlssnr --sequence_manifest data/inference_sequence.jsonl --bucket_width 512 --bucket_height 512 --numerics_profile train_surrogate --output_dir output/video
python dlssnr_merge_lora.py --base_model_dir models/canonical_dlssnr --adapter output/dlssnr_lora/dlssnr_310_8_0_lora_vit/final/adapter.safetensors --output_dir output/dlssnr_merged
```

## 现在明确不做的事

混合精度、gradient checkpointing、多卡、FP8 基座、以及 SageAttention / FlashAttention / xFormers 都会直接报错。配置里把 `gradient_checkpointing` 写成 true，或者 Accelerate 用了 `--mixed_precision` 不是 `no`，也会停。

历史重投影用的是双线性，不是原版五点 Catmull-Rom。历史按 FP32 保存，不是向零截断到 float16。

像素缓存尚未实现，`require_cache = true` 会报错。当前不提供多 worker 预取，因此不宣称已经通过多 worker 的恢复验收。FP32 入口会暂时关闭 TF32，结束后恢复原来的全局设置。

## 数据约定

每条 JSONL 记录必须包含唯一且可用于文件名的 `sample_id`、`sequence_id`、`schema = dlssnr_pairs_v1`、`source_encoding = srgb_proxy`、`controls_encoding = dlssnr_lanes_10_14_v1`。训练及配对评估还需 `target_encoding = srgb_proxy` 和 target。首帧必须 reset，`frame_index` 非负且严格递增；训练与验证不能共享 sequence ID。

`sample_id` 在所有平台上按不区分大小写的规则检查重复，例如 `Frame` 和 `frame` 不能出现在同一清单中。名称不能包含路径分隔符、Windows 非法字符、控制字符、末尾句点或空格，也不能是 `NUL`、`CON` 等 Windows 保留名称。清单索引阶段即报错，不会写出部分推理结果后才发现冲突；原始 ID 不会被自动改写。

- RGB 支持 8-bit 图像，或 `[3,H,W]` 的数值 NPY，proxy 值域为 `[0,1]`。不将高位深图片静默压成 8-bit。
- controls 为有限值 `[5,H,W]` NPY。motion 的布局由记录级 `motion_layout = chw|hwc` 明确指定，单位是 current-to-previous 像素位移。
- mask 支持单通道 PNG 或 `[1,H,W]` NPY。history/temporal mask 必须为二值；可选 `loss_mask_path` 支持 `[0,1]` 权重。
- temporal loss 非零时，非 reset 帧必须提供 temporal mask，且 `tbptt_length >= 2`。损失只计算可反传段内部的相邻帧，不跨越 burn-in 边界。
- `loss.temporal = 0` 允许不提供 temporal mask，但非 reset 帧仍必须有 motion 和 history mask。

## 保存与恢复

每次训练使用 `output.output_dir/output.output_name/` 作为独立运行目录，`final/`、周期权重、训练状态、日志和评估报告均位于该目录下。`output_name` 必须满足上述可跨平台使用的名称规则，不能包含路径。新训练拒绝使用已存在的同名运行目录；继续训练须显式传入 `--resume`，重新训练须选择其他名称或输出目录。

旧产物不会自动搬迁。此次保存路径和校验逻辑变更也会改变训练实现指纹，旧版本状态不能直接 exact resume；需要重新开跑时保留旧文件，并按权重加载方式另建运行目录。

全量 `final/` 保存 FP32 权重、canonical 配置、来源 manifest、原始 opaque records、数值与预处理信息及训练元数据。LoRA 保存 adapter、rank/alpha/targets 和完整基座身份，不再只哈希 targets。合并时同样保留 canonical 来源与 opaque 数据。

LoRA 前向使用权重侧的 `W + alpha/r * B@A`，与合并权重走相同的投影计算。真实权重测试发现，拆成两条 GEMM 再相加的 FP32 误差会在 ViT 中明显放大，不能依靠小线性层测试证明全网合并一致。dropout 作为 rank 空间的修正项加入，评估时关闭。这个选择会增加临时矩阵和反向计算开销，优先保证一致性；产物记录 `canonical_weight_plus_delta_v1`。

`save_state = true` 在定期及最终更新边界生成 `state-stepNNNNNN/`。状态包含 optimizer、已消费样本数、CPU/CUDA/Python/NumPy RNG、展开配置与数据/基座/实现指纹，并有文件完整性清单。`save_state = false` 仍按保存间隔输出 `stepNNNNNN/` 权重，但这些目录不能 exact resume。

仅修改 TOML 注释不会阻止恢复；修改有效配置、引用的数据内容、基座或训练实现会拒绝 exact resume。CUDA 反向部分算子的非确定性仍需按数值容差验收，不承诺所有硬件间逐位一致。

旧版 `dlssnr_train_state_v1` 不再视为精确恢复点。旧 adapter 若缺少完整基座身份、rank/alpha 或 forward mode 元数据，也不会被静默接受。保留旧文件，重新生成受校验的产物。

## 本轮验证

2026-09-23，本机全库测试 159 项通过，无跳过，Ruff 检查通过。测试包含原始 310.8.0 权重在 RTX 4090 上的 FP32 全量/LoRA 单步更新、全部 ViT LoRA targets 的梯度和合并后的 raw-head 对照。CUDA smoke 使用 48×48 合成配对输入，不代表正式分辨率的性能、真实数据训练质量或 native parity。

开发环境未安装项目包时，可在仓库根目录设置 `PYTHONPATH` 后测试；正常运行仍应使用项目依赖环境。真实权重 CUDA 测试需本机源权重，默认查找仓库相邻的 `../OpenDLSS-NR/models/nr`，也可通过 `DLSSNR_SOURCE_DIR` 显式指定目录。缺少权重时，这些外部权重测试会跳过；不代表已通过真实权重验收。

```powershell
$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:DLSSNR_RUN_CUDA_SMOKE = "1"
python -m pytest -q --tb=short
```
