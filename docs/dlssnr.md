# DLSS-NR 310.8.0 训练适配

## English Summary

This is an experimental, standalone supervised-training path for DLSS-NR 310.8.0,
not a diffusion trainer or an NVIDIA DLL replacement. It provides canonical
weight conversion, FP32 full/LoRA training, still/closed-loop inference, adapter
merge, and checked optimizer-state resume. The two dataset examples enable the
project's shared resolution buckets with a 512 x 512 area budget and `bucket_no_upscale` enabled.
Set `enable_bucket = false` for the original strict fixed-resolution behavior.
Each input and its target, controls, motion and masks must share the original pixel grid.

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

Use `--dataset_config` for a **dataset-only TOML** with `[general]` and one
`[[datasets]]` entry. Model, optimizer, LoRA, loss, evaluation and output settings
are CLI arguments, using the project's standard names such as `--optimizer_type`,
`--optimizer_args`, `--learning_rate`, `--network_dim` and `--network_alpha`.
The old all-in-one `--config_file` training interface is no longer accepted.
Dataset-relative paths resolve against the TOML directory; CLI paths resolve
against the working directory. The generated `run_config.json` is an effective
configuration snapshot for provenance/resume, not another user configuration file.

这条路径使用 FP32 主权重和矩阵累加，用来做配对监督。中间激活、注意力和归约包含明确的 half/E4M3 fake quant 与替代梯度，不能再视作普通 FP32 SiLU/softmax 网络。它仍不是原版 DLL 的逐字节推理，输出也还在 proxy 颜色空间里，没有 DLL 后面的 Natural / Cinematic 调色。

## 配置分工

和项目其他架构一样，数据集使用 `--dataset_config`，训练超参数使用命令行。单帧全量和 LoRA 共用 [单帧数据集示例](../configs/dlssnr_dataset_single.toml)，时序训练使用 [时序数据集示例](../configs/dlssnr_dataset_temporal.toml)。

```toml
[general]
resolution = [512, 512]
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
train_manifest = "../data/train_single.jsonl"
# validation_manifest = "../data/validation_single.jsonl"
# sequence_manifest = "../data/validation_sequences.jsonl"
```

`resolution`、`batch_size`、`enable_bucket` 和 `bucket_no_upscale` 可在 `[[datasets]]` 中覆盖 `[general]`。当前仍只支持一个 manifest 数据集入口，但同一清单可以混合横图、竖图和不同原始尺寸。编码、逐帧 motion 和 mask 等信息仍放在 JSONL manifest 中。

TOML 中的 `[model]`、`[training]`、`[optimizer]`、`[lora]`、`[loss]`、`[output]` 等旧训练配置会明确报错，不再作为另一套超参数来源。自动生成的 `run_config.json` 保留完整展开值，仅供追溯和恢复校验。

## 分桶与配对变换

分桶调用项目公共 `BucketSelector`，同桶组批调用 `BucketBatchManager` 的公共样本选择接口，不读取 diffusion 的 latent/text-encoder 缓存。NR 使用独立架构标识 `nr`，分桶步长为 16，桶的每一轴至少 48 像素；模型内部 padded field 的几何和前向没有改变。

- `enable_bucket = true` 时，`resolution` 的宽高乘积表示桶面积预算，选桶依据原始输入的宽高比，不会把所有图强制变成正方形。比如预算为 `128 x 128` 时，`320 x 192` 的输入进入 `160 x 96` 桶，竖图进入对应的竖桶。
- `bucket_no_upscale = true` 沿用项目语义：面积不超过预算的图像以原尺寸向下对齐到 16 的倍数，不放大。若向下对齐后的某一轴不足 48，会明确拒绝，不会偷偷放大或填充。超过预算的图像仍按宽高比选候选桶并做覆盖裁剪；极端宽高比下可能放大短轴，这个开关并非对任意输入都保证不放大。
- 不写 `enable_bucket` 时仍默认关闭，保留原来的严格固定尺寸模式：所有配对张量必须已符合 `resolution`，尺寸不符就报错。示例 TOML 显式开启分桶和 no-upscale。
- 所有配对张量必须先在原始像素网格上对齐。同一 clip 的所有帧必须具有相同的原始输入尺寸，并共享同一个桶、等比覆盖缩放和中心裁剪位置；不做逐帧随机裁剪，也不独立缩放 target 或 controls 来掩盖错误配对。
- RGB、encoded controls 和连续 loss mask 使用同一 FP32 bilinear/antialias 缩放；二值 history/temporal mask 使用对齐像素中心的 nearest-exact，保持二值。motion 使用 bilinear，并将 x/y 位移分别乘以实际取整后的缩放宽高比。同位置裁剪当前帧和前一帧，裁剪偏移相互抵消；越出裁剪区域的历史仍由重投影的 inside mask 排除。
- 一个 microbatch 内只有同尺寸、同帧数的样本。不同桶可出现在同一次梯度累积中，损失继续按有效像素总数归一化。每个桶不足 batch size 的尾批保留，不丢弃、不重复填充；`consumed_samples` 记录实际消费数。

索引阶段只读取源图头部或 NPY 的 mmap shape 来选桶，不将像素常驻内存。训练前仍会逐项解码校验配对、数值和变换后剩余的监督区域。训练和验证使用同一变换规则，`run_config.json` 的 `bucket_plan` 记录实际桶、每桶样本数、每轮批数及确定性批次顺序指纹。NR 保留固定的桶/清单顺序和确定性中心裁剪，不新增随机 shuffle 或随机增强。

分桶不等于原生输出等价：图像增强通常不满足“先增强再缩放”等于“先缩放再增强”。要优先保留原生配对的像素尺度，使用 no-upscale，并准备面积预算内、16 对齐的源尺寸；缩放后的配对只能作为训练数据变换，不能作为 DLL 前向一致性的证据。

## 当前范围

根目录脚本保持 thin entry point。模型、数值、数据、训练损失和评估位于 `src/musubi_tuner/dlssnr/`，LoRA 位于 `networks/lora_dlssnr.py`，生命周期位于 `training/dlssnr_trainer.py`。全量和 LoRA 共用 `NRTrainModule`、Accelerate 包装及同一更新循环，不调用 diffusion trainer。

- 单进程 FP32，默认自动选择 CUDA，缺少 CUDA 时使用 CPU；`--device` 可设为 `auto`、`cpu` 或 `cuda`。显式请求不可用的 CUDA 会报错。
- 单帧及一个有限时序段，支持全量/LoRA 梯度累积、定期保存与恢复。累积按有效元素总数归一化，计步单位是 optimizer update。
- 数据配置和命令行参数均严格校验。数据清单路径以 TOML 所在目录为基准，命令行路径以当前工作目录为基准；最终配置和来源指纹写入 `run_config.json`。
- 数据集中的 `validation_manifest` 和 `sequence_manifest` 会实际执行，使用独立学生 history。默认比较初始基座，写入 `evaluation/stepNNNNNN.json`；评估不会改变训练 RNG 或 dropout 状态。
- manifest 按需解码，不将整个训练集常驻内存；推理和评估按帧回放。

原生参考前向、真实片段上的兼容性和质量验收尚未完成。布局字节 round-trip、浮点训练测试和原生兼容是不同结论。

2026-09-24 已完成第一轮 Steam 截图与本地 310.8.0 显卡适配版 DLL 的对照，结论是**当前 FP32 前向不一致**。4 张截图、19 组图片/参数组合覆盖 style、intensity、tone、structure、skin 和 auto mask；默认参数的输出与 DLL 之间 MAE 为 0.0294 至 0.0420。相同 packed 权重不代表前向计算正确，不建议据此开始正式训练。详见 [截图前向实验记录](dlssnr_forward_experiment_2026-09-24.md)。

同日后续已排除主要默认参数因素，并修复 cubic 激活及关键数值边界。默认参数下四图 MAE 降至 0.00262 至 0.00520，19 组中 9 组达到原有三项显示指标门槛（含一组零强度直通），**尚未完成全部前向验收**。全量测试为 172 passed，包含真实权重 CUDA 全量/LoRA 更新及合并对照。详见 [可训练前向对齐记录](dlssnr_forward_alignment_2026-09-24.md)。数值实现版本已更新，旧训练状态不能跨版本 exact resume。

## 正式运行与实验运行

正常训练必须提供 `--model_dir`、完整 canonical 来源文件、通过 round-trip 的转换报告，以及绑定当前权重和数值实现的 forward validation report。缺少这些证据时，入口会停止，而不会隐式随机初始化。

当前开发实验须显式传 `--development_smoke`。这允许尚未通过参考验收的 canonical 权重；只有这种模式允许省略 `--model_dir` 使用随机权重。输出会标记 `experimental_surrogate`，不应作为兼容模型发布。

验证报告默认读取 `model_dir/forward_validation_report.json`，也可通过 `--forward_validation_report` 指定。契约为 `dlssnr_forward_validation_v1`，包含 `profile`、`numerics_profile`、当前 `model_sha256`、`implementation_sha256`、固定的 `reference_identity`、`float_validated` 和 `checks`。检查项为 `raw_head`、`neural_preclamp`、`rendered_proxy`；全部必须来自实际参考验证。实现指纹由 `json_sha256(implementation_identity())` 计算。普通训练 evaluation 不能代替这份报告，也不要手工填入通过标记。

## 命令

把自行准备的 OpenDLSS-NR `models/nr` 目录转成 canonical 权重：

```bash
python dlssnr_convert_model.py --source_dir ../OpenDLSS-NR/models/nr --output_dir models/canonical_dlssnr --verify_roundtrip
```

单帧全量训练。以下多行命令使用 Bash 续行；PowerShell 可合并为一行，或将行末反斜线换成反引号：

```bash
python dlssnr_train.py \
  --dataset_config configs/dlssnr_dataset_single.toml \
  --model_dir models/canonical_dlssnr \
  --optimizer_type AdamW --learning_rate 1e-5 \
  --optimizer_args weight_decay=0.0 \
  --output_dir output/dlssnr_full_single --output_name dlssnr_310_8_0_full \
  --max_train_steps 1000 --save_every_n_steps 100 --save_state \
  --development_smoke
```

时序全量训练：

```bash
python dlssnr_train.py \
  --dataset_config configs/dlssnr_dataset_temporal.toml \
  --model_dir models/canonical_dlssnr \
  --training_mode temporal --sequence_length 4 --burn_in 2 --tbptt_length 2 \
  --loss_temporal 0.10 --optimizer_type AdamW --learning_rate 1e-5 \
  --output_dir output/dlssnr_full_temporal --output_name dlssnr_310_8_0_temporal \
  --max_train_steps 1000 --save_every_n_steps 100 --save_state \
  --development_smoke
```

ViT LoRA。rank、alpha 和优化器都直接从参数选择，使用同一份单帧数据集 TOML：

```bash
python dlssnr_train_network.py \
  --dataset_config configs/dlssnr_dataset_single.toml \
  --model_dir models/canonical_dlssnr \
  --network_dim 16 --network_alpha 16 --network_dropout 0.0 \
  --optimizer_type AdamW --learning_rate 1e-4 \
  --optimizer_args weight_decay=0.0 betas=0.9,0.999 \
  --output_dir output/dlssnr_lora --output_name dlssnr_310_8_0_lora_vit \
  --max_train_steps 1000 --save_every_n_steps 100 --save_state \
  --development_smoke
```

常用参数：

| 参数 | 用途 |
| --- | --- |
| `--optimizer_type` / `--optimizer_args` | 复用项目公共优化器构造逻辑，例如 `AdamW`、`SGD`、`AdamW8bit` 或完整类路径。额外依赖须自行安装。 |
| `--learning_rate` | 全量默认 `1e-5`，LoRA 默认 `1e-4`；AdamW 默认 weight decay 为 `0.0`。 |
| `--network_dim` / `--network_alpha` | ViT LoRA rank 默认 `16`；未给 alpha 时跟随 rank。 |
| `--network_dropout` / `--network_args` | dropout 默认 `0`。多尺度 LoRA 用 `--network_args profile=multiscale`，可通过 `rank_by_width` / `alpha_by_width` 字典细化，不能同时给单一 dim/alpha。 |
| `--max_train_steps` / `--gradient_accumulation_steps` / `--seed` | 更新次数、梯度累积和随机种子。 |
| `--max_grad_norm` | 在累积后的更新边界裁剪梯度，默认 `0`，即不裁剪。 |
| `--loss_pre` / `--loss_out` / `--loss_edge` / `--loss_temporal` | 损失权重，默认分别为 `1`、`1`、`0.05`、`0`。 |
| `--prior_lr_multiplier` / `--scale_lr_multiplier` / `--temporal_blend_lr_multiplier` | 全量训练参数组倍率，默认 `0.1`；LoRA 入口不接受这些参数。 |
| `--sample_every_n_steps` / `--min_sequence_frames` | 验证间隔和时序评估的最小帧数。非零验证间隔要求数据 TOML 配置验证清单。 |
| `--no-compare_baseline` | 关闭初始基座对照，默认开启。 |
| `--save_state` / `--save_every_n_steps` / `--resume` | 保存训练状态、周期和恢复点；未给 `--save_state` 时仅保存权重。 |

例如，将优化器选项换成 `--optimizer_type SGD --optimizer_args momentum=0.9 weight_decay=0.0`，会实际构造 SGD，而不是只改变配置标签。Adafactor 必须显式给 `--optimizer_args relative_step=False warmup_init=False`。当前仍只支持 constant LR；schedule-free 的权重切换和需要 closure 的优化器不支持，会明确报错。

从 optimizer 更新边界继续时，在**原来的完整训练命令**后追加恢复参数，保留原有有效参数和数据配置。例如：

```text
--resume output/dlssnr_full_single/dlssnr_310_8_0_full/state-step000100
```

LoRA 的恢复目录对应 `output/dlssnr_lora/dlssnr_310_8_0_lora_vit/state-stepNNNNNN`。只加载权重、不带 optimizer 状态属于重新开跑，不是 resume。

静止图和闭环片段。推理清单可以不写 target。片段里非 reset 帧必须有 motion 和 history mask：

```bash
python dlssnr_generate_image.py --model_dir models/canonical_dlssnr --sample_manifest data/inference_single.jsonl --bucket_width 512 --bucket_height 512 --numerics_profile train_surrogate --output_dir output/preview
python dlssnr_generate_video.py --model_dir models/canonical_dlssnr --sequence_manifest data/inference_sequence.jsonl --bucket_width 512 --bucket_height 512 --numerics_profile train_surrogate --output_dir output/video
python dlssnr_merge_lora.py --base_model_dir models/canonical_dlssnr --adapter output/dlssnr_lora/dlssnr_310_8_0_lora_vit/final/adapter.safetensors --output_dir output/dlssnr_merged
```

## 现在明确不做的事

混合精度、gradient checkpointing、多卡、FP8 基座、以及 SageAttention / FlashAttention / xFormers 都未开放。未知 CLI 参数会报错；Accelerate 使用非 `no` 的 mixed precision 或多进程启动时也会停止。

历史重投影用的是双线性，不是原版五点 Catmull-Rom。历史按 FP32 保存，不是向零截断到 float16。

像素缓存和多 worker 预取尚未实现，数据 TOML 不接受缓存占位字段。FP32 入口会暂时关闭 TF32，结束后恢复原来的全局设置。

## 数据约定

每条 JSONL 记录必须包含唯一且可用于文件名的 `sample_id`、`sequence_id`、`schema = dlssnr_pairs_v1`、`source_encoding = srgb_proxy`、`controls_encoding = dlssnr_lanes_10_14_v1`。训练及配对评估还需 `target_encoding = srgb_proxy` 和 target。首帧必须 reset，`frame_index` 非负且严格递增；训练与验证不能共享 sequence ID。

`sample_id` 在所有平台上按不区分大小写的规则检查重复，例如 `Frame` 和 `frame` 不能出现在同一清单中。名称不能包含路径分隔符、Windows 非法字符、控制字符、末尾句点或空格，也不能是 `NUL`、`CON` 等 Windows 保留名称。清单索引阶段即报错，不会写出部分推理结果后才发现冲突；原始 ID 不会被自动改写。

- RGB 支持 8-bit 图像，或 `[3,H,W]` 的数值 NPY，proxy 值域为 `[0,1]`。不将高位深图片静默压成 8-bit。
- controls 为有限值 `[5,H,W]` NPY。motion 的布局由记录级 `motion_layout = chw|hwc` 明确指定，单位是 current-to-previous 像素位移。
- mask 支持单通道 PNG 或 `[1,H,W]` NPY。history/temporal mask 必须为二值；可选 `loss_mask_path` 支持 `[0,1]` 权重。
- temporal loss 非零时，非 reset 帧必须提供 temporal mask，且 `tbptt_length >= 2`。损失只计算可反传段内部的相邻帧，不跨越 burn-in 边界。
- `--loss_temporal 0` 允许不提供 temporal mask，但非 reset 帧仍必须有 motion 和 history mask。

## 保存与恢复

每次训练使用 `output_dir/output_name/` 作为独立运行目录，`final/`、周期权重、训练状态、日志和评估报告均位于该目录下。`--output_name` 必须满足上述可跨平台使用的名称规则，不能包含路径。新训练拒绝使用已存在的同名运行目录；继续训练须显式传入 `--resume`，重新训练须选择其他名称或输出目录。

旧产物不会自动搬迁。CLI/dataset 分离后，展开配置使用 schema version 2；分桶适配也改变了数据变换、组批和训练实现指纹。旧版大 TOML 或分桶适配前的状态不能直接跨版本 exact resume。保留旧权重，另建运行目录重新开跑，不要手改检查点身份标记。

全量 `final/` 保存 FP32 权重、canonical 配置、来源 manifest、原始 opaque records、数值与预处理信息及训练元数据。LoRA 保存 adapter、rank/alpha/targets 和完整基座身份，不再只哈希 targets。合并时同样保留 canonical 来源与 opaque 数据。

LoRA 前向使用权重侧的 `W + alpha/r * B@A`，与合并权重走相同的投影计算。真实权重测试发现，拆成两条 GEMM 再相加的 FP32 误差会在 ViT 中明显放大，不能依靠小线性层测试证明全网合并一致。dropout 作为 rank 空间的修正项加入，评估时关闭。这个选择会增加临时矩阵和反向计算开销，优先保证一致性；产物记录 `canonical_weight_plus_delta_v1`。

`--save_state` 在定期及最终更新边界生成 `state-stepNNNNNN/`。状态包含 optimizer、实际已消费样本数、CPU/CUDA/Python/NumPy RNG、展开配置与数据/组批计划/基座/实现指纹，并有文件完整性清单。恢复位置由 optimizer 更新次数和梯度累积次数定位到同桶批次，同时核验实际消费数，支持桶尾小批次和跨轮恢复。不传此参数时仍按保存间隔输出 `stepNNNNNN/` 权重，但这些目录不能 exact resume。

仅修改数据 TOML 注释或 CLI 参数排列顺序不会阻止恢复；修改学习率、优化器、rank、分桶设置等有效参数、引用的数据内容、基座或训练实现会拒绝 exact resume。CUDA 反向部分算子的非确定性仍需按数值容差验收，不承诺所有硬件间逐位一致。

旧版 `dlssnr_train_state_v1` 不再视为精确恢复点。旧 adapter 若缺少完整基座身份、rank/alpha 或 forward mode 元数据，也不会被静默接受。保留旧文件，重新生成受校验的产物。

## 历史验证与回归

2026-09-23，本机全库测试 159 项通过，无跳过，Ruff 检查通过。测试包含原始 310.8.0 权重在 RTX 4090 上的 FP32 全量/LoRA 单步更新、全部 ViT LoRA targets 的梯度和合并后的 raw-head 对照。CUDA smoke 使用 48×48 合成配对输入，不代表正式分辨率的性能、真实数据训练质量或 native parity。

开发环境未安装项目包时，可在仓库根目录设置 `PYTHONPATH` 后测试；正常运行仍应使用项目依赖环境。真实权重 CUDA 测试需本机源权重，默认查找仓库相邻的 `../OpenDLSS-NR/models/nr`，也可通过 `DLSSNR_SOURCE_DIR` 显式指定目录。缺少权重时，这些外部权重测试会跳过；不代表已通过真实权重验收。

```powershell
$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:DLSSNR_RUN_CUDA_SMOKE = "1"
python -m pytest -q --tb=short
```
