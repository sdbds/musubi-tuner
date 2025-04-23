> 📝 Click on the language section to expand / 言語をクリックして展開

# Advanced configuration / 高度な設定

## Table of contents / 目次

- [How to specify `network_args`](#how-to-specify-network_args--network_argsの指定方法)
- [LoRA+](#lora)
- [Select the target modules of LoRA](#select-the-target-modules-of-lora--loraの対象モジュールを選択する)
- [Save and view logs in TensorBoard format](#save-and-view-logs-in-tensorboard-format--tensorboard形式のログの保存と参照)
- [Save and view logs in wandb](#save-and-view-logs-in-wandb--wandbでログの保存と参照)
- [FP8 weight optimization for models](#fp8-weight-optimization-for-models--モデルの重みのfp8への最適化)
- [PyTorch Dynamo optimization for model training](#pytorch-dynamo-optimization-for-model-training--モデルの学習におけるpytorch-dynamoの最適化)

## How to specify `network_args` / `network_args`の指定方法

The `--network_args` option is an option for specifying detailed arguments to LoRA. Specify the arguments in the form of `key=value` in `--network_args`.

<details>
<summary>日本語</summary>
`--network_args`オプションは、LoRAへの詳細な引数を指定するためのオプションです。`--network_args`には、`key=value`の形式で引数を指定します。
</details>

### Example / 記述例

If you specify it on the command line, write as follows. / コマンドラインで指定する場合は以下のように記述します。

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 hv_train_network.py --dit ... 
    --network_module networks.lora --network_dim 32 
    --network_args "key1=value1" "key2=value2" ...
```

If you specify it in the configuration file, write as follows. / 設定ファイルで指定する場合は以下のように記述します。

```toml
network_args = ["key1=value1", "key2=value2", ...]
```

If you specify `"verbose=True"`, detailed information of LoRA will be displayed. / `"verbose=True"`を指定するとLoRAの詳細な情報が表示されます。

```bash
--network_args "verbose=True" "key1=value1" "key2=value2" ...
```

## LoRA+

LoRA+ is a method to improve the training speed by increasing the learning rate of the UP side (LoRA-B) of LoRA. Specify the multiplier for the learning rate. The original paper recommends 16, but adjust as needed. It seems to be good to start from around 4. For details, please refer to the [related PR of sd-scripts](https://github.com/kohya-ss/sd-scripts/pull/1233).

Specify `loraplus_lr_ratio` with `--network_args`.

<details>
<summary>日本語</summary>

LoRA+は、LoRAのUP側（LoRA-B）の学習率を上げることで学習速度を向上させる手法です。学習率に対する倍率を指定します。元論文では16を推奨していますが、必要に応じて調整してください。4程度から始めるとよいようです。詳細は[sd-scriptsの関連PR]https://github.com/kohya-ss/sd-scripts/pull/1233)を参照してください。

`--network_args`で`loraplus_lr_ratio`を指定します。
</details>

### Example / 記述例

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 hv_train_network.py --dit ... 
    --network_module networks.lora --network_dim 32 --network_args "loraplus_lr_ratio=4" ...
```

## Select the target modules of LoRA / LoRAの対象モジュールを選択する

*This feature is highly experimental and the specification may change. / この機能は特に実験的なもので、仕様は変更される可能性があります。*

By specifying `exclude_patterns` and `include_patterns` with `--network_args`, you can select the target modules of LoRA.

`exclude_patterns` excludes modules that match the specified pattern. `include_patterns` targets only modules that match the specified pattern.

Specify the values as a list. For example, `"exclude_patterns=[r'.*single_blocks.*', r'.*double_blocks\.[0-9]\..*']"`.

The pattern is a regular expression for the module name. The module name is in the form of `double_blocks.0.img_mod.linear` or `single_blocks.39.modulation.linear`. The regular expression is not a partial match but a complete match.

The patterns are applied in the order of `exclude_patterns`→`include_patterns`. By default, the Linear layers of `img_mod`, `txt_mod`, and `modulation` of double blocks and single blocks are excluded.

(`.*(img_mod|txt_mod|modulation).*` is specified.)

<details>
<summary>日本語</summary>

`--network_args`で`exclude_patterns`と`include_patterns`を指定することで、LoRAの対象モジュールを選択することができます。

`exclude_patterns`は、指定したパターンに一致するモジュールを除外します。`include_patterns`は、指定したパターンに一致するモジュールのみを対象とします。

値は、リストで指定します。`"exclude_patterns=[r'.*single_blocks.*', r'.*double_blocks\.[0-9]\..*']"`のようになります。

パターンは、モジュール名に対する正規表現です。モジュール名は、たとえば`double_blocks.0.img_mod.linear`や`single_blocks.39.modulation.linear`のような形式です。正規表現は部分一致ではなく完全一致です。

パターンは、`exclude_patterns`→`include_patterns`の順で適用されます。デフォルトは、double blocksとsingle blocksのLinear層のうち、`img_mod`、`txt_mod`、`modulation`が除外されています。

（`.*(img_mod|txt_mod|modulation).*`が指定されています。）
</details>

### Example / 記述例

Only the modules of double blocks / double blocksのモジュールのみを対象とする場合:

```bash
--network_args "exclude_patterns=[r'.*single_blocks.*']"
```

Only the modules of single blocks from the 10th / single blocksの10番目以降のLinearモジュールのみを対象とする場合:

```bash
--network_args "exclude_patterns=[r'.*']" "include_patterns=[r'.*single_blocks\.\d{2}\.linear.*']"
```

## Save and view logs in TensorBoard format / TensorBoard形式のログの保存と参照

Specify the folder to save the logs with the `--logging_dir` option. Logs in TensorBoard format will be saved.

For example, if you specify `--logging_dir=logs`, a `logs` folder will be created in the working folder, and logs will be saved in the date folder inside it.

Also, if you specify the `--log_prefix` option, the specified string will be added before the date. For example, use `--logging_dir=logs --log_prefix=lora_setting1_` for identification.

To view logs in TensorBoard, open another command prompt and activate the virtual environment. Then enter the following in the working folder.

```powershell
tensorboard --logdir=logs
```

(tensorboard installation is required.)

Then open a browser and access http://localhost:6006/ to display it.

<details>
<summary>日本語</summary>
`--logging_dir`オプションにログ保存先フォルダを指定してください。TensorBoard形式のログが保存されます。

たとえば`--logging_dir=logs`と指定すると、作業フォルダにlogsフォルダが作成され、その中の日時フォルダにログが保存されます。

また`--log_prefix`オプションを指定すると、日時の前に指定した文字列が追加されます。`--logging_dir=logs --log_prefix=lora_setting1_`などとして識別用にお使いください。

TensorBoardでログを確認するには、別のコマンドプロンプトを開き、仮想環境を有効にしてから、作業フォルダで以下のように入力します。

```powershell
tensorboard --logdir=logs
```

（tensorboardのインストールが必要です。）

その後ブラウザを開き、http://localhost:6006/ へアクセスすると表示されます。
</details>

## Save and view logs in wandb / wandbでログの保存と参照

`--log_with wandb` option is available to save logs in wandb format. `tensorboard` or `all` is also available. The default is `tensorboard`.

Specify the project name with `--log_tracker_name` when using wandb.

<details>
<summary>日本語</summary>
`--log_with wandb`オプションを指定するとwandb形式でログを保存することができます。`tensorboard`や`all`も指定可能です。デフォルトは`tensorboard`です。

wandbを使用する場合は、`--log_tracker_name`でプロジェクト名を指定してください。
</details>

## FP8 weight optimization for models / モデルの重みのFP8への最適化

The `--fp8_scaled` option is available to quantize the weights of the model to FP8 (E4M3) format with appropriate scaling. This reduces the VRAM usage while maintaining precision. Important weights are kept in FP16/BF16/FP32 format.

The model weights must be in fp16 or bf16. Weights that have been pre-converted to float8_e4m3 cannot be used.

Wan2.1 inference and training are supported.

Specify the `--fp8_scaled` option in addition to the `--fp8` option during inference.

Specify the `--fp8_scaled` option in addition to the `--fp8_base` option during training.

Acknowledgments: This feature is based on the [implementation](https://github.com/Tencent/HunyuanVideo/blob/7df4a45c7e424a3f6cd7d653a7ff1f60cddc1eb1/hyvideo/modules/fp8_optimization.py) of [HunyuanVideo](https://github.com/Tencent/HunyuanVideo). The selection of high-precision modules is based on the [implementation](https://github.com/tdrussell/diffusion-pipe/blob/407c04fdae1c9ab5e67b54d33bef62c3e0a8dbc7/models/wan.py) of [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe). I would like to thank these repositories.

<details>
<summary>日本語</summary>
重みを単純にFP8へcastするのではなく、適切なスケーリングでFP8形式に量子化することで、精度を維持しつつVRAM使用量を削減します。また、重要な重みはFP16/BF16/FP32形式で保持します。

モデルの重みは、fp16またはbf16が必要です。あらかじめfloat8_e4m3に変換された重みは使用できません。

Wan2.1の推論、学習のみ対応しています。

推論時は`--fp8`オプションに加えて `--fp8_scaled`オプションを指定してください。

学習時は`--fp8_base`オプションに加えて `--fp8_scaled`オプションを指定してください。

謝辞：この機能は、[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)の[実装](https://github.com/Tencent/HunyuanVideo/blob/7df4a45c7e424a3f6cd7d653a7ff1f60cddc1eb1/hyvideo/modules/fp8_optimization.py)を参考にしました。また、高精度モジュールの選択においては[diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)の[実装](https://github.com/tdrussell/diffusion-pipe/blob/407c04fdae1c9ab5e67b54d33bef62c3e0a8dbc7/models/wan.py)を参考にしました。これらのリポジトリに感謝します。

</details>

### Key features and implementation details / 主な特徴と実装の詳細

- Implements FP8 (E4M3) weight quantization for Linear layers
- Reduces VRAM requirements by using 8-bit weights for storage (slightly increased compared to existing `--fp8` `--fp8_base` options)
- Quantizes weights to FP8 format with appropriate scaling instead of simple cast to FP8
- Maintains computational precision by dequantizing to original precision (FP16/BF16/FP32) during forward pass
- Preserves important weights in FP16/BF16/FP32 format

The implementation:

1. Quantizes weights to FP8 format with appropriate scaling
2. Replaces weights by FP8 quantized weights and stores scale factors in model state dict
3. Applies monkey patching to Linear layers for transparent dequantization during computation

<details>
<summary>日本語</summary>

- Linear層のFP8（E4M3）重み量子化を実装
- 8ビットの重みを使用することでVRAM使用量を削減（既存の`--fp8` `--fp8_base` オプションに比べて微増）
- 単純なFP8へのcastではなく、適切な値でスケールして重みをFP8形式に量子化
- forward時に元の精度（FP16/BF16/FP32）に逆量子化して計算精度を維持
- 精度が重要な重みはFP16/BF16/FP32のまま保持

実装:

1. 精度を維持できる適切な倍率で重みをFP8形式に量子化
2. 重みをFP8量子化重みに置き換え、倍率をモデルのstate dictに保存
3. Linear層にmonkey patchingすることでモデルを変更せずに逆量子化
 </details>

 ## PyTorch Dynamo optimization for model training / モデルの学習におけるPyTorch Dynamoの最適化

The PyTorch Dynamo options are now available to optimize the training process. PyTorch Dynamo is a Python-level JIT compiler designed to make unmodified PyTorch programs faster by using TorchInductor, a deep learning compiler. This integration allows for potential speedups in training while maintaining model accuracy.

[PR #215](https://github.com/kohya-ss/musubi-tuner/pull/215) added this feature.

Specify the `--dynamo_backend` option to enable Dynamo optimization with one of the available backends from the `DynamoBackend` enum.

Additional options allow for fine-tuning the Dynamo behavior:
- `--dynamo_mode`: Controls the optimization strategy
- `--dynamo_fullgraph`: Enables fullgraph mode for potentially better optimization
- `--dynamo_dynamic`: Enables dynamic shape handling

The `--dynamo_dynamic` option has been reported to have many problems based on the validation in PR #215.

### Available options:

```
--dynamo_backend {NO, INDUCTOR, NVFUSER, CUDAGRAPHS, CUDAGRAPHS_FALLBACK, etc.}
    Specifies the Dynamo backend to use (default is NO, which disables Dynamo)

--dynamo_mode {default, reduce-overhead, max-autotune}
    Specifies the optimization mode (default is 'default')
    - 'default': Standard optimization
    - 'reduce-overhead': Focuses on reducing compilation overhead
    - 'max-autotune': Performs extensive autotuning for potentially better performance

--dynamo_fullgraph
    Flag to enable fullgraph mode, which attempts to capture and optimize the entire model graph

--dynamo_dynamic
    Flag to enable dynamic shape handling for models with variable input shapes
```

### Usage example:

```bash
python train_video_model.py --dynamo_backend INDUCTOR --dynamo_mode default
```

For more aggressive optimization:
```bash
python train_video_model.py --dynamo_backend INDUCTOR --dynamo_mode max-autotune --dynamo_fullgraph
```

Note: The best combination of options may depend on your specific model and hardware. Experimentation may be necessary to find the optimal configuration.

<details>
<summary>日本語</summary>
PyTorch Dynamoオプションが学習プロセスを最適化するために追加されました。PyTorch Dynamoは、TorchInductor（ディープラーニングコンパイラ）を使用して、変更を加えることなくPyTorchプログラムを高速化するためのPythonレベルのJITコンパイラです。この統合により、モデルの精度を維持しながら学習の高速化が期待できます。

[PR #215](https://github.com/kohya-ss/musubi-tuner/pull/215) で追加されました。

`--dynamo_backend`オプションを指定して、`DynamoBackend`列挙型から利用可能なバックエンドの一つを選択することで、Dynamo最適化を有効にします。

追加のオプションにより、Dynamoの動作を微調整できます：
- `--dynamo_mode`：最適化戦略を制御します
- `--dynamo_fullgraph`：より良い最適化の可能性のためにフルグラフモードを有効にします
- `--dynamo_dynamic`：動的形状処理を有効にします

PR #215での検証によると、`--dynamo_dynamic`には問題が多いことが報告されています。

__利用可能なオプション：__

```
--dynamo_backend {NO, INDUCTOR, NVFUSER, CUDAGRAPHS, CUDAGRAPHS_FALLBACK, など}
    使用するDynamoバックエンドを指定します（デフォルトはNOで、Dynamoを無効にします）

--dynamo_mode {default, reduce-overhead, max-autotune}
    最適化モードを指定します（デフォルトは 'default'）
    - 'default'：標準的な最適化
    - 'reduce-overhead'：コンパイルのオーバーヘッド削減に焦点を当てる
    - 'max-autotune'：より良いパフォーマンスのために広範な自動調整を実行

--dynamo_fullgraph
    フルグラフモードを有効にするフラグ。モデルグラフ全体をキャプチャして最適化しようとします

--dynamo_dynamic
    可変入力形状を持つモデルのための動的形状処理を有効にするフラグ
```

__使用例：__

```bash
python train_video_model.py --dynamo_backend INDUCTOR --dynamo_mode default
```

より積極的な最適化の場合：
```bash
python train_video_model.py --dynamo_backend INDUCTOR --dynamo_mode max-autotune --dynamo_fullgraph
```

注意：最適なオプションの組み合わせは、特定のモデルとハードウェアに依存する場合があります。最適な構成を見つけるために実験が必要かもしれません。
</details>
