> 📝 Click on the language section to expand / 言語をクリックして展開

# Sampling during training / 学習中のサンプル画像生成

By preparing a prompt file, you can generate sample images during training.

Please be aware that it consumes a considerable amount of VRAM, so be careful when generating sample images for videos with a large number of frames. Also, since it takes time to generate, adjust the frequency of sample image generation as needed.

<details>
<summary>日本語</summary>

プロンプトファイルを用意することで、学習中にサンプル画像を生成することができます。

VRAMをそれなりに消費しますので、特にフレーム数が多い動画を生成する場合は注意してください。また生成には時間がかかりますので、サンプル画像生成の頻度は適宜調整してください。
</details>

## How to use / 使い方

### Command line options for training with sampling / サンプル画像生成に関連する学習時のコマンドラインオプション

Example of command line options for training with sampling / 記述例:  

```bash
--vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt 
--vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128
--text_encoder1 path/to/ckpts/text_encoder 
--text_encoder2 path/to/ckpts/text_encoder_2 
--sample_prompts /path/to/prompt_file.txt 
--sample_every_n_epochs 1 --sample_every_n_steps 1000 -- sample_at_first
```

`--vae`, `--vae_chunk_size`, `--vae_spatial_tile_sample_min_size`, `--text_encoder1`, `--text_encoder2` are the same as when generating images, so please refer to [here](/README.md#inference) for details. `--fp8_llm` can also be specified.

`--sample_prompts` specifies the path to the prompt file used for sample image generation. Details are described below.

`--sample_every_n_epochs` specifies how often to generate sample images in epochs, and `--sample_every_n_steps` specifies how often to generate sample images in steps.

`--sample_at_first` is specified when generating sample images at the beginning of training.

Sample images and videos are saved in the `sample` directory in the directory specified by `--output_dir`. They are saved as `.png` for still images and `.mp4` for videos.

<details>
<summary>日本語</summary>

`--vae`、`--vae_chunk_size`、`--vae_spatial_tile_sample_min_size`、`--text_encoder1`、`--text_encoder2`は、画像生成時と同様ですので、詳細は[こちら](/README.ja.md#推論)を参照してください。`--fp8_llm`も指定可能です。

`--sample_prompts`は、サンプル画像生成に使用するプロンプトファイルのパスを指定します。詳細は後述します。

`--sample_every_n_epochs`は、何エポックごとにサンプル画像を生成するかを、`--sample_every_n_steps`は、何ステップごとにサンプル画像を生成するかを指定します。

`--sample_at_first`は、学習開始時にサンプル画像を生成する場合に指定します。

サンプル画像、動画は、`--output_dir`で指定したディレクトリ内の、`sample`ディレクトリに保存されます。静止画の場合は`.png`、動画の場合は`.mp4`で保存されます。
</details>

### Prompt file / プロンプトファイル

The prompt file is a text file that contains the prompts for generating sample images. The example is as follows. / プロンプトファイルは、サンプル画像生成のためのプロンプトを記述したテキストファイルです。例は以下の通りです。

```
# prompt 1: for generating a cat video
A cat walks on the grass, realistic style. --w 640 --h 480 --f 25 --d 1 --s 20

# prompt 2: for generating a dog image
A dog runs on the beach, realistic style. --w 960 --h 544 --f 1 --d 2 --s 20
```

A line starting with `#` is a comment.

* `--w` specifies the width of the generated image or video. The default is 256.
* `--h` specifies the height. The default is 256.
* `--f` specifies the number of frames. The default is 1, which generates a still image.
* `--d` specifies the seed. The default is random.
* `--s` specifies the number of steps in generation. The default is 20.
* `--g` specifies the guidance scale. The default is 6.0, which is the default value during inference of HunyuanVideo.
* `--fs` specifies the discrete flow shift. The default is 14.5, which corresponds to the number of steps 20. In the HunyuanVideo paper, 7.0 is recommended for 50 steps, and 17.0 is recommended for less than 20 steps (e.g. 10).

<details>
<summary>日本語</summary>

`#` で始まる行はコメントです。

* `--w` 生成画像、動画の幅を指定します。省略時は256です。
* `--h` 高さを指定します。省略時は256です。
* `--f` フレーム数を指定します。省略時は1で、静止画を生成します。
* `--d` シードを指定します。省略時はランダムです。
* `--s` 生成におけるステップ数を指定します。省略時は20です。
* `--g` guidance scaleを指定します。省略時は6.0で、HunyuanVideoの推論時のデフォルト値です。
* `--fs` discrete flow shiftを指定します。省略時は14.5で、ステップ数20の場合に対応した値です。HunyuanVideoの論文では、ステップ数50の場合は7.0、ステップ数20未満（10など）で17.0が推奨されています。
</details>