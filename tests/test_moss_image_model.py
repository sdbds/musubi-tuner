import pytest
import torch
from torch import nn
from torch.nn import functional as F

from musubi_tuner.moss_image import (
    vae as flux2_vae,
    model as moss_image_models,
    sampling as moss_image_train_utils,
    utils as moss_image_utils,
)


# region reference implementation (per-sample loop, copied from the official pipeline_moss_image.py, Apache-2.0)


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj["q"].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](_attention(q, k, self.proj["v"](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        exponent = torch.arange(self.num_bands, dtype=torch.float32) / self.num_bands
        self.register_buffer("bands", 1 / (self.base**exponent))

    def forward(self, height, width, text_length, device):
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class ReferenceMossImage(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, num_layers, text_dim, freq_dim, latent_channels, rope_base):
        super().__init__()
        self.freq_dim = freq_dim
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def forward(self, hidden_states, timestep, encoder_hidden_states, mask, skip_layers=()):
        skip_layers = frozenset(skip_layers)
        batch, _, height, width = hidden_states.shape
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1).expand(batch)
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding).float()
        mod = self.shared_block_modulation(time).view(batch, 6, -1)
        head = self.video_head_modulation(time).view(batch, 2, -1)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index : index + 1, mask[index].bool()]
            video = hidden_states[index : index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.contiguous())
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width :])
            x = torch.cat((video, text), 1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    x = layer(x, rope, mod[index : index + 1])
            x = self.final_norm_video(x[:, : height * width])
            x = x * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
            x = self.final_linear_video(x)
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        return torch.cat(outputs)


# endregion


SMALL_CONFIG = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=24, freq_dim=16, latent_channels=8, rope_base=10000.0)


def _make_models(split_attn: bool):
    torch.manual_seed(0)
    reference = ReferenceMossImage(**SMALL_CONFIG).eval()
    model = moss_image_models.MossImage(**SMALL_CONFIG, attn_mode="torch", split_attn=split_attn).eval()
    missing, unexpected = model.load_state_dict(reference.state_dict(), strict=True), None
    return reference, model


def _make_inputs():
    torch.manual_seed(1)
    batch, height, width = 2, 3, 5
    lens = [4, 7]
    x = torch.randn(batch, SMALL_CONFIG["latent_channels"], height, width)
    context = torch.randn(batch, max(lens), SMALL_CONFIG["text_dim"])
    mask = torch.zeros(batch, max(lens), dtype=torch.int32)
    for i, l in enumerate(lens):
        mask[i, :l] = 1
    context = context * mask[..., None]  # padded positions are zero like the official pipeline
    timesteps = torch.tensor([750.0, 120.0])
    return x, timesteps, context, mask


@pytest.mark.parametrize("split_attn", [False, True])
def test_batched_forward_matches_reference_loop(split_attn):
    reference, model = _make_models(split_attn)
    x, timesteps, context, mask = _make_inputs()
    with torch.no_grad():
        expected = reference(x, timesteps, context, mask)
        actual = model(x, timesteps, context, mask)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_batched_forward_matches_reference_with_skip_layers():
    reference, model = _make_models(False)
    x, timesteps, context, mask = _make_inputs()
    with torch.no_grad():
        expected = reference(x, timesteps, context, mask, skip_layers=(1,))
        actual = model(x, timesteps, context, mask, skip_layers=(1,))
        without_skip = model(x, timesteps, context, mask)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(actual, without_skip)


def test_forward_without_mask_is_full_length():
    reference, model = _make_models(False)
    x, timesteps, context, mask = _make_inputs()
    full_mask = torch.ones_like(mask)
    with torch.no_grad():
        expected = reference(x, timesteps, context, full_mask)
        actual = model(x, timesteps, context, None)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_gradient_checkpointing_matches_plain_forward():
    _, model = _make_models(False)
    x, timesteps, context, mask = _make_inputs()
    with torch.no_grad():
        expected = model(x, timesteps, context, mask)
    model.train()
    model.enable_gradient_checkpointing()
    x_grad = x.clone().requires_grad_(True)
    out = model(x_grad, timesteps, context, mask)
    out.sum().backward()
    torch.testing.assert_close(out.detach(), expected, atol=1e-5, rtol=1e-5)
    assert x_grad.grad is not None and torch.isfinite(x_grad.grad).all()


def test_split_fused_qkv_hook():
    dim = 32
    fused = torch.randn(3 * dim, dim)
    keys, _ = moss_image_utils.split_fused_qkv("layers.3.proj.qkv.weight", None, dim)
    assert keys == ["layers.3.proj.q.weight", "layers.3.proj.k.weight", "layers.3.proj.v.weight"]
    keys, tensors = moss_image_utils.split_fused_qkv("model.diffusion_model.layers.3.proj.qkv.weight", fused, dim)
    assert keys == ["layers.3.proj.q.weight", "layers.3.proj.k.weight", "layers.3.proj.v.weight"]
    assert torch.equal(torch.cat(tensors, 0), fused)
    assert moss_image_utils.split_fused_qkv("layers.3.proj.q.weight", fused, dim) == (None, None)
    assert moss_image_utils.split_fused_qkv("text_embedder.blocks.0.qkv.weight", fused, dim) == (None, None)


def test_sampling_sigmas_match_official_scheduler():
    # values computed with the official MOSSImageScheduler (shift 5.0, training_shift 1.0, 1000 steps)
    sigmas = moss_image_train_utils.get_sampling_sigmas(50, 5.0)
    assert sigmas[0] == 1.0 and sigmas[-1] == 0.0
    assert torch.all(sigmas[:-1] > sigmas[1:])  # strictly descending (duplicates removed)
    training = torch.linspace(1, 1 / 1000, 1000)
    # every sigma (except the trailing 0) lies on the training grid
    for s in sigmas[:-1]:
        assert torch.isclose(training, s).any()
    # the number of steps is reduced by snapping near sigma 0 where the shifted grid is denser than the training grid
    assert 40 <= len(sigmas) - 1 <= 50


def test_pad_text_embeds():
    a = torch.randn(3, 4)
    b = torch.randn(5, 4)
    embeds, mask = moss_image_utils.pad_text_embeds([a, b])
    assert embeds.shape == (2, 5, 4) and mask.shape == (2, 5)
    assert torch.equal(embeds[0, :3], a) and torch.equal(embeds[0, 3:], torch.zeros(2, 4))
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


def test_latent_patchify_roundtrip():
    z = torch.randn(2, 32, 8, 6)
    p = flux2_vae.patchify_latents(z)
    assert p.shape == (2, 128, 4, 3)
    assert torch.equal(flux2_vae.unpatchify_latents(p), z)
    # channel order matches the official pipeline: z.view(B, C, H/2, 2, W/2, 2).permute(0, 1, 3, 5, 2, 4)
    expected = z.view(2, 32, 4, 2, 3, 2).permute(0, 1, 3, 5, 2, 4).reshape(2, 128, 4, 3)
    assert torch.equal(p, expected)
    mean, var = torch.randn(128), torch.rand(128) + 0.5
    n = flux2_vae.normalize_latents(p, mean, var)
    torch.testing.assert_close(flux2_vae.denormalize_latents(n, mean, var), p, atol=1e-5, rtol=1e-5)


def test_flux2_vae_diffusers_conversion():
    diffusers = pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLFlux2

    torch.manual_seed(0)
    ref = AutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=[64, 128],
        decoder_block_out_channels=[32, 64],
        layers_per_block=2,
        latent_channels=4,
        norm_num_groups=32,
        sample_size=32,
        mid_block_add_attention=True,
        use_quant_conv=True,
        use_post_quant_conv=True,
        batch_norm_eps=1e-4,
        patch_size=[2, 2],
        force_upcast=True,
    ).eval()

    sd = flux2_vae._convert_diffusers_vae_state_dict(ref.state_dict())
    params = flux2_vae.AutoEncoderParams(ch=64, ch_mult=[1, 2], z_channels=4, decoder_ch=32)
    ae = flux2_vae.AutoEncoder(params).eval()
    ae.load_state_dict(sd, strict=True)

    pixels = torch.randn(1, 3, 32, 32)
    latents = torch.randn(1, 4, 16, 16)
    with torch.no_grad():
        expected_latents = ref.encode(pixels, return_dict=False)[0].mode()
        actual_latents = flux2_vae.unpatchify_latents(ae.inv_normalize(ae.encode(pixels)))
        expected_pixels = ref.decode(latents, return_dict=False)[0]
        actual_pixels = ae.decode(ae.normalize(flux2_vae.patchify_latents(latents)))
    torch.testing.assert_close(actual_latents, expected_latents, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_pixels, expected_pixels, atol=1e-5, rtol=1e-5)


def test_padding_values_cannot_change_predictions():
    _, model = _make_models(False)
    x, times, context, mask = _make_inputs()
    other = context.clone()
    other[0, 4:] = 10000
    torch.testing.assert_close(model(x, times, context, mask), model(x, times, other, mask))


def test_rope_survives_bfloat16_roundtrip():
    _, model = _make_models(False)
    expected = model.rope._default_bands(torch.device("cpu"))
    model.bfloat16().float()
    assert torch.equal(model.rope.bands, expected)


@pytest.mark.parametrize(
    "bad_mask", [torch.zeros(2, 7), torch.ones(2, 6), torch.tensor([[1, 0, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]])]
)
def test_invalid_or_nonprefix_masks_rejected(bad_mask):
    _, model = _make_models(False)
    x, times, context, _ = _make_inputs()
    with pytest.raises(ValueError):
        model(x, times, context, bad_mask)


def test_canonical_checkpoint_roundtrip(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    _, original = _make_models(False)
    monkeypatch.setattr(moss_image_models, "MOSS_IMAGE_CONFIG", SMALL_CONFIG)
    path = tmp_path / "model.safetensors"
    save_file(original.state_dict(), str(path))
    loaded = moss_image_utils.load_moss_image_model("cpu", str(path), "torch", False, "cpu", torch.float32)
    inputs = _make_inputs()
    torch.testing.assert_close(loaded(*inputs), original(*inputs))
    assert set(loaded.state_dict()) == set(original.state_dict())


@pytest.mark.parametrize("prefix", ["model.diffusion_model.", "diffusion_model."])
@pytest.mark.parametrize("fused", [False, True])
def test_prefixed_checkpoint_roundtrip(tmp_path, monkeypatch, prefix, fused):
    from safetensors.torch import save_file

    _, original = _make_models(False)
    monkeypatch.setattr(moss_image_models, "MOSS_IMAGE_CONFIG", SMALL_CONFIG)
    state = dict(original.state_dict())
    if fused:
        for index in range(original.num_layers):
            base = f"layers.{index}.proj."
            state[base + "qkv.weight"] = torch.cat([state.pop(base + part + ".weight") for part in ("q", "k", "v")])
    path = tmp_path / "prefixed.safetensors"
    save_file({prefix + key: value for key, value in state.items()}, str(path))
    loaded = moss_image_utils.load_moss_image_model("cpu", str(path), "torch", False, "cpu", torch.float32)
    torch.testing.assert_close(loaded(*_make_inputs()), original(*_make_inputs()))
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("duplicate", ["fused", "prefixed", "sharded"])
def test_ambiguous_checkpoint_weights_are_rejected(tmp_path, monkeypatch, duplicate):
    from safetensors.torch import save_file

    _, original = _make_models(False)
    monkeypatch.setattr(moss_image_models, "MOSS_IMAGE_CONFIG", SMALL_CONFIG)
    state = dict(original.state_dict())
    path = tmp_path / "ambiguous.safetensors"
    if duplicate == "fused":
        state["layers.0.proj.qkv.weight"] = torch.cat([state[f"layers.0.proj.{part}.weight"] for part in ("q", "k", "v")]) + 100
    elif duplicate == "prefixed":
        state["model.diffusion_model.video_embedder.weight"] = state["video_embedder.weight"] + 100
    else:
        path = tmp_path / "diffusion_pytorch_model-00001-of-00002.safetensors"
        save_file(
            {"video_embedder.weight": state["video_embedder.weight"] + 100},
            str(tmp_path / "diffusion_pytorch_model-00002-of-00002.safetensors"),
        )
    save_file(state, str(path))
    with pytest.raises(ValueError, match="[Dd]uplicate|[Aa]mbiguous"):
        moss_image_utils.load_moss_image_model("cpu", str(path), "torch", False, "cpu", torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CPU activation offloading")
def test_cpu_offload_checkpointing_cuda_forward_backward():
    _, model = _make_models(False)
    model = model.cuda().train()
    model.enable_gradient_checkpointing(cpu_offload=True)
    x, times, context, mask = [tensor.cuda() for tensor in _make_inputs()]
    result = model(x.requires_grad_(), times, context, mask)
    result.square().mean().backward()
    assert result.is_cuda
    assert torch.isfinite(x.grad).all()


def test_existing_flux2_parameter_order_is_unchanged():
    from musubi_tuner.flux_2.flux2_models import AutoEncoderParams

    params = AutoEncoderParams(256, 3, 128, 3, [1, 2, 4, 4], 2, 32)
    assert params.num_res_blocks == 2 and params.z_channels == 32
    assert params.decoder_ch is None


def test_xformers_requires_split_attention_for_variable_text_lengths():
    with pytest.raises(ValueError, match="split_attn"):
        moss_image_models.MossImage(**SMALL_CONFIG, attn_mode="xformers")
    model = moss_image_models.MossImage(**SMALL_CONFIG, attn_mode="xformers", split_attn=True)
    assert model.split_attn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for H2D-only streaming")
def test_h2d_only_streaming_can_be_drained_after_backward():
    from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
    from musubi_tuner.networks.lora_moss_image import create_arch_network

    _, model = _make_models(False)
    model.cuda().train().requires_grad_(False)
    model.enable_gradient_checkpointing()
    model.enable_block_swap(1, BlockSwapConfig(torch.device("cuda"), supports_backward=True, h2d_only=True))
    adapter = create_arch_network(1.0, 2, 2.0, None, None, model)
    adapter.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    adapter.cuda()
    model.move_to_device_except_swap_blocks(torch.device("cuda"))
    model.prepare_block_swap_before_forward()
    x, times, context, mask = [tensor.cuda() for tensor in _make_inputs()]
    model(x, times, context, mask).square().mean().backward()
    model.wait_for_pending_block_moves()
    assert any(parameter.grad is not None for parameter in adapter.parameters())
