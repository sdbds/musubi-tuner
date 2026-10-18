import torch
from safetensors.torch import save_file

from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.moss_image import cache, utils
from musubi_tuner.moss_image_cache_text_encoder_outputs import encode_and_save_batch


def tiny_text_model(hidden_size=32):
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=hidden_size,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        use_cache=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 2],
            "mrope_interleaved": True,
        },
    )
    model = Qwen3_5TextModel(config).eval()
    model.load_state_dict({key: value.to(torch.bfloat16) for key, value in model.state_dict().items()}, assign=True)
    return model, config


def test_qwen35_loader_strips_prefix_and_preserves_fp32_rope(tmp_path, monkeypatch):
    model, config = tiny_text_model()
    monkeypatch.setattr(utils, "load_text_encoder_config", lambda: config)
    state = {"model.language_model." + key: value for key, value in model.state_dict().items()}
    state["model.visual.fake.weight"] = torch.ones(2, 2)
    state["lm_head.weight"] = torch.ones(2, 2)
    path = tmp_path / "text.safetensors"
    save_file(state, path)
    loaded = utils.load_text_encoder(str(path), device="cpu")
    assert loaded.rotary_emb.inv_freq.dtype == torch.float32
    ids = [torch.tensor([1, 4, 7]), torch.tensor([1, 4, 7, 9, 12])]
    with torch.no_grad():
        rows = utils.encode_tokens(loaded, ids)
        for row, tokens in zip(rows, ids):
            expected = model(input_ids=tokens[None], use_cache=False).last_hidden_state[0]
            assert torch.equal(row, expected)
            assert row.dtype == torch.bfloat16
            assert torch.isfinite(row).all()


def test_text_cache_entrypoint_encodes_unpadded_qwen35_rows(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "short": 1, "longer": 2, "caption": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[UNK]")
    model, _ = tiny_text_model(hidden_size=4096)
    items = []
    for index, text in enumerate(["short", "longer caption"]):
        item = ItemInfo(str(index), text, (32, 32), (32, 32))
        item.text_encoder_output_cache_path = str(tmp_path / f"{index}_moss_te.safetensors")
        items.append(item)
    with torch.no_grad():
        uncond = utils.encode_tokens(model, utils.tokenize_prompts(tokenizer, [""]))[0]
        encode_and_save_batch(tokenizer, model, items, torch.device("cpu"), uncond)
        for item in items:
            expected = utils.encode_tokens(model, utils.tokenize_prompts(tokenizer, [item.caption]))[0]
            actual = cache.load_cache(item.text_encoder_output_cache_path, "text")
            assert torch.equal(actual[cache.TEXT_KEYS[0]], expected)
            assert torch.equal(actual[cache.TEXT_KEYS[1]], uncond)


def test_packaged_qwen_config_and_prompt_template():
    config = utils.load_text_encoder_config()
    assert config.hidden_size == 4096
    assert config.use_cache is False
    assert utils.format_prompt("draw a cup") == "<|im_start|>user\ndraw a cup<|im_end|>\n<|im_start|>assistant\n<think>\n"
