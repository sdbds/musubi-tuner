from dataclasses import dataclass
from pathlib import Path
from typing import Optional


LANCE_VAE_DOWNSAMPLE = (4, 16, 16)
LANCE_VAE_Z_CHANNELS = 48
LANCE_DEFAULT_LATENT_PATCH_SIZE = (1, 1, 1)


@dataclass(frozen=True)
class LanceModelPaths:
    model_path: Path
    llm_config: Path
    tokenizer_path: Path
    vit_path: Optional[Path]
    vae_path: Optional[Path]


def _optional_existing_path(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def resolve_lance_model_paths(model_path: str, vit_path: Optional[str] = None, vae_path: Optional[str] = None) -> LanceModelPaths:
    """Resolve the public Lance checkpoint tree without changing user layout."""
    root = Path(model_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Lance model path does not exist: {root}")

    llm_config = root / "llm_config.json"
    if not llm_config.exists():
        raise FileNotFoundError(f"Lance llm_config.json not found under model path: {llm_config}")

    resolved_vit_path = Path(vit_path).expanduser().resolve() if vit_path else _optional_existing_path(root / "Qwen2.5-VL-ViT")
    resolved_vae_path = Path(vae_path).expanduser().resolve() if vae_path else _optional_existing_path(root / "Wan2.2_VAE.pth")

    if vit_path is not None and not resolved_vit_path.exists():
        raise FileNotFoundError(f"Lance ViT path does not exist: {resolved_vit_path}")
    if vae_path is not None and not resolved_vae_path.exists():
        raise FileNotFoundError(f"Lance VAE path does not exist: {resolved_vae_path}")

    return LanceModelPaths(
        model_path=root,
        llm_config=llm_config,
        tokenizer_path=root,
        vit_path=resolved_vit_path,
        vae_path=resolved_vae_path,
    )
