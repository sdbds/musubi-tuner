import argparse
import importlib
import importlib.machinery
import sys
import types
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _ensure_package(name: str):
    package_path = []
    if name == "musubi_tuner" or name.startswith("musubi_tuner."):
        package_dir = SRC_ROOT.joinpath(*name.split("."))
        if package_dir.exists():
            package_path = [str(package_dir)]

    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        module.__path__ = package_path
        sys.modules[name] = module
    elif not hasattr(module, "__path__"):
        module.__path__ = package_path
    elif package_path and not module.__path__:
        module.__path__ = package_path
    return module


def _register_module(name: str, **attrs):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = _ensure_package(parent_name)
        setattr(parent, child_name, module)
    return module


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def import_zimage_train_network_module():
    class DummyNetworkTrainer:
        pass

    _ensure_package("musubi_tuner")
    _ensure_package("musubi_tuner.dataset")
    _ensure_package("musubi_tuner.zimage")
    _ensure_package("musubi_tuner.utils")
    _register_module(
        "musubi_tuner.dataset.image_video_dataset",
        ARCHITECTURE_Z_IMAGE="zi",
        ARCHITECTURE_Z_IMAGE_FULL="z_image",
    )
    _register_module(
        "musubi_tuner.zimage.zimage_config",
        SEQ_MULTI_OF=1,
        ZIMAGE_VAE_SHIFT_FACTOR=0.0,
        ZIMAGE_VAE_SCALING_FACTOR=1.0,
        ZIMAGE_VAE_SCALE_FACTOR=8,
    )
    _register_module(
        "musubi_tuner.zimage.zimage_model",
        ZImageTransformer2DModel=object,
        load_zimage_model=lambda *args, **kwargs: None,
    )
    _register_module("musubi_tuner.zimage.zimage_utils")
    _register_module(
        "musubi_tuner.zimage.zimage_autoencoder",
        load_autoencoder_kl=lambda *args, **kwargs: None,
    )
    _register_module(
        "musubi_tuner.hv_train_network",
        NetworkTrainer=DummyNetworkTrainer,
        load_prompts=lambda *args, **kwargs: [],
        clean_memory_on_device=lambda *args, **kwargs: None,
        setup_parser_common=lambda: argparse.ArgumentParser(),
        read_config_from_file=lambda args, parser: args,
    )
    _register_module(
        "musubi_tuner.utils.model_utils",
        dtype_to_str=lambda dtype: str(dtype),
        compile_transformer=lambda *args, **kwargs: None,
    )

    sys.modules.pop("musubi_tuner.zimage_train_network", None)
    importlib.invalidate_caches()
    return importlib.import_module("musubi_tuner.zimage_train_network")


def import_zimage_train_module():
    class DummyZImageNetworkTrainer:
        pass

    def zimage_setup_parser(parser):
        return parser

    _ensure_package("musubi_tuner")
    _ensure_package("musubi_tuner.dataset")
    _ensure_package("musubi_tuner.modules")
    _ensure_package("musubi_tuner.zimage")
    _ensure_package("musubi_tuner.utils")
    _register_module(
        "musubi_tuner.zimage_train_network",
        ZImageNetworkTrainer=DummyZImageNetworkTrainer,
        zimage_setup_parser=zimage_setup_parser,
    )
    _register_module(
        "musubi_tuner.dataset.config_utils",
        BlueprintGenerator=type("BlueprintGenerator", (), {}),
        ConfigSanitizer=type("ConfigSanitizer", (), {}),
    )
    _register_module(
        "musubi_tuner.modules.scheduling_flow_match_discrete",
        FlowMatchDiscreteScheduler=type("FlowMatchDiscreteScheduler", (), {}),
    )
    _register_module("musubi_tuner.zimage.zimage_model")
    _register_module(
        "musubi_tuner.hv_train_network",
        SS_METADATA_KEY_BASE_MODEL_VERSION="ss_base_model_version",
        SS_METADATA_MINIMUM_KEYS=[],
        collator_class=lambda *args, **kwargs: None,
        clean_memory_on_device=lambda *args, **kwargs: None,
        compute_loss_weighting_for_sd3=lambda *args, **kwargs: None,
        get_sigmas=lambda *args, **kwargs: None,
        prepare_accelerator=lambda *args, **kwargs: None,
        setup_parser_common=lambda: argparse.ArgumentParser(),
        read_config_from_file=lambda args, parser: args,
        should_sample_images=lambda *args, **kwargs: False,
        set_seed=lambda *args, **kwargs: None,
    )
    _register_module("musubi_tuner.utils.huggingface_utils")
    _register_module("musubi_tuner.utils.model_utils")
    _register_module("musubi_tuner.utils.sai_model_spec")
    _register_module("musubi_tuner.utils.train_utils")
    _register_module(
        "musubi_tuner.utils.safetensors_utils",
        mem_eff_save_file=lambda *args, **kwargs: None,
    )

    sys.modules.pop("musubi_tuner.zimage_train", None)
    importlib.invalidate_caches()
    return importlib.import_module("musubi_tuner.zimage_train")


def import_flux_2_train_network_module():
    class DummyNetworkTrainer:
        pass

    if not _module_exists("diffusers.utils.torch_utils"):
        _ensure_package("diffusers")
        _ensure_package("diffusers.utils")
        _register_module("diffusers.utils.torch_utils", randn_tensor=lambda *args, **kwargs: None)
    if not _module_exists("einops"):
        _register_module("einops", rearrange=lambda tensor, *args, **kwargs: tensor)

    _ensure_package("musubi_tuner")
    _ensure_package("musubi_tuner.flux_2")
    _ensure_package("musubi_tuner.utils")
    _register_module("musubi_tuner.flux_2.flux2_models", Flux2=object)
    _register_module(
        "musubi_tuner.flux_2.flux2_utils",
        FLUX2_MODEL_INFO={},
        add_model_version_args=lambda parser: parser,
    )
    _register_module(
        "musubi_tuner.hv_train_network",
        NetworkTrainer=DummyNetworkTrainer,
        load_prompts=lambda *args, **kwargs: [],
        clean_memory_on_device=lambda *args, **kwargs: None,
        setup_parser_common=lambda: argparse.ArgumentParser(),
        read_config_from_file=lambda args, parser: args,
    )
    _register_module(
        "musubi_tuner.utils.model_utils",
        compile_transformer=lambda *args, **kwargs: None,
    )

    sys.modules.pop("musubi_tuner.flux_2_train_network", None)
    importlib.invalidate_caches()
    return importlib.import_module("musubi_tuner.flux_2_train_network")


def import_qwen_image_train_network_module():
    class DummyNetworkTrainer:
        pass

    _ensure_package("musubi_tuner")
    _ensure_package("musubi_tuner.dataset")
    _ensure_package("musubi_tuner.qwen_image")
    _ensure_package("musubi_tuner.utils")
    _register_module(
        "musubi_tuner.dataset.image_video_dataset",
        ARCHITECTURE_QWEN_IMAGE="qwen_image",
        ARCHITECTURE_QWEN_IMAGE_FULL="Qwen Image",
        ARCHITECTURE_QWEN_IMAGE_EDIT="qwen_image_edit",
        ARCHITECTURE_QWEN_IMAGE_EDIT_FULL="Qwen Image Edit",
        ARCHITECTURE_QWEN_IMAGE_LAYERED="qwen_image_layered",
        ARCHITECTURE_QWEN_IMAGE_LAYERED_FULL="Qwen Image Layered",
    )
    _register_module("musubi_tuner.qwen_image.qwen_image_autoencoder_kl", AutoencoderKLQwenImage=object)
    _register_module(
        "musubi_tuner.qwen_image.qwen_image_model",
        QwenImageTransformer2DModel=object,
        load_qwen_image_model=lambda *args, **kwargs: None,
    )
    _register_module(
        "musubi_tuner.qwen_image.qwen_image_utils",
        VAE_SCALE_FACTOR=8,
        add_model_version_args=lambda parser: parser,
        resolve_model_version_args=lambda args: args,
    )
    _register_module(
        "musubi_tuner.hv_train_network",
        NetworkTrainer=DummyNetworkTrainer,
        load_prompts=lambda *args, **kwargs: [],
        clean_memory_on_device=lambda *args, **kwargs: None,
        setup_parser_common=lambda: argparse.ArgumentParser(),
        read_config_from_file=lambda args, parser: args,
    )
    _register_module(
        "musubi_tuner.utils.model_utils",
        compile_transformer=lambda *args, **kwargs: None,
    )
    _register_module(
        "musubi_tuner.utils.sai_model_spec",
        CUSTOM_ARCH_QWEN_IMAGE_EDIT_PLUS="qwen_image_edit_plus",
        CUSTOM_ARCH_QWEN_IMAGE_EDIT_2511="qwen_image_edit_2511",
    )

    sys.modules.pop("musubi_tuner.qwen_image_train_network", None)
    importlib.invalidate_caches()
    return importlib.import_module("musubi_tuner.qwen_image_train_network")
