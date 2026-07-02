import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "src" / "musubi_tuner" / "krea2_train_network.py"


def _method_ast(class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def _is_call_to(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _is_method_call(node: ast.AST, attr: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attr


def test_krea2_sample_decode_moves_pixels_to_cpu_before_cuda_cleanup():
    method = _method_ast("Krea2NetworkTrainer", "do_inference")

    clean_lines = [node.lineno for node in ast.walk(method) if _is_call_to(node, "clean_memory_on_device")]
    cpu_lines = [node.lineno for node in ast.walk(method) if _is_method_call(node, "cpu")]

    assert cpu_lines, "do_inference should move decoded pixels to CPU before CUDA cleanup"
    assert min(cpu_lines) < min(clean_lines), "decoded pixels still live on GPU when clean_memory_on_device is called"


def test_krea2_sample_prompt_gpu_temporaries_are_deleted_before_cuda_cleanup():
    method = _method_ast("Krea2NetworkTrainer", "process_sample_prompts")

    clean_line = min(node.lineno for node in ast.walk(method) if _is_call_to(node, "clean_memory_on_device"))
    deleted_name_sets = []
    for node in ast.walk(method):
        if isinstance(node, ast.Delete) and node.lineno < clean_line:
            deleted_name_sets.append({target.id for target in node.targets if isinstance(target, ast.Name)})

    expected = {"hiddens", "mask", "embed"}
    assert any(expected <= names for names in deleted_name_sets), "sample prompt GPU temporaries survive until CUDA cleanup"
