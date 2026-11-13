import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_CORE = (
    PROJECT_ROOT / "src/interpolation_study/resize.py",
    PROJECT_ROOT / "src/interpolation_study/kernels.py",
    PROJECT_ROOT / "src/interpolation_study/metrics.py",
)
ALLOWED_EXTERNAL_IMPORTS = {"numpy", "collections", "__future__"}
FORBIDDEN_RESAMPLING_LIBRARIES = {
    "PIL",
    "cv2",
    "scipy",
    "skimage",
    "torch",
    "tensorflow",
    "imageio",
    "kornia",
}


def test_scratch_core_does_not_import_external_resampling_libraries():
    for path in SCRATCH_CORE:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
                assert not roots.intersection(FORBIDDEN_RESAMPLING_LIBRARIES), path
                assert roots.issubset(ALLOWED_EXTERNAL_IMPORTS), (path, roots)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    # Внутренние импорты пакета допустимы (например, .kernels).
                    continue
                root = (node.module or "").split(".")[0]
                assert root not in FORBIDDEN_RESAMPLING_LIBRARIES, path
                assert root in ALLOWED_EXTERNAL_IMPORTS, (path, root)
