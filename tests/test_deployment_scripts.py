from pathlib import Path

from scripts.evaluate_checkpoints import CHECKPOINT_DIRS
from scripts.export_model import default_export_path


def test_checkpoint_evaluation_uses_dataset_scoped_directories():
    for dataset, models in CHECKPOINT_DIRS.items():
        assert all(Path(path).parts[0] == dataset for path in models.values())


def test_default_export_path_includes_dataset_model_and_subject():
    checkpoint_path = Path("outputs/checkpoints/bcic2a/fbcnet/A01.pt")
    checkpoint = {
        "model_name": "fbcnet",
        "config": {"dataset_name": "bcic2a"},
    }
    output = default_export_path(checkpoint_path, checkpoint)
    assert output.name == "bcic2a-fbcnet-A01.torchscript.pt"
