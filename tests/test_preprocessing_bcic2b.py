from pathlib import Path

import numpy as np
from scipy.io import savemat

from bci_maze.preprocessing_bcic2b import BCIC2bPreprocessConfig, read_bnci_mat


def test_read_bnci_mat_extracts_three_channels_and_binary_labels(tmp_path: Path):
    config = BCIC2bPreprocessConfig()
    signal = np.zeros((3000, 6), dtype=np.float32)
    session = {
        "X": signal,
        "trial": np.asarray([1, 1001], dtype=np.int32),
        "y": np.asarray([1, 2], dtype=np.uint8),
        "fs": 250,
        "classes": np.asarray(["left hand", "right hand"], dtype=object),
        "artifacts": np.asarray([0, 1], dtype=np.uint8),
    }
    path = tmp_path / "B01T.mat"
    savemat(path, {"data": np.asarray([session], dtype=object)})
    result = read_bnci_mat(path, config)
    assert result["x"].shape == (2, 3, 1000)
    assert result["raw_x"].shape == (2, 3, 1000)
    assert result["y"].tolist() == [0, 1]
    assert result["artifact"].tolist() == [False, True]

