from pathlib import Path


dataset_path = Path("/home/zonghuan/tudelft/projects/datasets/conflab/")

ALL_CLUES = ["head", "shoulder", "hip", "foot"]
RAW_jSON_PATH = dataset_path / "annotations/pose/coco/"
SEGS = ["429", "431", "631", "634", "636", "831", "832", "833", "833", "834", "835"]