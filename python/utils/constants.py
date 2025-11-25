from pathlib import Path


dataset_path = Path("/home/zonghuan/tudelft/projects/datasets/conflab/")

ALL_CLUES = ["head", "shoulder", "hip", "foot"]
RAW_jSON_PATH = dataset_path / "annotations/pose/coco/"
GROUP_ANNOTATIONS_PATH = dataset_path / "annotations/f_formations/"
# SEGS = ["429", "431", "631", "634", "636", "831", "832", "833", "833", "834", "835"]

SEGS = ["229", "232" "233", "235", "236", "429", "431", "434", "628", "629", "631", 
"634", "636", "828", "833", "834", "835"]