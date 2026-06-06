from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    weights_path = Path(__file__).resolve().parent.parent / "runs" / "obb" / "train2" / "weights" / "last.pt"

    weights_path = weights_path.resolve()

    print(weights_path)

    model = YOLO(str(weights_path))
    model.train(resume=True, workers=0)
