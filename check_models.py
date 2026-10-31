
from pathlib import Path
from ultralytics import YOLO

# Find the two model files
model_folder = Path(__file__).parent / "model"

organism_path = model_folder / "best.pt"
pollution_path = model_folder / "pollution_best.pt"

# Check whether both files exist
if not organism_path.exists():
    print("ERROR: model/best.pt was not found.")

elif not pollution_path.exists():
    print("ERROR: model/pollution_best.pt was not found.")

else:
    print("Loading marine organism model...")
    organism_model = YOLO(str(organism_path))

    print("Loading marine pollution model...")
    pollution_model = YOLO(str(pollution_path))

    print("\nSUCCESS: Both models loaded!")

    print("\nMarine organism classes:")
    print(organism_model.names)

    print("\nMarine pollution classes:")
    print(pollution_model.names)
