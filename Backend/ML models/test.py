from ultralytics import YOLO

model = YOLO("Backend/ML models/weights/best.pt")

print("Epoch:", model.ckpt.get("epoch"))
print("Best fitness:", model.ckpt.get("best_fitness"))