from ultralytics import YOLO

# Build a YOLOv9c model from pretrained weight
model = YOLO("yolov9c-seg.pt")

# Display model information (optional)
model.info()

# Train the model
results = model.train(data="./datasets/annotation.json", epochs=100, imgsz=520)