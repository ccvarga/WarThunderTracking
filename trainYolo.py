from ultralytics import YOLO
import torch_directml
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

model.train(
    data="data.yaml",                   # Path to your dataset YAML file
    epochs=50,                          # Number of training epochs
    batch=-1,                           # Batch size (adjust based on GPU memory)
    imgsz=960,                         # Input image size (adjust based on your images)
    model="yolov8n.pt",                 # Path to pretrained weights (if starting from pretrained)
    device=torch_directml.device(),                       # GPU device to use (set to "cpu" if no GPU available)
    project="war-thunder-targeting",    # Name of the training project
    name="exp1",                        # Name of the training experiment
    cache=True
)