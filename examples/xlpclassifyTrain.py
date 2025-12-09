from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x-cls.yaml")  # build a new model from YAML
model = YOLO("yolo11x-cls.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11x-cls.yaml").load("yolo11x-cls.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/data/users/lxing/File/medicalImg/CAS/selected_folders/plaque/dataset",  imgsz=128,cfg="/data/users/lxing/gitRepository/ultralytics250316/ultralytics/cfg/defaultNewClassify.yaml")