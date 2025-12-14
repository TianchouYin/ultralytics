from pathlib import Path
from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11m-cls.yaml")  # build a new model from YAML
# model = YOLO("yolo11m-cls.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11x-cls.yaml").load( "yolo11x-cls.pt")  # build from YAML and transfer weights
modelname = model.model_name

# Train the model
results = model.train(
    project="runs/classify/plaque",
    name=modelname,
    data="/data/users/lxing/File/medicalImg/CAS/selected_folders/plaque/dataset",
    imgsz=128,
    cfg="ultralytics/cfg/defaultNewClassify.yaml",
)
# 从results对象中获取保存路径，并将整个results对象和modelname保存到txt
results_save_path = Path(results.save_dir) / "a_train_results.txt"
with open(results_save_path, 'w') as f:
    f.write(f"Model Name: {modelname}\n") # 追加modelname
    f.write(str(results))
    # f.write(str(results.results_dict)) # 仅保存results_dict

print(f"Full results and model name saved to: {results_save_path}")