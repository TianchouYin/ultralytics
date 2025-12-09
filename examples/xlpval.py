import os
import ultralytics
from ultralytics import YOLO
import yaml
def xlpval():
    ultralytics.checks()
    mymodel = r"runs/detect/CASnoRado\yolo12x\train\bc32imgsz512dpout0.5/weights/best.pt"

    dataset =   r"./ultralytics/cfg/datasets/CASnoRadom.yaml"

    head, tail = os.path.split(mymodel)
    head, tail = os.path.split(head)
    head, weightName = os.path.split(head)
    model = YOLO(mymodel)  # load a pretrained model (recommended for training)
    print('*mymodel='+mymodel)

    # 从YAML文件中加载字典
    with open('ultralytics/cfg/default.yaml', 'r') as file:
        yaml_dict = yaml.safe_load(file)
    name = (f"{yaml_dict['split']}-{model.ckpt['train_args']['model'].rstrip('.yaml')}/{yaml_dict['split']}_{os.path.basename(dataset.rstrip('.yaml'))}+"
            # f"weit={weightName}+"
            f"conf{yaml_dict['conf']}+"
            f"bc{yaml_dict['batch']}"
            f"iz{yaml_dict['imgsz']}_")
    results = model.val(data=dataset, name=name,visualize=True)

if __name__ == '__main__':
    xlpval()
