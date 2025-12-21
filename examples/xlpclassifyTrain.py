from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parents[1]#本文件为起始目录
sys.path.insert(0, str(repo_root))  # 确保本地源码优先

from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
# model = YOLO("yolo11m-cls.pt")  # load a pretrained model (recommended for training)
# model = YOLO("resnet18",pretrained=True)  # load a pretrained model (recommended for training)

model = YOLO("yolo11m-cls.yaml").load("yolo11m-cls.pt")  # build from YAML and transfer weights
# model = YOLO("yolov8x-cls.yaml").load( "yolov8x-cls.pt")  # build from YAML and transfer weights
modelname = model.model_name

ch = 3
# Train the model
results = model.train(
    name=f'{modelname}/train',
    # cfg="ultralytics/cfg/defaultNewClassify.yaml",
    # cfg="runs/classify/plaque/yolo11x-cls.yaml/train6/args.yaml",
    # cfg="cfg/default_copy.yaml",
    cfg="cfg/defaultClassify240.yaml",
    imgsz=256,
    patience=100,
    # ch=ch, #报错 无该参数'ch' is not a valid YOLO argument. 
    # TODO DEBUG
    epochs=1,
    # batch=16,
    save=True,  # 分类不适用
    # fraction=0.01,
)

# 获取最优epoch
# 从results_dict中查找
if hasattr(results, 'results_dict'):
    # 查找验证精度最高的epoch
    val_acc_list = results.results_dict.get('metrics/accuracy_top1', [])
    # 检查val_acc_list的类型
    if isinstance(val_acc_list, list) and len(val_acc_list) > 0:
        best_epoch = val_acc_list.index(max(val_acc_list))
    elif isinstance(val_acc_list, (int, float)):
        # 如果只有一个数值，说明只训练了1个epoch
        best_epoch = 0
    else:
        best_epoch = "未找到"
        
# 从results对象中获取保存路径，并将整个results对象和modelname保存到txt
results_save_path = Path(results.save_dir) / f"a_results_train_ch{ch}.txt"
with open(results_save_path, 'w') as f:
    f.write(f"Model Name: {modelname}\n")  # 追加modelname
    f.write("[2400 / (2 * 600)=2, 2400 / (2 * 1800)= 0.6667] \n\
    weight = torch.as_tensor([2,0.667], device=preds.device, dtype=preds.dtype)\n\
    loss = F.cross_entropy(preds, batch[\"cls\"], weight=weight, reduction=\"mean\")\n")  # 追加weighted accuracy计算公式
    f.write(f"Best Epoch: {best_epoch}\n")  # 追加最优epoch
    f.write(f"input Channel: {ch}\n")  # 追加channel信息
    f.write(str(results))
    # f.write(str(results.results_dict)) # 仅保存results_dict

print(f"Full results and model name saved to: {results_save_path}")