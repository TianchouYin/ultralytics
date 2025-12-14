import os
from pathlib import Path
import sys
import logging
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.logger import ConsoleLogger
import yaml

def xlpval(num):
    # 禁用 TQDM 的动态更新
    os.environ['TQDM_MININTERVAL'] = '1'
    
    ultralytics.checks()

    # 从YAML文件中加载字典
    with open('runs/classify/train'+str(num)+'/args.yaml', 'r') as file:
        yaml_dict = yaml.safe_load(file)
        yaml_dict['split'] = 'test'
    
    mymodel = f"{yaml_dict['save_dir']}/weights/best.pt"
    dataset = f"{yaml_dict['data']}"
    
    model = YOLO(mymodel)
    
    save_dir_root = f"{yaml_dict['save_dir']}/{yaml_dict['split']}-{model.ckpt['train_args']['model'].rstrip('.yaml')}"
    save_dir = (f"{save_dir_root}/{yaml_dict['split']}_{os.path.basename(dataset.rstrip('.yaml'))}+"
            f"conf{yaml_dict['conf']}+"
            f"bc{yaml_dict['batch']}"
            f"iz{yaml_dict['imgsz']}_")
    
    # 关键修改：使用 increment_path 预先计算实际会使用的路径
    # 这样可以和 YOLO 内部使用的路径保持一致
    actual_save_dir = increment_path(Path(save_dir), exist_ok=False)
    actual_save_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(actual_save_dir, exist_ok=True)
    
    # 使用 ConsoleLogger 自动处理输出清理和去重
    output_path = os.path.join(actual_save_dir, 'output.txt')
    logger = ConsoleLogger(output_path)
    
    logger.start_capture()
    
    print('*mymodel=' + mymodel)
    
    try:
        results = model.val(
            name=str(actual_save_dir), 
            visualize=True, 
            split=yaml_dict['split'],
            save_txt=True, 
            save_conf=True,
            save_json=True, 
            save_hybrid=True, 
            verbose=False,
            exist_ok=True  # 重要：允许使用已存在的目录 因为此前已经处理过了重复的情况
        )
        
        # 手动打印结果确保被捕获
        print(f"\nResults: {results}")
        # print(f"Top-1 Accuracy: {results.top1:.4f}")
        # print(f"Top-5 Accuracy: {results.top5:.4f}")
        
    finally:
        logger.stop_capture()
    
    print(f"Log saved to: {output_path}")

if __name__ == '__main__':
    # for i in ['', '2', '3', '4', '5', '6', '7']:
    for i in [2]:
        xlpval(i)
