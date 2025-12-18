import torch
import os

# 替换为您要检查的权重文件的路径
# 例如，如果您想检查训练后保存的last.pt文件：
# weights_path = "runs/classify/plaque/yolo11x-cls/train/weights/last.pt"
# 或者您脚本中使用的预训练权重文件：
weights_path = "yolo11x-cls.pt" 
# weights_path = "runs/classify/plaque/yolo11x-cls.yaml/train/weights/best.pt" 

if not os.path.exists(weights_path):
    print(f"错误：权重文件 '{weights_path}' 不存在。请检查路径是否正确。")
else:
    try:
        # 加载权重文件
        # map_location='cpu' 可以确保即使您在GPU上训练，也可以在CPU上加载，避免GPU内存问题
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

        print(f"加载的权重文件: {weights_path}")
        print("检查点中包含的键:")
        for key in checkpoint.keys():
            print(f"- {key}")

        print("--- 检查点中包含的键及其值 ---")
        for key, value in checkpoint.items(): # 遍历键和值
            if key == 'model':
                # 对于 'model' 键，只打印其类型或简短描述，因为其详细权重在后面打印
                print(f"- {key}: <PyTorch Model / State Dict Object> (详细权重信息另行打印)")
            elif key == 'train_results' and isinstance(value, dict):
                # 特别处理 'train_results' 字典，打印其所有内容
                print(f"- {key}: Dictionary (内容如下):")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        print(f"  - {sub_key}: Tensor (shape={sub_value.shape}, dtype={sub_value.dtype})")
                    elif isinstance(sub_value, (list, tuple)):
                        list_length = len(sub_value)
                        if list_length > 10:  # 如果列表长度大于10，打印前5个和后5个
                            print(f"  - {sub_key}: {type(sub_value).__name__} (len={list_length}) - [{sub_value[:5]} ... {sub_value[-5:]}]")
                        else:  # 否则，打印整个列表/元组
                            print(f"  - {sub_key}: {type(sub_value).__name__} (len={list_length}) - {sub_value}")
                    elif isinstance(sub_value, dict):
                         print(f"  - {sub_key}: Dictionary (keys: {list(sub_value.keys())})")
                    else:
                        print(f"  - {sub_key}: {sub_value}")                
            elif isinstance(value, torch.Tensor):
                print(f"- {key}: Tensor (shape={value.shape}, dtype={value.dtype})")
            elif isinstance(value, dict):
                print(f"- {key}: Dictionary (keys: {list(value.keys())})")
            elif isinstance(value, (list, tuple)):
                print(f"- {key}: {type(value).__name__} (len={len(value)}) - {value[:5]}{'...' if len(value) > 5 else ''}")
            elif isinstance(value, (int, float, str, bool, type(None))):
                print(f"- {key}: {value}")
            else:
                print(f"- {key}: <Object of type {type(value).__name__}>")
                
                            
        # 检查是否存在 'epoch' 键
        if 'epoch' in checkpoint:
            loaded_epoch = checkpoint['epoch']
            print(f"从权重文件 '{weights_path}' 中读取到的 epoch 为: {loaded_epoch}")
        else:
            print(f"权重文件 '{weights_path}' 中未找到 'epoch' 信息。")

    except Exception as e:
        print(f"加载权重文件 '{weights_path}' 时发生错误: {e}")
