"""
超参数批量测试脚本
支持一次性测试多个超参数组合，并自动记录结果
"""

import itertools
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO


def run_hyperparameter_sweep(
    base_config: str | dict,
    hyperparameters: dict[str, list],
    model_path: str = "yolo11n-cls.pt",
    output_dir: str = "runs/sweep",
    epochs: int = 100,
    imgsz: int = 256,
):
    """
    批量测试多个超参数组合

    Args:
        base_config: 基础配置文件路径或配置字典
        hyperparameters: 要测试的超参数及其候选值，如 {"lr0": [0.01, 0.001], "dropout": [0.1, 0.3]}
        model_path: 模型路径
        output_dir: 输出目录
        epochs: 训练轮数
        imgsz: 图像尺寸

    Returns:
        DataFrame: 包含所有实验结果的表格
    """
    # 加载基础配置
    if isinstance(base_config, str):
        with open(base_config, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
    else:
        base_cfg = base_config.copy()

    # 生成所有超参数组合
    param_names = list(hyperparameters.keys())
    param_values = list(hyperparameters.values())
    combinations = list(itertools.product(*param_values))

    print(f"共有 {len(combinations)} 个超参数组合待测试")
    print(f"测试参数: {param_names}")
    print("-" * 60)

    # 存储结果
    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 记录开始时间
    sweep_start = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = output_path / f"sweep_results_{sweep_start}.csv"

    for idx, combo in enumerate(combinations):
        # 构建当前超参数组合
        current_params = dict(zip(param_names, combo))
        exp_name = "_".join([f"{k}{v}" for k, v in current_params.items()])

        print(f"\n[{idx + 1}/{len(combinations)}] 测试超参数: {current_params}")

        try:
            # 创建模型
            model = YOLO(model_path)

            # 合并训练参数
            train_args = {
                "data": base_cfg.get("data"),
                "epochs": epochs,
                "imgsz": imgsz,
                "project": str(output_path),
                "name": exp_name,
                "exist_ok": True,
                "verbose": False,
                **current_params,  # 覆盖为当前测试的超参数
            }

            # 训练模型
            train_results = model.train(
                **train_args, cfg="cfg/defaultClassify240.yaml", batch=256
            )

            # 提取关键指标
            metrics = {
                "experiment": exp_name,
                **current_params,
                "top1_accuracy": getattr(train_results, "top1", None),
                "top5_accuracy": getattr(train_results, "top5", None),
                "fitness": getattr(train_results, "fitness", None),
            }

            results.append(metrics)
            print(
                f"  ✓ 完成 - Top1: {metrics['top1_accuracy']:.4f}"
                if metrics["top1_accuracy"]
                else "  ✓ 完成"
            )

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results.append({"experiment": exp_name, **current_params, "error": str(e)})

        # 每次实验后保存中间结果
        df = pd.DataFrame(results)
        df.to_csv(results_csv, index=False, encoding="utf-8")

    # 最终结果汇总
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("超参数测试完成！结果汇总:")
    print(df.to_string(index=False))
    print(f"\n结果已保存至: {results_csv}")

    return df


def run_grid_search(
    data: str,
    model_path: str = "yolo11n-cls.pt",
    output_dir: str = "runs/grid_search",
    epochs: int = 50,
):
    """
    网格搜索示例 - 预设常用超参数组合

    Args:
        data: 数据集路径
        model_path: 模型路径
        output_dir: 输出目录
        epochs: 训练轮数
    """
    # 定义要搜索的超参数空间
    hyperparameters = {
        "lr0": [0.01, 0.005, 0.001],
        "dropout": [0.0, 0.3, 0.5],
        "weight_decay": [0.0005, 0.001],
    }

    base_config = {"data": data}

    return run_hyperparameter_sweep(
        base_config=base_config,
        hyperparameters=hyperparameters,
        model_path=model_path,
        output_dir=output_dir,
        epochs=epochs,
    )


def run_custom_sweep():
    """自定义超参数测试 - 根据你的配置文件"""
    # 你的数据路径
    data_path = "/data/users/lxing/File/medicalImg/CAS/selected_folders/plaque/dataset"

    # 要测试的超参数组合
    hyperparameters = {
        # 学习率
        "lr0": [0],
        # Dropout
        "dropout": [0, 0.1, 0.3, 0.5],
        # 权重衰减
        # "weight_decay": [0.0005, 0.001, 0.005],
        "weight_decay": [0.01, 0.05, 0.1],
    }

    # 也可以测试其他超参数
    # hyperparameters = {
    #     "lr0": [0.01, 0.001],
    #     "lrf": [0.01, 0.1],
    #     "momentum": [0.9, 0.937],
    #     "warmup_epochs": [1.0, 3.0, 5.0],
    #     "cos_lr": [True, False],
    # }

    base_config = {
        "data": data_path,
        "cache": True,
        "device": -1,
        "cos_lr": True,
        "pretrained": True,
    }

    return run_hyperparameter_sweep(
        base_config=base_config,
        hyperparameters=hyperparameters,
        model_path="yolo11m-cls.pt",
        output_dir="runs/classify/plaque_sweep_m",
        epochs=100,
        imgsz=256,
    )


if __name__ == "__main__":
    # 运行自定义超参数测试
    results = run_custom_sweep()

    # 找出最佳超参数组合
    if "top1_accuracy" in results.columns:
        best_idx = results["top1_accuracy"].idxmax()
        print("\n🏆 最佳超参数组合:")
        print(results.iloc[best_idx])
