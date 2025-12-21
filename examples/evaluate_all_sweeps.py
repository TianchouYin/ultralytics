"""
批量评估脚本
评估所有训练好的模型在测试集上的 AUC、各类别 P/R/F1、整体 P/R/F1
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# 确保可以导入 ultralytics
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO


def evaluate_single_model(
    model_path: Path,
    data_path: str,
    imgsz: int = 256,
    batch: int = 32,
    device: str = "0",
) -> dict:
    """
    评估单个模型

    Args:
        model_path: best.pt 模型路径
        data_path: 数据集路径
        imgsz: 图像尺寸
        batch: 批次大小
        device: 设备

    Returns:
        dict: 包含所有评估指标的字典
    """
    try:
        # 加载模型
        model = YOLO(str(model_path))

        # 运行验证
        results = model.val(
            data=data_path,
            imgsz=imgsz,
            batch=batch,
            device=device,
            split="test",  # 使用测试集
            verbose=False,
        )

        # 获取预测结果用于计算 AUC
        # 需要重新预测以获取概率
        from ultralytics.data.build import build_classification_dataloader
        from ultralytics.data.utils import check_cls_dataset

        # 获取测试集路径
        dataset_info = check_cls_dataset(data_path)
        test_path = dataset_info.get("test", dataset_info.get("val"))

        # 收集预测和真实标签
        y_true = []
        y_pred = []
        y_probs = []

        # 使用模型预测测试集
        import os

        for class_idx, class_name in enumerate(sorted(os.listdir(test_path))):
            class_dir = Path(test_path) / class_name
            if not class_dir.is_dir():
                continue

            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                    pred_results = model.predict(str(img_path), verbose=False)
                    if pred_results and pred_results[0].probs is not None:
                        probs = pred_results[0].probs.data.cpu().numpy()
                        pred_class = probs.argmax()

                        y_true.append(class_idx)
                        y_pred.append(pred_class)
                        y_probs.append(probs)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)

        # 获取类别名称
        class_names = sorted(os.listdir(test_path))
        class_names = [n for n in class_names if (Path(test_path) / n).is_dir()]
        n_classes = len(class_names)

        # 计算整体指标
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # 计算 AUC
        if n_classes == 2:
            # 二分类 AUC
            auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            # 多分类 AUC (OvR)
            try:
                auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
            except ValueError:
                auc = None

        # 计算各类别指标
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

        # 构建结果字典
        result = {
            "model_path": str(model_path),
            "accuracy": accuracy,
            "auc": auc,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "top1_accuracy": getattr(results, "top1", accuracy),
            "top5_accuracy": getattr(results, "top5", 1.0),
        }

        # 添加各类别指标
        for class_name in class_names:
            if class_name in report:
                result[f"{class_name}_precision"] = report[class_name]["precision"]
                result[f"{class_name}_recall"] = report[class_name]["recall"]
                result[f"{class_name}_f1"] = report[class_name]["f1-score"]
                result[f"{class_name}_support"] = report[class_name]["support"]

        return result

    except Exception as e:
        print(f"评估模型 {model_path} 失败: {e}")
        import traceback

        traceback.print_exc()
        return {"model_path": str(model_path), "error": str(e)}


def batch_evaluate(
    sweep_dir: str | Path,
    data_path: str,
    output_dir: str | Path = None,
    imgsz: int = 256,
    batch: int = 32,
    device: str = "0",
) -> pd.DataFrame:
    """
    批量评估指定目录下所有实验的 best.pt 模型

    Args:
        sweep_dir: 超参数搜索结果目录 (如 runs/classify/plaque_sweep_n)
        data_path: 数据集路径
        output_dir: 输出目录，默认为 sweep_dir
        imgsz: 图像尺寸
        batch: 批次大小
        device: 设备

    Returns:
        DataFrame: 包含所有模型评估结果的表格
    """
    sweep_dir = Path(sweep_dir)
    output_dir = Path(output_dir) if output_dir else sweep_dir

    # 查找所有 best.pt 文件
    model_paths = list(sweep_dir.glob("*/weights/best.pt"))
    print(f"找到 {len(model_paths)} 个模型待评估")

    if not model_paths:
        print(f"在 {sweep_dir} 下未找到任何 best.pt 文件")
        return pd.DataFrame()

    results = []

    for idx, model_path in enumerate(model_paths):
        exp_name = model_path.parent.parent.name
        print(f"\n[{idx + 1}/{len(model_paths)}] 评估: {exp_name}")

        # 评估模型
        result = evaluate_single_model(
            model_path=model_path,
            data_path=data_path,
            imgsz=imgsz,
            batch=batch,
            device=device,
        )
        result["experiment"] = exp_name

        # 解析超参数名称
        # 格式: lr00_dropout0_weight_decay0.001
        parts = exp_name.split("_")
        for part in parts:
            if part.startswith("lr0"):
                result["lr0"] = float(part.replace("lr0", "") or "0")
            elif part.startswith("dropout"):
                result["dropout"] = float(part.replace("dropout", ""))
            elif part.startswith("decay"):
                result["weight_decay"] = float(part.replace("decay", ""))
            elif "weight" in exp_name and "decay" in part:
                # 处理 weight_decay 格式
                pass

        # 从实验名称提取 weight_decay
        if "weight_decay" in exp_name:
            wd_part = exp_name.split("weight_decay")[-1]
            try:
                result["weight_decay"] = float(wd_part)
            except ValueError:
                pass

        results.append(result)

        # 保存单个模型的详细结果
        exp_result_path = model_path.parent.parent / "test_evaluation.txt"
        with open(exp_result_path, "w", encoding="utf-8") as f:
            f.write(f"实验名称: {exp_name}\n")
            f.write(f"模型路径: {model_path}\n")
            f.write("=" * 60 + "\n")
            for key, value in result.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        print(f"  结果已保存至: {exp_result_path}")

    # 创建 DataFrame
    df = pd.DataFrame(results)

    # 重新排列列顺序
    priority_cols = [
        "experiment",
        "lr0",
        "dropout",
        "weight_decay",
        "accuracy",
        "auc",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "top1_accuracy",
        "top5_accuracy",
    ]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[[c for c in priority_cols if c in df.columns] + other_cols]

    # 保存汇总结果
    summary_csv = output_dir / "test_evaluation_summary.csv"
    df.to_csv(summary_csv, index=False, encoding="utf-8")
    print(f"\n汇总结果已保存至: {summary_csv}")

    # 找出最优模型
    if "auc" in df.columns and df["auc"].notna().any():
        best_by_auc = df.loc[df["auc"].idxmax()]
        print("\n" + "=" * 60)
        print("🏆 最优模型 (按 AUC):")
        print(f"  实验: {best_by_auc['experiment']}")
        print(f"  AUC: {best_by_auc['auc']:.6f}")
        if "f1_macro" in best_by_auc:
            print(f"  F1-Macro: {best_by_auc['f1_macro']:.6f}")

    if "f1_macro" in df.columns:
        best_by_f1 = df.loc[df["f1_macro"].idxmax()]
        print("\n🏆 最优模型 (按 F1-Macro):")
        print(f"  实验: {best_by_f1['experiment']}")
        print(f"  F1-Macro: {best_by_f1['f1_macro']:.6f}")
        if "auc" in best_by_f1 and best_by_f1["auc"] is not None:
            print(f"  AUC: {best_by_f1['auc']:.6f}")

    if "accuracy" in df.columns:
        best_by_acc = df.loc[df["accuracy"].idxmax()]
        print("\n🏆 最优模型 (按 Accuracy):")
        print(f"  实验: {best_by_acc['experiment']}")
        print(f"  Accuracy: {best_by_acc['accuracy']:.6f}")

    # 保存最优模型信息
    best_model_path = output_dir / "best_models.txt"
    with open(best_model_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("最优模型汇总\n")
        f.write("=" * 60 + "\n\n")

        if "auc" in df.columns and df["auc"].notna().any():
            best = df.loc[df["auc"].idxmax()]
            f.write("【最优 AUC】\n")
            for key, value in best.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

        if "f1_macro" in df.columns:
            best = df.loc[df["f1_macro"].idxmax()]
            f.write("【最优 F1-Macro】\n")
            for key, value in best.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

        if "accuracy" in df.columns:
            best = df.loc[df["accuracy"].idxmax()]
            f.write("【最优 Accuracy】\n")
            for key, value in best.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")

    print(f"\n最优模型信息已保存至: {best_model_path}")

    return df


def evaluate_all_sweeps(
    base_dir: str = "runs/classify",
    data_path: str = "/data/users/lxing/File/medicalImg/CAS/selected_folders/plaque/dataset",
    sweep_patterns: list = None,
    **kwargs,
):
    """
    评估所有超参数搜索目录

    Args:
        base_dir: 基础目录
        data_path: 数据集路径
        sweep_patterns: 要评估的目录模式列表
        **kwargs: 传递给 batch_evaluate 的其他参数
    """
    base_dir = Path(base_dir)

    if sweep_patterns is None:
        sweep_patterns = ["plaque_sweep*"]

    all_results = []

    for pattern in sweep_patterns:
        for sweep_dir in base_dir.glob(pattern):
            if sweep_dir.is_dir():
                print("\n" + "=" * 80)
                print(f"评估目录: {sweep_dir}")
                print("=" * 80)

                df = batch_evaluate(sweep_dir=sweep_dir, data_path=data_path, **kwargs)

                if not df.empty:
                    df["sweep_dir"] = sweep_dir.name
                    all_results.append(df)

    if all_results:
        # 合并所有结果
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_csv = base_dir / "all_sweeps_evaluation.csv"
        combined_df.to_csv(combined_csv, index=False, encoding="utf-8")
        print(f"\n\n所有超参数搜索的评估结果已保存至: {combined_csv}")

        # 找出全局最优
        print("\n" + "=" * 80)
        print("🌟 全局最优模型")
        print("=" * 80)

        if "auc" in combined_df.columns and combined_df["auc"].notna().any():
            best = combined_df.loc[combined_df["auc"].idxmax()]
            print(f"\n【全局最优 AUC】")
            print(f"  目录: {best.get('sweep_dir', 'N/A')}")
            print(f"  实验: {best['experiment']}")
            print(f"  AUC: {best['auc']:.6f}")

        if "f1_macro" in combined_df.columns:
            best = combined_df.loc[combined_df["f1_macro"].idxmax()]
            print(f"\n【全局最优 F1-Macro】")
            print(f"  目录: {best.get('sweep_dir', 'N/A')}")
            print(f"  实验: {best['experiment']}")
            print(f"  F1-Macro: {best['f1_macro']:.6f}")

        return combined_df

    return pd.DataFrame()


if __name__ == "__main__":
    # 配置参数
    DATA_PATH = "/data/users/lxing/File/medicalImg/CAS/selected_folders/plaque/dataset"

    # 方式1: 评估单个超参数搜索目录
    # results = batch_evaluate(
    #     sweep_dir="runs/classify/plaque_sweep_n",
    #     data_path=DATA_PATH,
    #     imgsz=256,
    #     batch=32,
    #     device="0",
    # )

    # 方式2: 评估所有超参数搜索目录
    results = evaluate_all_sweeps(
        base_dir="runs/classify",
        data_path=DATA_PATH,
        sweep_patterns=["plaque_sweep_n", "plaque_sweep_m", "plaque_sweep"],
        imgsz=256,
        batch=32,
        device="0",
    )

    # 打印结果摘要
    if not results.empty:
        print("\n" + "=" * 80)
        print("评估完成！结果摘要:")
        print("=" * 80)

        # 按 AUC 排序显示前 10
        if "auc" in results.columns:
            print("\nTop 10 模型 (按 AUC 排序):")
            top10 = results.nlargest(10, "auc")[
                ["experiment", "sweep_dir", "auc", "f1_macro", "accuracy"]
            ]
            print(top10.to_string(index=False))