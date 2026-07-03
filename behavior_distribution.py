"""
Behavior Distribution Analysis: 统计和分析Behavior分布，生成PR-AUC曲线数据
"""
import json
import os
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

from prompt_builder import get_binary_questions_for_action, get_all_questions_for_action


def clean_for_json(obj):
    """
    递归清理数据，将数值转换为 JSON 兼容格式
    处理: float, numpy.floating, numpy.integer, numpy.bool_, NaN, Inf, ndarray
    """
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)  # numpy.bool_ -> Python bool (必须在 float 之前检查)
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    else:
        return obj


def analyze_test_data_distribution(experiment_data: Dict, output_path: str = None) -> Dict:
    """
    分析测试数据中各个Behavior的分布
    
    Args:
        experiment_data: 从prepare_experiment_data生成的实验数据
        output_path: 输出路径（可选），如果提供则保存到文件
    
    Returns:
        分布统计字典
    """
    print("\n" + "=" * 80)
    print("分析测试数据中Behavior的分布")
    print("=" * 80)
    
    # 收集所有字段的真实值
    field_values = defaultdict(list)  # field -> [values]
    field_types = {}  # field -> type
    
    for user in experiment_data.get("users", []):
        for test_action in user.get("test_actions", []):
            action = test_action.get("action")
            questions = get_all_questions_for_action(action)  # 获取所有类型问题
            
            for question in questions:
                field = question["field"]
                field_type = question["type"]
                true_value = question["true_value"]
                
                field_types[field] = field_type
                field_values[field].append(true_value)
    
    # 统计分布
    distribution_stats = {
        "metadata": {
            "total_users": len(experiment_data.get("users", [])),
            "total_samples": sum(len(v) for v in field_values.values()),
        },
        "fields": {}
    }
    
    # 按类型分组统计
    binary_fields = {}
    continuous_fields = {}
    text_fields = {}
    
    for field, values in field_values.items():
        field_type = field_types[field]
        
        if field_type == "binary":
            # 二分类字段统计
            values_int = [int(v) for v in values]
            positive_count = sum(values_int)
            negative_count = len(values_int) - positive_count
            positive_rate = positive_count / len(values_int) if values_int else 0
            
            binary_fields[field] = {
                "type": "binary",
                "total_samples": len(values_int),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_rate": positive_rate,
                "distribution": {
                    "0": negative_count,
                    "1": positive_count
                }
            }
            
        elif field_type == "continuous":
            # 连续值字段统计
            values_float = [float(v) for v in values]
            continuous_fields[field] = {
                "type": "continuous",
                "total_samples": len(values_float),
                "mean": np.mean(values_float),
                "std": np.std(values_float),
                "min": np.min(values_float),
                "max": np.max(values_float),
                "median": np.median(values_float),
                "percentiles": {
                    "25%": np.percentile(values_float, 25),
                    "50%": np.percentile(values_float, 50),
                    "75%": np.percentile(values_float, 75),
                }
            }
            
        elif field_type == "text":
            # 文本字段统计
            text_fields[field] = {
                "type": "text",
                "total_samples": len(values),
                "avg_length": np.mean([len(str(v)) for v in values]),
                "unique_count": len(set(values))
            }
    
    distribution_stats["fields"]["binary"] = binary_fields
    distribution_stats["fields"]["continuous"] = continuous_fields
    distribution_stats["fields"]["text"] = text_fields
    
    # 打印统计信息
    print(f"\n总样本数: {distribution_stats['metadata']['total_samples']}")
    print(f"总用户数: {distribution_stats['metadata']['total_users']}")
    
    if binary_fields:
        print(f"\n二分类字段分布 (共 {len(binary_fields)} 个字段):")
        print("-" * 80)
        print(f"{'字段名称':<30s} {'总样本':<10s} {'正样本':<10s} {'负样本':<10s} {'正样本率':<10s}")
        print("-" * 80)
        for field, stats in sorted(binary_fields.items()):
            print(f"{field:<30s} {stats['total_samples']:<10d} "
                  f"{stats['positive_count']:<10d} {stats['negative_count']:<10d} "
                  f"{stats['positive_rate']:<10.2%}")
    
    if continuous_fields:
        print(f"\n连续值字段分布 (共 {len(continuous_fields)} 个字段):")
        print("-" * 80)
        print(f"{'字段名称':<30s} {'总样本':<10s} {'均值':<12s} {'标准差':<12s} {'中位数':<12s}")
        print("-" * 80)
        for field, stats in sorted(continuous_fields.items()):
            print(f"{field:<30s} {stats['total_samples']:<10d} "
                  f"{stats['mean']:<12.2f} {stats['std']:<12.2f} {stats['median']:<12.2f}")
    
    if text_fields:
        print(f"\n文本字段分布 (共 {len(text_fields)} 个字段):")
        print("-" * 80)
        print(f"{'字段名称':<30s} {'总样本':<10s} {'平均长度':<12s} {'唯一值数':<12s}")
        print("-" * 80)
        for field, stats in sorted(text_fields.items()):
            print(f"{field:<30s} {stats['total_samples']:<10d} "
                  f"{stats['avg_length']:<12.1f} {stats['unique_count']:<12d}")
    
    # 保存到文件
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_for_json(distribution_stats), f, ensure_ascii=False, indent=2)
        print(f"\n测试数据分布统计已保存到: {output_path}")
    
    return distribution_stats


def analyze_prediction_distribution(prediction_results: List[Dict], output_path: str = None) -> Dict:
    """
    分析预测结果中各个Behavior的分布
    
    Args:
        prediction_results: 从evaluator生成的预测结果
        output_path: 输出路径（可选），如果提供则保存到文件
    
    Returns:
        预测分布统计字典
    """
    print("\n" + "=" * 80)
    print("分析预测结果中Behavior的分布")
    print("=" * 80)
    
    # 收集所有字段的真实值和预测值
    field_data = defaultdict(lambda: {"y_true": [], "y_pred": [], "type": None})
    
    for result in prediction_results:
        if not result.get("success"):
            continue
            
        for q in result.get("questions", []):
            field = q["field"]
            field_type = q["type"]
            true_val = q["true_value"]
            pred_val = q["predicted_value"]
            
            # 如果预测值为None，跳过
            if pred_val is None:
                continue
            
            field_data[field]["type"] = field_type
            field_data[field]["y_true"].append(true_val)
            field_data[field]["y_pred"].append(pred_val)
    
    # 统计分布
    distribution_stats = {
        "metadata": {
            "total_predictions": len(prediction_results),
            "successful_predictions": sum(1 for r in prediction_results if r.get("success")),
        },
        "fields": {}
    }
    
    # 按类型分组统计
    binary_fields = {}
    continuous_fields = {}
    text_fields = {}
    
    for field, data in field_data.items():
        field_type = data["type"]
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        
        if not y_true or not y_pred:
            continue
        
        if field_type == "binary":
            # 二分类字段统计
            y_true_int = [int(v) for v in y_true]
            y_pred_float = [float(v) for v in y_pred]
            
            # 真实值分布
            true_positive_count = sum(y_true_int)
            true_negative_count = len(y_true_int) - true_positive_count
            true_positive_rate = true_positive_count / len(y_true_int)
            
            # 预测值分布（概率）
            pred_mean = np.mean(y_pred_float)
            pred_std = np.std(y_pred_float)
            
            # 预测标签分布（阈值0.5）
            y_pred_labels = [1 if p >= 0.5 else 0 for p in y_pred_float]
            pred_positive_count = sum(y_pred_labels)
            pred_negative_count = len(y_pred_labels) - pred_positive_count
            pred_positive_rate = pred_positive_count / len(y_pred_labels)
            
            binary_fields[field] = {
                "type": "binary",
                "total_samples": len(y_true_int),
                "true_distribution": {
                    "positive_count": true_positive_count,
                    "negative_count": true_negative_count,
                    "positive_rate": true_positive_rate,
                },
                "predicted_probability": {
                    "mean": pred_mean,
                    "std": pred_std,
                    "min": np.min(y_pred_float),
                    "max": np.max(y_pred_float),
                    "median": np.median(y_pred_float),
                },
                "predicted_labels": {
                    "positive_count": pred_positive_count,
                    "negative_count": pred_negative_count,
                    "positive_rate": pred_positive_rate,
                }
            }
            
        elif field_type == "continuous":
            # 连续值字段统计
            y_true_float = [float(v) for v in y_true]
            y_pred_float = [float(v) for v in y_pred]
            
            continuous_fields[field] = {
                "type": "continuous",
                "total_samples": len(y_true_float),
                "true_distribution": {
                    "mean": np.mean(y_true_float),
                    "std": np.std(y_true_float),
                    "min": np.min(y_true_float),
                    "max": np.max(y_true_float),
                    "median": np.median(y_true_float),
                },
                "predicted_distribution": {
                    "mean": np.mean(y_pred_float),
                    "std": np.std(y_pred_float),
                    "min": np.min(y_pred_float),
                    "max": np.max(y_pred_float),
                    "median": np.median(y_pred_float),
                }
            }
            
        elif field_type == "text":
            # 文本字段统计
            text_fields[field] = {
                "type": "text",
                "total_samples": len(y_true),
                "true_avg_length": np.mean([len(str(v)) for v in y_true]),
                "pred_avg_length": np.mean([len(str(v)) for v in y_pred]),
            }
    
    distribution_stats["fields"]["binary"] = binary_fields
    distribution_stats["fields"]["continuous"] = continuous_fields
    distribution_stats["fields"]["text"] = text_fields
    
    # 打印统计信息
    print(f"\n总预测数: {distribution_stats['metadata']['total_predictions']}")
    print(f"成功预测数: {distribution_stats['metadata']['successful_predictions']}")
    
    if binary_fields:
        print(f"\n二分类字段预测分布 (共 {len(binary_fields)} 个字段):")
        print("-" * 100)
        print(f"{'字段名称':<25s} {'样本数':<8s} {'真实正样本率':<14s} "
              f"{'预测正样本率':<14s} {'预测概率均值':<14s}")
        print("-" * 100)
        for field, stats in sorted(binary_fields.items()):
            print(f"{field:<25s} {stats['total_samples']:<8d} "
                  f"{stats['true_distribution']['positive_rate']:<14.2%} "
                  f"{stats['predicted_labels']['positive_rate']:<14.2%} "
                  f"{stats['predicted_probability']['mean']:<14.4f}")
    
    if continuous_fields:
        print(f"\n连续值字段预测分布 (共 {len(continuous_fields)} 个字段):")
        print("-" * 100)
        print(f"{'字段名称':<25s} {'样本数':<8s} {'真实均值':<12s} "
              f"{'预测均值':<12s} {'真实标准差':<12s} {'预测标准差':<12s}")
        print("-" * 100)
        for field, stats in sorted(continuous_fields.items()):
            print(f"{field:<25s} {stats['total_samples']:<8d} "
                  f"{stats['true_distribution']['mean']:<12.2f} "
                  f"{stats['predicted_distribution']['mean']:<12.2f} "
                  f"{stats['true_distribution']['std']:<12.2f} "
                  f"{stats['predicted_distribution']['std']:<12.2f}")
    
    # 保存到文件
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_for_json(distribution_stats), f, ensure_ascii=False, indent=2)
        print(f"\n预测结果分布统计已保存到: {output_path}")
    
    return distribution_stats


def generate_pr_curve_data(prediction_results: List[Dict], output_path: str = None) -> Dict:
    """
    为二分类字段生成PR曲线数据（用于绘制PR-AUC曲线）
    
    Args:
        prediction_results: 从evaluator生成的预测结果
        output_path: 输出路径（可选），如果提供则保存到文件
    
    Returns:
        PR曲线数据字典
    """
    print("\n" + "=" * 80)
    print("生成PR曲线数据")
    print("=" * 80)
    
    # 收集二分类字段的真实值和预测概率
    binary_data = defaultdict(lambda: {"y_true": [], "y_pred": []})
    
    for result in prediction_results:
        if not result.get("success"):
            continue
            
        for q in result.get("questions", []):
            if q["type"] != "binary":
                continue
                
            field = q["field"]
            true_val = q["true_value"]
            pred_val = q["predicted_value"]
            
            # 如果预测值为None，跳过
            if pred_val is None:
                continue
            
            try:
                binary_data[field]["y_true"].append(int(true_val))
                binary_data[field]["y_pred"].append(float(pred_val))
            except (ValueError, TypeError):
                continue
    
    # 为每个二分类字段生成PR曲线数据
    pr_curve_data = {
        "metadata": {
            "total_fields": len(binary_data),
            "description": "Precision-Recall curve data for binary classification tasks"
        },
        "fields": {}
    }
    
    for field, data in binary_data.items():
        y_true = np.array(data["y_true"])
        y_pred = np.array(data["y_pred"])
        
        if len(y_true) < 2:
            print(f"  跳过 {field}: 样本数不足 (< 2)")
            continue
        
        # 检查是否只有一个类别
        if len(np.unique(y_true)) < 2:
            print(f"  跳过 {field}: 只有一个类别")
            continue
        
        try:
            # 计算PR曲线
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            
            # 计算AUC
            pr_auc = auc(recall, precision)
            
            # 生成曲线数据点（转换为Python原生类型以便JSON序列化）
            pr_curve_data["fields"][field] = {
                "total_samples": int(len(y_true)),
                "positive_samples": int(np.sum(y_true)),
                "negative_samples": int(len(y_true) - np.sum(y_true)),
                "positive_rate": float(np.mean(y_true)),
                "pr_auc": float(pr_auc),
                "curve_points": {
                    "precision": [float(p) for p in precision],
                    "recall": [float(r) for r in recall],
                    "thresholds": [float(t) for t in thresholds] + [1.0],  # 添加最后一个阈值
                },
                "key_metrics": {
                    "max_f1_score": float(np.max(2 * (precision * recall) / (precision + recall + 1e-10))),
                    "precision_at_50_recall": float(precision[np.argmin(np.abs(recall - 0.5))]) if len(recall) > 0 else 0.0,
                    "recall_at_50_precision": float(recall[np.argmin(np.abs(precision - 0.5))]) if len(precision) > 0 else 0.0,
                }
            }
            
            print(f"  ✓ {field:<30s}: PR-AUC = {pr_auc:.4f}, "
                  f"样本数 = {len(y_true)}, 正样本率 = {np.mean(y_true):.2%}")
            
        except Exception as e:
            print(f"  ✗ {field}: 生成PR曲线失败 - {e}")
            continue
    
    # 保存到文件
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_for_json(pr_curve_data), f, ensure_ascii=False, indent=2)
        print(f"\nPR曲线数据已保存到: {output_path}")
        print(f"  包含 {len(pr_curve_data['fields'])} 个二分类字段的PR曲线数据")
    
    return pr_curve_data


def generate_distribution_comparison(
    test_distribution: Dict,
    pred_distribution: Dict,
    output_path: str = None
) -> Dict:
    """
    生成测试数据和预测结果的分布对比
    
    Args:
        test_distribution: 测试数据分布统计
        pred_distribution: 预测结果分布统计
        output_path: 输出路径（可选）
    
    Returns:
        对比数据字典
    """
    print("\n" + "=" * 80)
    print("生成分布对比数据")
    print("=" * 80)
    
    comparison = {
        "metadata": {
            "test_total_samples": test_distribution["metadata"]["total_samples"],
            "pred_total_samples": pred_distribution["metadata"]["total_predictions"],
        },
        "binary_fields": {},
        "continuous_fields": {},
    }
    
    # 对比二分类字段
    test_binary = test_distribution["fields"].get("binary", {})
    pred_binary = pred_distribution["fields"].get("binary", {})
    
    for field in test_binary.keys():
        if field not in pred_binary:
            continue
        
        test_stats = test_binary[field]
        pred_stats = pred_binary[field]
        
        comparison["binary_fields"][field] = {
            "test": {
                "total_samples": test_stats["total_samples"],
                "positive_rate": test_stats["positive_rate"],
                "positive_count": test_stats["positive_count"],
                "negative_count": test_stats["negative_count"],
            },
            "prediction": {
                "total_samples": pred_stats["total_samples"],
                "true_positive_rate": pred_stats["true_distribution"]["positive_rate"],
                "predicted_positive_rate": pred_stats["predicted_labels"]["positive_rate"],
                "predicted_prob_mean": pred_stats["predicted_probability"]["mean"],
            },
            "difference": {
                "sample_count_diff": pred_stats["total_samples"] - test_stats["total_samples"],
                "positive_rate_diff": pred_stats["predicted_labels"]["positive_rate"] - test_stats["positive_rate"],
            }
        }
    
    # 对比连续值字段
    test_continuous = test_distribution["fields"].get("continuous", {})
    pred_continuous = pred_distribution["fields"].get("continuous", {})
    
    for field in test_continuous.keys():
        if field not in pred_continuous:
            continue
        
        test_stats = test_continuous[field]
        pred_stats = pred_continuous[field]
        
        comparison["continuous_fields"][field] = {
            "test": {
                "total_samples": test_stats["total_samples"],
                "mean": test_stats["mean"],
                "std": test_stats["std"],
            },
            "prediction": {
                "total_samples": pred_stats["total_samples"],
                "true_mean": pred_stats["true_distribution"]["mean"],
                "predicted_mean": pred_stats["predicted_distribution"]["mean"],
            },
            "difference": {
                "mean_diff": pred_stats["predicted_distribution"]["mean"] - test_stats["mean"],
            }
        }
    
    # 打印对比信息
    if comparison["binary_fields"]:
        print(f"\n二分类字段分布对比:")
        print("-" * 100)
        print(f"{'字段名称':<25s} {'测试正样本率':<14s} {'预测正样本率':<14s} {'差异':<12s}")
        print("-" * 100)
        for field, comp in sorted(comparison["binary_fields"].items()):
            print(f"{field:<25s} "
                  f"{comp['test']['positive_rate']:<14.2%} "
                  f"{comp['prediction']['predicted_positive_rate']:<14.2%} "
                  f"{comp['difference']['positive_rate_diff']:>+12.4f}")
    
    # 保存到文件
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_for_json(comparison), f, ensure_ascii=False, indent=2)
        print(f"\n分布对比数据已保存到: {output_path}")
    
    return comparison
