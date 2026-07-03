"""
Metrics: Calculate evaluation metrics for binary, continuous, and text predictions
"""
import numpy as np
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from typing import List, Tuple, Dict
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

# BERTScore 支持（使用 chinese-roberta-wwm-ext-large）
try:
    from bert_score import score as bert_score_func
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    bert_score_func = None

# BERTScore 配置
BERTSCORE_MODEL = "/path/to/chinese-roberta-wwm-ext-large"  # 中文 RoBERTa 大模型
BERTSCORE_BATCH_SIZE = 8  # 降低批次大小以避免多GPU环境下的CUDA错误
BERTSCORE_MAX_LENGTH = 512  # 最大文本长度（截断过长文本）


def calculate_accuracy(y_true: List[int], y_pred_labels: List[int], field_name: str = "") -> float:
    """
    计算准确率 (Accuracy)
    
    Args:
        y_true: 真实标签 (0 or 1)
        y_pred_labels: 预测标签 (0 or 1)，基于模型输出YES/NO直接判定
        field_name: 字段名称（用于调试信息）
    
    Returns:
        Accuracy score (0 to 1)
    """
    try:
        # 计算准确率
        correct = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == pred)
        accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0
        return accuracy
    except Exception as e:
        print(f"计算Accuracy时出错 ({field_name}): {e}")
        return np.nan


def calculate_logloss(y_true: List[int], y_pred: List[float]) -> float:
    """
    计算LogLoss (Cross-Entropy Loss)
    
    Args:
        y_true: 真实标签 (0 or 1)
        y_pred: 预测概率 (0 to 1)
    
    Returns:
        LogLoss value (lower is better)
    """
    try:
        # 将概率限制在[1e-7, 1-1e-7]范围内，避免log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # 明确指定labels参数，避免只有单一类别时报错
        return log_loss(y_true, y_pred_clipped, labels=[0, 1])
    except Exception as e:
        print(f"计算LogLoss时出错: {e}")
        return np.nan


def calculate_precision(y_true: List[int], y_pred_labels: List[int]) -> float:
    """
    计算精确率 (Precision)
    
    Precision = TP / (TP + FP)
    
    Args:
        y_true: 真实标签 (0 or 1)
        y_pred_labels: 预测标签 (0 or 1)，基于模型输出YES/NO直接判定
    
    Returns:
        Precision score (0 to 1)
    """
    try:
        # 计算 TP 和 FP
        tp = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 1 and pred == 1)
        fp = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 0 and pred == 1)
        
        # 计算 Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return precision
    except Exception as e:
        print(f"计算Precision时出错: {e}")
        return np.nan


def calculate_recall(y_true: List[int], y_pred_labels: List[int]) -> float:
    """
    计算召回率 (Recall)
    
    Recall = TP / (TP + FN)
    
    Args:
        y_true: 真实标签 (0 or 1)
        y_pred_labels: 预测标签 (0 or 1)，基于模型输出YES/NO直接判定
    
    Returns:
        Recall score (0 to 1)
    """
    try:
        # 计算 TP 和 FN
        tp = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 1 and pred == 1)
        fn = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 1 and pred == 0)
        
        # 计算 Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return recall
    except Exception as e:
        print(f"计算Recall时出错: {e}")
        return np.nan


def calculate_f1(y_true: List[int], y_pred_labels: List[int]) -> float:
    """
    计算F1分数
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        y_true: 真实标签 (0 or 1)
        y_pred_labels: 预测标签 (0 or 1)，基于模型输出YES/NO直接判定
    
    Returns:
        F1 score (0 to 1)
    """
    try:
        precision = calculate_precision(y_true, y_pred_labels)
        recall = calculate_recall(y_true, y_pred_labels)
        
        # 计算 F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return f1
    except Exception as e:
        print(f"计算F1时出错: {e}")
        return np.nan


def calculate_ece(y_true: List[int], y_pred: List[float], n_bins: int = 10) -> float:
    """
    计算ECE (Expected Calibration Error)
    
    衡量预测概率的校准程度
    
    Args:
        y_true: 真实标签 (0 or 1)
        y_pred: 预测概率 (0 to 1)
        n_bins: 区间数量
    
    Returns:
        ECE value (0 to 1, lower is better)
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 创建区间
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        for i in range(n_bins):
            # 找到落在当前区间的样本
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
            
            if np.sum(in_bin) > 0:
                # 计算该区间的平均置信度和准确率
                bin_confidence = np.mean(y_pred[in_bin])
                bin_accuracy = np.mean(y_true[in_bin])
                
                # 加权到ECE
                bin_weight = np.sum(in_bin) / len(y_true)
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
        return ece
    except Exception as e:
        print(f"计算ECE时出错: {e}")
        return np.nan


def calculate_auc(y_true: List[int], y_pred: List[float]) -> float:
    """
    计算AUC (Area Under ROC Curve)
    
    衡量分类器对正负样本的区分能力，不依赖于特定的分类阈值
    
    Args:
        y_true: 真实标签 (0 or 1)
        y_pred: 预测概率 (0 to 1)
    
    Returns:
        AUC value (0 to 1, higher is better)
        0.5表示随机分类，1.0表示完美分类
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 检查是否只有一个类别
        if len(np.unique(y_true)) < 2:
            print(f"警告: 样本中只有一个类别，无法计算AUC")
            return np.nan
        
        # 计算AUC
        return roc_auc_score(y_true, y_pred)
    except Exception as e:
        print(f"计算AUC时出错: {e}")
        return np.nan


def calculate_mae(y_true: List[float], y_pred: List[float]) -> float:
    """
    计算MAE (Mean Absolute Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        MAE value
    """
    try:
        return mean_absolute_error(y_true, y_pred)
    except Exception as e:
        print(f"计算MAE时出错: {e}")
        return np.nan


def calculate_mse(y_true: List[float], y_pred: List[float]) -> float:
    """
    计算MSE (Mean Squared Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        MSE value
    """
    try:
        return mean_squared_error(y_true, y_pred)
    except Exception as e:
        print(f"计算MSE时出错: {e}")
        return np.nan


def calculate_rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    计算RMSE (Root Mean Squared Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        RMSE value
    """
    try:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    except Exception as e:
        print(f"计算RMSE时出错: {e}")
        return np.nan


def calculate_nmae(y_true: List[float], y_pred: List[float], 
                   normalizers: List[float]) -> float:
    """
    计算NMAE (Normalized Mean Absolute Error)
    
    将每个样本的绝对误差除以其归一化因子（如视频时长），然后取平均，
    最终乘以100表示为百分比形式。
    
    NMAE = (1/n) * Σ(clip(|y_true_i - y_pred_i|, 0, normalizer_i) / normalizer_i) * 100
    
    Args:
        y_true: 真实值列表
        y_pred: 预测值列表
        normalizers: 归一化因子列表（如视频时长），用于将误差归一化到0-1范围
    
    Returns:
        NMAE value (0 to 100, lower is better)
        如果归一化因子为0或None，该样本将被跳过
    """
    try:
        if len(y_true) != len(y_pred) or len(y_true) != len(normalizers):
            print(f"计算NMAE时出错: 输入列表长度不一致 (y_true={len(y_true)}, y_pred={len(y_pred)}, normalizers={len(normalizers)})")
            return np.nan
        
        normalized_errors = []
        skipped_count = 0
        for true_val, pred_val, norm in zip(y_true, y_pred, normalizers):
            # 跳过归一化因子为0或None的样本
            if norm is None or norm <= 0:
                skipped_count += 1
                continue
            
            # 计算绝对误差，并clip到[0, norm]范围内
            # abs_error = abs(true_val - pred_val)
            # clipped_error = abs(min(max(true_val, 0), norm) - min(max(pred_val, 0), norm))
            # clipped_error = min(max(abs_error, 0), norm)  # clip to [0, norm]

            # 计算绝对误差，并clip到[0, norm]范围内
            abs_error = abs(true_val - pred_val)
            clipped_error = min(max(abs_error, 0), norm)  # clip to [0, norm]
            
            # 计算归一化误差（结果在0-1范围内）
            normalized_error = clipped_error / norm
            normalized_errors.append(normalized_error)
        
        if not normalized_errors:
            print(f"警告: 没有有效的归一化因子，无法计算NMAE（跳过了 {skipped_count} 个样本）")
            return np.nan
        
        # 计算平均归一化误差并转换为百分比
        nmae = np.mean(normalized_errors) * 100
        return nmae
    except Exception as e:
        print(f"计算NMAE时出错: {e}")
        return np.nan


def calculate_all_binary_metrics(
    y_true: List[int], 
    y_pred_labels: List[int],
    y_pred_probs: List[float],
    n_bins: int = 10,
    field_name: str = ""
) -> Dict[str, float]:
    """
    计算所有二分类指标
    
    Args:
        y_true: 真实标签
        y_pred_labels: 预测标签 (0 or 1)，基于模型输出YES/NO直接判定
        y_pred_probs: 预测概率 (0 to 1)，用于计算LogLoss、AUC、ECE
        n_bins: ECE计算的区间数
        field_name: 字段名称（用于调试）
    
    Returns:
        Dict containing Accuracy, Precision, Recall, F1, LogLoss, AUC, ECE, sample_count, positive_rate
        以及 TP, FP, FN 用于后续计算 Micro F1
    """
    # 计算 TP, FP, FN 用于 Micro F1 汇总
    tp = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == 1 and pred == 0)
    
    return {
        "Accuracy": calculate_accuracy(y_true, y_pred_labels, field_name=field_name),
        "Precision": calculate_precision(y_true, y_pred_labels),
        "Recall": calculate_recall(y_true, y_pred_labels),
        "F1": calculate_f1(y_true, y_pred_labels),
        "LogLoss": calculate_logloss(y_true, y_pred_probs),
        "AUC": calculate_auc(y_true, y_pred_probs),
        "ECE": calculate_ece(y_true, y_pred_probs, n_bins),
        "sample_count": len(y_true),
        "positive_rate": np.mean(y_true),
        # 用于计算 Micro F1 的中间值
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }



def calculate_micro_macro_f1(binary_metrics: Dict[str, Dict]) -> Dict[str, float]:
    """
    计算 Micro 和 Macro F1。
    
    注意：对于 Macro F1，如果输入字典中的 F1 为 None/NaN，
    本函数默认将其视为 0.0 参与平均（更严格的评价标准）。
    
    排除的字段: video_downloaded, ad_converted（这些字段已被剔除，不参与汇总计算）
    """
    if not binary_metrics:
        return {
            "Micro_F1": 0.0, "Macro_F1": 0.0,
            "Micro_Precision": 0.0, "Micro_Recall": 0.0,
            "Total_TP": 0, "Total_FP": 0, "Total_FN": 0,
        }
    
    # 排除不参与汇总计算的字段
    EXCLUDED_FIELDS = {"video_downloaded", "ad_converted"}
    filtered_metrics = {k: v for k, v in binary_metrics.items() if k not in EXCLUDED_FIELDS}
    
    if not filtered_metrics:
        return {
            "Micro_F1": 0.0, "Macro_F1": 0.0,
            "Micro_Precision": 0.0, "Micro_Recall": 0.0,
            "Total_TP": 0, "Total_FP": 0, "Total_FN": 0,
        }
    
    # --- 1. Micro F1 计算 ---
    # 使用 sum 生成器效率较高，适合处理大规模 Dict
    total_tp = sum(m.get("TP", 0) for m in filtered_metrics.values())
    total_fp = sum(m.get("FP", 0) for m in filtered_metrics.values())
    total_fn = sum(m.get("FN", 0) for m in filtered_metrics.values())
    
    # 避免重复计算分母
    denom_prec = total_tp + total_fp
    denom_rec = total_tp + total_fn
    
    micro_precision = total_tp / denom_prec if denom_prec > 0 else 0.0
    micro_recall = total_tp / denom_rec if denom_rec > 0 else 0.0
    
    denom_f1 = micro_precision + micro_recall
    micro_f1 = (2 * micro_precision * micro_recall) / denom_f1 if denom_f1 > 0 else 0.0
    
    # --- 2. Macro F1 计算 ---
    # 将 NaN 视为 0.0 (np.nan_to_num)
    raw_f1_values = []
    for m in filtered_metrics.values():
        val = m.get("F1", 0.0)  # 如果没有key，默认取0
        if val is None:
            val = 0.0
        raw_f1_values.append(val)
    
    # 将列表转为 numpy array 以处理 nan
    f1_array = np.array(raw_f1_values, dtype=float)
    
    # 策略选择：
    # 将 NaN 视为 0，分母为所有任务总数
    f1_array = np.nan_to_num(f1_array, nan=0.0) 
    macro_f1 = np.mean(f1_array) if len(f1_array) > 0 else 0.0

    return {
        "Micro_F1": micro_f1,
        "Macro_F1": macro_f1,
        "Micro_Precision": micro_precision,
        "Micro_Recall": micro_recall,
        "Total_TP": total_tp,
        "Total_FP": total_fp,
        "Total_FN": total_fn,
    }

def calculate_relative_accuracy(y_true: List[float], y_pred: List[float]) -> float:
    """
    计算相对准确率 (Relative Accuracy)
    
    score = (1 - (abs(y_true - y_pred) / max(y_pred, y_true))) * 100
    
    Args:
        y_true: 真实值列表
        y_pred: 预测值列表
    
    Returns:
        Relative Accuracy score (0 to 100, higher is better)
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        scores = []
        for t, p in zip(y_true, y_pred):
            # 处理分母为0的情况
            max_val = max(p, t)
            if max_val == 0:
                # 如果真实值和预测值都为0，认为准确率为100%
                scores.append(100.0)
            else:
                score = (1 - (abs(t - p) / max_val)) * 100
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    except Exception as e:
        print(f"计算Relative Accuracy时出错: {e}")
        return np.nan


def calculate_symmetry_consistency(y_true: List[float], y_pred: List[float]) -> float:
    """
    计算对称一致性 (Symmetry Consistency)
    
    score = (1 - (abs(y_true - y_pred) / (y_pred + y_true))) * 100
    
    Args:
        y_true: 真实值列表
        y_pred: 预测值列表
    
    Returns:
        Symmetry Consistency score (0 to 100, higher is better)
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        scores = []
        for t, p in zip(y_true, y_pred):
            # 处理分母为0的情况
            sum_val = p + t
            if sum_val == 0:
                # 如果真实值和预测值都为0，认为一致性为100%
                scores.append(100.0)
            else:
                score = (1 - (abs(t - p) / sum_val)) * 100
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    except Exception as e:
        print(f"计算Symmetry Consistency时出错: {e}")
        return np.nan


def calculate_all_continuous_metrics(
    y_true: List[float], 
    y_pred: List[float],
    normalizers: List[float] = None
) -> Dict[str, float]:
    """
    计算所有连续值指标
    
    Args:
        y_true: 真实值列表
        y_pred: 预测值列表
        normalizers: 可选的归一化因子列表（如视频时长），用于计算 NMAE
    
    Returns:
        Dict containing MAE, MSE, RMSE, relative_accuracy, symmetry_consistency 以及可选的 NMAE（如果提供了 normalizers）
    """
    result = {
        "MAE": calculate_mae(y_true, y_pred),
        "MSE": calculate_mse(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "relative_accuracy": calculate_relative_accuracy(y_true, y_pred),
        "symmetry_consistency": calculate_symmetry_consistency(y_true, y_pred),
        "sample_count": len(y_true),
        "mean_true": np.mean(y_true),
        "mean_pred": np.mean(y_pred),
    }
    
    # 如果提供了归一化因子，计算 NMAE
    if normalizers is not None and len(normalizers) > 0:
        nmae = calculate_nmae(y_true, y_pred, normalizers)
        result["NMAE"] = nmae
    
    return result


def calculate_bleu(references: List[str], hypotheses: List[str]) -> float:
    """
    计算BLEU分数（用于文本匹配）
    
    Args:
        references: 参考文本列表
        hypotheses: 预测文本列表
    
    Returns:
        平均BLEU分数 (0 to 1)
    """
    try:
        if not BLEU_AVAILABLE:
            print("警告: nltk未安装，无法计算BLEU。请运行: pip install nltk")
            return np.nan
        
        smooth = SmoothingFunction().method1
        scores = []
        
        for ref, hyp in zip(references, hypotheses):
            # 中文分词：按字符切分（对于中文更准确）
            # 如果文本主要是中文，按字符分词；如果是英文，按空格分词
            def tokenize_chinese_aware(text: str) -> list:
                """智能分词：中文按字符，英文按词"""
                # 移除标点和空格
                import re
                text = text.strip()
                
                # 检测是否主要为中文
                chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
                total_chars = len(text.replace(' ', ''))
                
                if total_chars == 0:
                    return []
                
                # 如果中文字符占比超过50%，使用字符级分词
                if chinese_chars / total_chars > 0.5:
                    # 只保留中文字符、数字、英文字母
                    tokens = [c for c in text if c.strip() and not c in '，。！？、；：""''（）【】《》,.!?;:\'"()[]<>']
                    return tokens
                else:
                    # 英文文本，按空格分词
                    return text.split()
            
            ref_tokens = tokenize_chinese_aware(ref)
            hyp_tokens = tokenize_chinese_aware(hyp)
            
            # 如果分词后为空，跳过
            if not ref_tokens or not hyp_tokens:
                scores.append(0.0)
                continue
            
            # 计算BLEU（使用更宽松的n-gram权重，对短文本更友好）
            # 对于很短的文本，使用1-gram和2-gram
            weights = (0.5, 0.5, 0, 0) if len(ref_tokens) <= 5 else (0.25, 0.25, 0.25, 0.25)
            score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smooth)
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    except Exception as e:
        print(f"计算BLEU时出错: {e}")
        return np.nan


def calculate_bertscore(references: List[str], hypotheses: List[str], 
                        model_name: str = None, batch_size: int = None,
                        use_gpu: bool = True, max_length: int = None) -> Dict[str, float]:
    """
    使用 chinese-roberta-wwm-ext-large 计算 BERTScore
    
    BERTScore 基于预训练语言模型计算文本相似度，比 BLEU 更能捕捉语义相似性。
    
    Args:
        references: 参考文本列表
        hypotheses: 预测文本列表
        model_name: 使用的模型名称或本地路径，默认 chinese-roberta-wwm-ext-large
        batch_size: 批处理大小，默认 32
        use_gpu: 是否使用 GPU，默认 True
        max_length: 最大文本长度，超过将被截断，默认 512
    
    Returns:
        Dict containing:
            - BERTScore_P: Precision 平均值
            - BERTScore_R: Recall 平均值
            - BERTScore_F1: F1 平均值（主要指标）
    """
    if not BERTSCORE_AVAILABLE:
        print("警告: bert_score 未安装，无法计算 BERTScore。请运行: pip install bert-score")
        return {
            "BERTScore_P": np.nan,
            "BERTScore_R": np.nan,
            "BERTScore_F1": np.nan,
        }
    
    if not references or not hypotheses:
        return {
            "BERTScore_P": 0.0,
            "BERTScore_R": 0.0,
            "BERTScore_F1": 0.0,
        }
    
    try:
        import torch
        from bert_score import BERTScorer
        import re
        
        # 设置默认参数
        if model_name is None:
            model_name = BERTSCORE_MODEL
        if batch_size is None:
            batch_size = BERTSCORE_BATCH_SIZE
        if max_length is None:
            max_length = BERTSCORE_MAX_LENGTH
        
        # 检测 GPU 可用性（强制使用单个GPU避免多GPU冲突）
        if use_gpu and torch.cuda.is_available():
            device = "cuda:0"  # 明确指定使用 GPU 0
            # 清理 GPU 缓存
            torch.cuda.empty_cache()
            print(f"BERTScore 使用设备: {device}")
        else:
            device = "cpu"
            if use_gpu:
                print("警告: GPU 不可用，BERTScore 将使用 CPU 计算（速度较慢）")
        
        def clean_and_truncate_text(text: str, max_len: int) -> str:
            """清理和截断文本"""
            # 移除控制字符和不可见字符
            text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
            # 规范化空白字符
            text = re.sub(r'\s+', ' ', text)
            # 去除首尾空白
            text = text.strip()
            # 截断过长文本
            if len(text) > max_len:
                text = text[:max_len]
            return text
        
        # 过滤和清理输入（更严格的验证）
        valid_pairs = []
        for ref, hyp in zip(references, hypotheses):
            # 确保是字符串类型
            if not isinstance(ref, str):
                ref = str(ref) if ref is not None else ""
            if not isinstance(hyp, str):
                hyp = str(hyp) if hyp is not None else ""
            
            # 清理和截断
            ref = clean_and_truncate_text(ref, max_length)
            hyp = clean_and_truncate_text(hyp, max_length)
            
            # 只保留非空且有效的文本对
            if ref and hyp and len(ref) > 0 and len(hyp) > 0:
                valid_pairs.append((ref, hyp))
        
        if not valid_pairs:
            print("警告: 没有有效的文本对可以计算 BERTScore")
            return {
                "BERTScore_P": 0.0,
                "BERTScore_R": 0.0,
                "BERTScore_F1": 0.0,
            }
        
        valid_refs, valid_hyps = zip(*valid_pairs)
        valid_refs, valid_hyps = list(valid_refs), list(valid_hyps)
        
        # 判断是否为本地路径（包含 / 或 \）
        import os
        is_local_path = os.path.exists(model_name) or '/' in model_name or '\\' in model_name
        
        # 分批处理以避免 CUDA 错误
        all_P, all_R, all_F1 = [], [], []
        
        # 使用更小的批次进行计算（多GPU环境下进一步降低）
        if device.startswith("cuda"):
            effective_batch_size = min(batch_size, 8)  # GPU模式：最大8
        else:
            effective_batch_size = min(batch_size, 16)  # CPU模式：最大16
        
        print(f"BERTScore 批量大小: {effective_batch_size}, 总批次: {(len(valid_refs) + effective_batch_size - 1) // effective_batch_size}")
        
        for i in range(0, len(valid_refs), effective_batch_size):
            batch_refs = valid_refs[i:i+effective_batch_size]
            batch_hyps = valid_hyps[i:i+effective_batch_size]
            
            try:
                # 在每批次前清理缓存
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                if is_local_path:
                    # 本地模型：使用 BERTScorer 类直接加载
                    scorer = BERTScorer(
                        model_type=model_name,
                        num_layers=24,  # chinese-roberta-wwm-ext-large 有 24 层
                        batch_size=effective_batch_size,
                        device=device,
                        rescale_with_baseline=False,
                        lang="zh",
                        idf=False,  # 不使用 IDF 权重，避免额外计算
                    )
                    P, R, F1 = scorer.score(cands=batch_hyps, refs=batch_refs)
                else:
                    # 使用预定义的模型名称
                    P, R, F1 = bert_score_func(
                        cands=batch_hyps,
                        refs=batch_refs,
                        model_type=model_name,
                        lang="zh",
                        device=device,
                        batch_size=effective_batch_size,
                        verbose=False,
                        rescale_with_baseline=False,
                        idf=False,
                    )
                
                all_P.extend(P.cpu().tolist())
                all_R.extend(R.cpu().tolist())
                all_F1.extend(F1.cpu().tolist())
                
                # 立即清理 GPU 缓存
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    
            except RuntimeError as batch_error:
                error_msg = str(batch_error)
                if "CUDA" in error_msg or "out of memory" in error_msg:
                    print(f"批次 {i//effective_batch_size + 1} 计算失败（CUDA错误）: {error_msg}")
                    # CUDA错误：清理缓存后继续
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
            except Exception as batch_error:
                print(f"批次 {i//effective_batch_size + 1} 计算失败: {batch_error}")
                # 跳过这个批次，继续处理下一批次
                continue
        
        if not all_F1:
            print("警告: 所有批次计算都失败了")
            return {
                "BERTScore_P": np.nan,
                "BERTScore_R": np.nan,
                "BERTScore_F1": np.nan,
            }
        
        # 计算平均值
        return {
            "BERTScore_P": np.mean(all_P),
            "BERTScore_R": np.mean(all_R),
            "BERTScore_F1": np.mean(all_F1),
        }
        
    except Exception as e:
        print(f"计算 BERTScore 时出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理 GPU 缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        return {
            "BERTScore_P": np.nan,
            "BERTScore_R": np.nan,
            "BERTScore_F1": np.nan,
        }


def calculate_char_level_f1(references: List[str], hypotheses: List[str]) -> float:
    """
    计算字符级F1分数（对中文更友好）
    
    Args:
        references: 参考文本列表
        hypotheses: 预测文本列表
    
    Returns:
        平均字符级F1分数 (0 to 1)
    """
    try:
        f1_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            # 去除空格和标点
            def get_clean_chars(text):
                return [c for c in text if c.strip() and not c in '，。！？、；：""''（）【】《》,.!?;:\'"()[]<>']
            
            ref_chars = get_clean_chars(ref)
            hyp_chars = get_clean_chars(hyp)
            
            if not ref_chars and not hyp_chars:
                f1_scores.append(1.0)  # 两者都为空视为完全匹配
                continue
            
            if not ref_chars or not hyp_chars:
                f1_scores.append(0.0)  # 一个为空一个不为空
                continue
            
            # 使用Counter计算多重集交集（考虑字符出现次数）
            ref_counter = Counter(ref_chars)
            hyp_counter = Counter(hyp_chars)
            
            # 交集：取两者中较小的计数
            intersection_count = sum((ref_counter & hyp_counter).values())
            
            # 计算精确率和召回率
            precision = intersection_count / len(hyp_chars) if hyp_chars else 0
            recall = intersection_count / len(ref_chars) if ref_chars else 0
            
            # 计算F1
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            f1_scores.append(f1)
        
        return np.mean(f1_scores) if f1_scores else 0.0
    except Exception as e:
        print(f"计算字符级F1时出错: {e}")
        return np.nan


def calculate_all_text_metrics(
    references: List[str],
    hypotheses: List[str],
    compute_bertscore: bool = True,
    bertscore_flags: List[bool] = None
) -> Dict[str, float]:
    """
    计算所有文本匹配指标
    
    Args:
        references: 参考文本列表
        hypotheses: 预测文本列表
        compute_bertscore: 是否计算 BERTScore（默认 True，使用 GPU 加速）
        bertscore_flags: 可选，布尔列表，指示每个样本是否计算 BERTScore。
                        如果为 None，则计算所有样本。
                        如果对应值为 False，该样本将不参与 BERTScore 计算（跳过）。
    
    Returns:
        Dict containing:
            - BLEU: BLEU 分数
            - CharF1: 字符级 F1 分数
            - BERTScore_P: BERTScore Precision（如果 compute_bertscore=True）
            - BERTScore_R: BERTScore Recall（如果 compute_bertscore=True）
            - BERTScore_F1: BERTScore F1（如果 compute_bertscore=True）
            - sample_count: 样本数量
    """
    result = {
        "BLEU": calculate_bleu(references, hypotheses),
        "CharF1": calculate_char_level_f1(references, hypotheses),
        "sample_count": len(references),
    }
    
    # 计算 BERTScore（使用 chinese-roberta-wwm-ext-large）
    if compute_bertscore:
        if bertscore_flags is not None:
            # 根据 flags 过滤需要计算的样本
            if len(bertscore_flags) != len(references):
                print(f"警告: bertscore_flags 长度 ({len(bertscore_flags)}) 与样本长度 ({len(references)}) 不一致，将忽略 flags")
                bert_refs = references
                bert_hyps = hypotheses
            else:
                bert_refs = [ref for ref, flag in zip(references, bertscore_flags) if flag]
                bert_hyps = [hyp for hyp, flag in zip(hypotheses, bertscore_flags) if flag]
                
                # 如果过滤后没有样本，返回 NaN
                if not bert_refs:
                    result.update({
                        "BERTScore_P": np.nan,
                        "BERTScore_R": np.nan,
                        "BERTScore_F1": np.nan,
                    })
                    return result
        else:
            # 计算所有样本
            bert_refs = references
            bert_hyps = hypotheses
            
        bertscore_results = calculate_bertscore(bert_refs, bert_hyps)
        result.update(bertscore_results)
    
    return result
