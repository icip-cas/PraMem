#!/usr/bin/env python3
"""
从 prediction_results.json 重新计算指标并生成 evaluation_report.json 和 evaluation_report.txt

使用示例:
    # 从 output 目录的预测结果重新生成 results 目录的报告
    python recalculate_metrics.py output/2026-01-19/2026-01-19_10-30-00/xxx_prediction_results.json
    
    # 指定输出目录
    python recalculate_metrics.py output/xxx_prediction_results.json -o results/
    
    # 批量处理目录下所有预测结果
    python recalculate_metrics.py output/2026-01-19/
    
    # 跳过 BERTScore 计算（快速模式，不需要 GPU）
    python recalculate_metrics.py output/xxx_prediction_results.json --skip-bertscore
    
    # 只计算 BERTScore（用于给旧结果补充 BERTScore）
    python recalculate_metrics.py output/xxx_prediction_results.json --only-bertscore
"""

import os

import json
import argparse
import os
import re
from pathlib import Path
from datetime import datetime

# 导入评估器的指标计算和报告生成功能
from evaluator import UserSimulationEvaluator
from config import RESULTS_DIR
from metrics import calculate_all_text_metrics, calculate_bertscore

# 全局配置：是否计算 BERTScore
COMPUTE_BERTSCORE = True

# 尝试导入 LLM Judge 模块
try:
    from recalculate_metrics_llm_judge import evaluate_ecommerce_dialogues, JUDGE_MODEL_NAME
    LLM_JUDGE_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: recalculate_metrics_llm_judge.py not found or failed to import.")
    LLM_JUDGE_AVAILABLE = False
    JUDGE_MODEL_NAME = None


def fix_json_trailing_commas(json_str: str) -> str:
    """
    修复 JSON 字符串中的尾随逗号问题
    
    例如: {"a": 1,} -> {"a": 1}
         [1, 2, 3,] -> [1, 2, 3]
    """
    # 移除对象尾随逗号: ,}
    json_str = re.sub(r',\s*}', '}', json_str)
    # 移除数组尾随逗号: ,]
    json_str = re.sub(r',\s*\]', ']', json_str)
    return json_str


def load_json_with_fix(file_path: str) -> list:
    """
    加载 JSON 文件，自动修复常见问题：
    1. 尾随逗号
    2. 截断的 JSON 文件（通过恢复有效记录）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 首先尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON 解析失败，尝试修复...")
        print(f"   原始错误位置: 行 {e.lineno}, 列 {e.colno}, 字符位置 {e.pos}")
        print(f"   错误信息: {e.msg}")
        
        # 显示错误位置附近的内容
        error_pos = e.pos
        context_start = max(0, error_pos - 200)
        context_end = min(len(content), error_pos + 100)
        print(f"\n   错误位置附近的内容:")
        print(f"   ...{content[context_start:error_pos]}<<<ERROR HERE>>>{content[error_pos:context_end]}...")
        
        # 尝试修复尾随逗号
        fixed_content = fix_json_trailing_commas(content)
        try:
            result = json.loads(fixed_content)
            print(f"✓ 尾随逗号修复成功!")
            return result
        except json.JSONDecodeError:
            pass
        
        # 尝试恢复截断的 JSON 数组
        print(f"\n⚠️  尝试恢复截断的 JSON 数组...")
        recovered = recover_truncated_json_array(content, error_pos)
        if recovered is not None:
            print(f"✓ 成功恢复 {len(recovered)} 条记录!")
            return recovered
        
        print(f"❌ 无法恢复 JSON 文件")
        raise e


def recover_truncated_json_array(content: str, error_pos: int) -> list:
    """
    尝试从损坏的 JSON 数组中恢复有效记录
    
    策略：向前搜索找到最后一个完整的顶层对象（},{ 模式表示一个顶层对象的结束和下一个的开始）
    """
    # 检查是否是 JSON 数组格式
    content_stripped = content.strip()
    if not content_stripped.startswith('['):
        print("   不是 JSON 数组格式，无法恢复")
        return None
    
    # 我们需要找到错误位置之前最后一个完整的顶层对象
    # 顶层对象的模式是: }, 后面跟着 { 或者 }, 后面跟着 ]
    # 我们使用正则表达式来找到所有的 },\s*{ 模式
    
    # 只在错误位置之前搜索
    search_content = content[:error_pos]
    
    # 找到所有 },[ 换行/空格 ]{ 的位置，这表示一个顶层对象的结束
    # 模式: }后面是可选的空白和逗号，然后是可选的空白和{
    pattern = re.compile(r'\}\s*,\s*\{')
    
    matches = list(pattern.finditer(search_content))
    
    if not matches:
        print("   未找到顶层对象分隔符 },{ ")
        # 尝试另一种方法：找最后一个 } 然后逐步回退
        return recover_by_backtracking(content, error_pos)
    
    print(f"   找到 {len(matches)} 个顶层对象分隔符")
    
    # 从最后一个匹配开始，逐个尝试恢复
    for match in reversed(matches):
        # 截取到这个 } 的位置（不包括逗号和后面的 {）
        end_pos = match.start() + 1  # +1 包含 }
        truncated = content[:end_pos] + '\n]'
        
        try:
            result = json.loads(truncated)
            return result
        except json.JSONDecodeError:
            continue
    
    print("   所有分隔符位置都无法恢复")
    return recover_by_backtracking(content, error_pos)


def recover_by_backtracking(content: str, error_pos: int) -> list:
    """
    通过回退的方式恢复：从错误位置向前逐步减少内容，直到找到有效的 JSON
    """
    print("   使用回退方法尝试恢复...")
    
    # 每次回退的步长（按对象大小估计）
    # 假设每个对象大约 500-2000 字符
    step_sizes = [500, 1000, 2000, 5000, 10000, 50000, 100000]
    
    for step in step_sizes:
        # 从错误位置向前回退 step 个字符
        test_pos = error_pos - step
        if test_pos < 100:
            continue
            
        # 在 test_pos 附近找一个 },{ 或 }] 的位置
        search_start = max(0, test_pos - 1000)
        search_end = test_pos + 1000
        search_content = content[search_start:min(search_end, error_pos)]
        
        # 找 }, 后跟 { 的位置
        pattern = re.compile(r'\}\s*,\s*\{')
        matches = list(pattern.finditer(search_content))
        
        if matches:
            # 使用最后一个匹配
            match = matches[-1]
            end_pos = search_start + match.start() + 1
            truncated = content[:end_pos] + '\n]'
            
            try:
                result = json.loads(truncated)
                print(f"   在回退 {step} 字符后成功恢复")
                return result
            except json.JSONDecodeError:
                continue
    
    print("   回退方法也无法恢复")
    return None


def extract_experiment_info_from_filename(filename: str) -> dict:
    """
    从文件名中提取实验信息
    
    例如: d4b7bf_30t_h~0930_t1001~1130_s42_128k_hsshop_Qwen3-235B_prediction_results.json
    """
    # 移除 _prediction_results.json 后缀
    name = filename.replace("_prediction_results.json", "")
    
    return {
        "experiment_name": name,
        "filename": filename
    }


def extract_timestamp_from_path(file_path: Path) -> tuple:
    """
    从文件路径中提取日期和时间戳
    
    路径格式: output/2026-01-19/2026-01-19_10-30-00/xxx_prediction_results.json
    """
    parts = file_path.parts
    
    run_date = None
    run_timestamp = None
    
    # 尝试从路径中找到日期格式的目录
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
    timestamp_pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}')
    
    for part in parts:
        if timestamp_pattern.match(part):
            run_timestamp = part
            run_date = part.split('_')[0]
        elif date_pattern.match(part) and run_date is None:
            run_date = part
    
    return run_date, run_timestamp


def recalculate_and_save_report(prediction_file: str, output_dir: str = None, 
                                compute_bertscore: bool = True, dry_run_judge: bool = False):
    """
    从 prediction_results.json 重新计算指标并保存报告
    
    Args:
        prediction_file: prediction_results.json 文件路径
        output_dir: 输出目录，默认为 results/
        compute_bertscore: 是否计算 BERTScore（需要 GPU）
        dry_run_judge: 是否以 dry-run 模式运行 LLM Judge (不调 API)
    """
    prediction_path = Path(prediction_file)
    
    if not prediction_path.exists():
        print(f"❌ 文件不存在: {prediction_file}")
        return False
    
    print(f"\n{'=' * 80}")
    print(f"📂 处理文件: {prediction_file}")
    print('=' * 80)
    
    # 加载预测结果（支持自动修复尾随逗号）
    results = load_json_with_fix(str(prediction_path))
    
    print(f"✓ 加载了 {len(results)} 条预测记录")
    
    # 提取实验信息
    info = extract_experiment_info_from_filename(prediction_path.name)
    experiment_name = info["experiment_name"]
    
    # 提取时间戳信息
    run_date, run_timestamp = extract_timestamp_from_path(prediction_path)
    
    if run_date is None:
        # 使用当前时间
        from pytz import timezone
        shanghai_tz = timezone('Asia/Shanghai')
        now = datetime.now(shanghai_tz)
        run_date = now.strftime("%Y-%m-%d")
        run_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        print(f"⚠️  未能从路径提取时间，使用当前时间: {run_timestamp}")
    else:
        print(f"✓ 提取时间信息: 日期={run_date}, 时间戳={run_timestamp}")
    
    # 创建一个轻量级的评估器（不需要模型配置）
    # 使用一个虚拟的模型配置
    dummy_model_config = {
        "name": "recalculate",
        "api_type": "dummy"
    }
    
    evaluator = UserSimulationEvaluator(
        model_config=dummy_model_config,
        experiment_name=experiment_name,
        run_date=run_date,
        run_timestamp=run_timestamp
    )
    
    
    # [新增] 运行 LLM Judge 评估 (如果是电商客服对话)
    # 检查是否包含电商客服对话
    has_ecommerce = any(r.get("action_type") == "电商客服对话" for r in results)
    
    if LLM_JUDGE_AVAILABLE and has_ecommerce:
        mode_str = " (DRY RUN)" if dry_run_judge else ""
        print(f"\n🤖 运行 LLM Judge 评估 (模型: {JUDGE_MODEL_NAME}){mode_str}...")
        # 注意：这会修改 results 列表，增加 llm_judge_scores
        results = evaluate_ecommerce_dialogues(results, judge_model_name=JUDGE_MODEL_NAME, dry_run=dry_run_judge)
    elif not LLM_JUDGE_AVAILABLE and has_ecommerce:
        print(f"\n⚠️  跳过 LLM Judge 评估 (模块未加载)")
    
    # 计算指标
    bertscore_info = "（含 BERTScore）" if compute_bertscore else "（跳过 BERTScore）"
    print(f"\n📊 计算指标...{bertscore_info}")
    metrics = evaluator.calculate_metrics(results, compute_bertscore=compute_bertscore)
    
    # 确定输出路径
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    # 保存报告
    report_filename = f"{experiment_name}_evaluation_report.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    print(f"\n💾 保存报告...")
    evaluator.save_metrics_report(metrics, report_path)
    
    # [新增] 如果计算了 BERTScore，保存带有分数的完整结果到新文件
    if compute_bertscore:
        # 新文件名：原文件名 + _with_bertscore.json
        # 注意：prediction_path.name 可能是 xxx_prediction_results.json
        new_filename = prediction_path.name.replace(".json", "_with_bertscore.json")
        new_output_path = os.path.join(output_dir, new_filename)
        
        print(f"\n💾 保存带有 BERTScore 的完整结果到: {new_output_path}")
        with open(new_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印关键指标
    print(f"\n📈 关键指标:")
    overall = metrics.get("overall", {})
    print(f"   Micro_F1: {overall.get('Micro_F1', 'N/A')*100:.2f}%" if overall.get('Micro_F1') else "   Micro_F1: N/A")
    print(f"   Macro_F1: {overall.get('Macro_F1', 'N/A')*100:.2f}%" if overall.get('Macro_F1') else "   Macro_F1: N/A")
    print(f"   Total_TP: {overall.get('Total_TP', 'N/A')}")
    print(f"   Total_FP: {overall.get('Total_FP', 'N/A')}")
    print(f"   Total_FN: {overall.get('Total_FN', 'N/A')}")
    
    print(f"   Total_FN: {overall.get('Total_FN', 'N/A')}")
    
    # 打印 LLM Judge 指标
    llm_judge_metrics = metrics.get("llm_judge_metrics", {})
    if llm_judge_metrics:
        print(f"\n🤖 LLM Judge 指标 (电商客服):")
        avg = llm_judge_metrics.get("average_score")
        print(f"   Average Score: {avg:.2f}" if avg else "   Average Score: N/A")
        print(f"   Intent Fidelity: {llm_judge_metrics.get('intent_fidelity', 0):.2f}")
        print(f"   Persona Mimicry: {llm_judge_metrics.get('persona_mimicry', 0):.2f}")
    
    # 打印文本指标（包括 BERTScore）
    text_metrics = metrics.get("text_metrics", {})
    if text_metrics:
        print(f"\n📝 文本指标:")
        for field, field_metrics in text_metrics.items():
            bleu = field_metrics.get("BLEU", float('nan'))
            char_f1 = field_metrics.get("CharF1", float('nan'))
            bert_f1 = field_metrics.get("BERTScore_F1", None)
            
            if bert_f1 is not None and not (isinstance(bert_f1, float) and str(bert_f1) == 'nan'):
                print(f"   {field}: BLEU={bleu*100:.2f}%, CharF1={char_f1*100:.2f}%, BERTScore_F1={bert_f1*100:.2f}%")
            else:
                print(f"   {field}: BLEU={bleu*100:.2f}%, CharF1={char_f1*100:.2f}%")
    
    print(f"\n✅ 完成!")
    return True


def find_prediction_files(directory: Path) -> list:
    """在目录中递归查找所有预测结果文件"""
    files = list(directory.rglob("*prediction_results*.json"))
    # 排除 metrics 文件
    files = [f for f in files if '_metrics' not in f.name]
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="从 prediction_results.json 重新计算指标并生成 evaluation_report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件（包含 BERTScore）
  python recalculate_metrics.py output/2026-01-19/xxx_prediction_results.json
  
  # 指定输出目录
  python recalculate_metrics.py output/xxx_prediction_results.json -o results/
  
  # 批量处理目录
  python recalculate_metrics.py output/2026-01-19/
  
  # 跳过 BERTScore（快速模式，不需要 GPU）
  python recalculate_metrics.py output/xxx_prediction_results.json --skip-bertscore
        """
    )
    parser.add_argument("input_path", help="prediction_results.json 文件路径或目录")
    parser.add_argument("--output", "-o", help="输出目录 (默认: results/)")
    parser.add_argument("--skip-bertscore", action="store_true", 
                       help="跳过 BERTScore 计算（快速模式，不需要 GPU）")
    parser.add_argument("--dry-run-judge", action="store_true",
                       help="以 dry-run 模式运行 LLM Judge (不调 API，返回模拟分数)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_dir = f"{args.input_path}_results"
    compute_bertscore = not args.skip_bertscore
    dry_run_judge = args.dry_run_judge
    
    if compute_bertscore:
        print("📌 将计算 BERTScore（使用 chinese-roberta-wwm-ext-large，需要 GPU）")
    else:
        print("📌 跳过 BERTScore 计算（快速模式）")
    
    if not input_path.exists():
        print(f"❌ 路径不存在: {args.input_path}")
        return 1
    
    if input_path.is_file():
        # 单文件处理
        success = recalculate_and_save_report(str(input_path), output_dir, compute_bertscore, dry_run_judge)
        return 0 if success else 1
    
    elif input_path.is_dir():
        # 目录处理
        files = find_prediction_files(input_path)
        
        if not files:
            print(f"❌ 目录中未找到 prediction_results 文件: {args.input_path}")
            return 1
        
        print(f"找到 {len(files)} 个预测结果文件:")
        for f in files:
            print(f"  - {f}")
        
        success_count = 0
        for file_path in files:
            try:
                if recalculate_and_save_report(str(file_path), output_dir, compute_bertscore, dry_run_judge):
                    success_count += 1
            except Exception as e:
                print(f"❌ 处理文件出错 {file_path}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'=' * 80}")
        print(f"处理完成: {success_count}/{len(files)} 个文件成功")
        return 0 if success_count == len(files) else 1
    
    else:
        print(f"❌ 无效的路径: {args.input_path}")
        return 1


if __name__ == "__main__":
    exit(main())
