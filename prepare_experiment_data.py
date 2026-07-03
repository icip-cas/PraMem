"""
准备固定实验数据集
选择用户并将其行为分割为上下文集和测试集

采样策略（均衡采样，确保凑满 TEST_LENGTH 个行为）：
1. 时间均衡：将测试时间范围按时间切分为M个桶，保证时间维度的均衡性
2. 领域均衡：轮询各个领域（视频浏览、直播间等），保证各领域样本数量尽可能一致
3. 价值均衡：在选定领域内，以50%概率采样高价值（如点赞、购买）或低价值Item
4. 只跳过 context 信息不足的行为，继续寻找直到凑满 TEST_LENGTH 个

缓存复用功能：
- 根据配置参数生成唯一签名，避免重复采样
- 如果相同配置的数据集已存在，直接复用
"""
import json
import os
import random
import hashlib
import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from config import *
from data_processor import load_user_data
from prompt_builder import estimate_token_count, format_action_context, format_action_result, should_filter_action
from behavior_distribution import analyze_test_data_distribution
from datetime import datetime
import pytz


def get_stable_user_seed_offset(user_id: str) -> int:
    """
    基于用户ID生成稳定的随机种子偏移量
    
    该函数确保：
    - 同一用户ID总是映射到相同的偏移量
    - 不同用户ID映射到不同的偏移量（极低碰撞概率）
    - 偏移量范围在合理区间内（0到10^9）
    
    这样无论用户在列表中的位置如何变化，同一用户ID总是使用相同的随机种子，
    从而保证采样结果的一致性。
    
    Args:
        user_id: 用户ID（字符串）
    
    Returns:
        稳定的整数偏移量
    """
    # 使用 SHA256 生成稳定的哈希值（比 MD5 碰撞概率更低）
    hash_obj = hashlib.sha256(user_id.encode('utf-8'))
    # 取前8个字节转换为整数，然后模一个大数得到偏移量
    hash_bytes = hash_obj.digest()[:8]
    offset = int.from_bytes(hash_bytes, byteorder='big') % (10**9)
    return offset


def generate_config_signature(
    num_users: int,
    test_length: int,
    history_time_start: str = None,
    history_time_end: str = None,
    test_time_start: str = None,
    test_time_end: str = None,
    seed: int = 42,
    input_data_path: str = None,
) -> str:
    """
    根据配置参数生成唯一签名
    
    重要：只包含影响"用户选择"和"测试数据采样"的参数。
    评估参数（max_history_tokens, max_history_days, history_scene_filter）
    不影响签名，它们只在评估时影响 prompt 构建。
    
    这样可以确保：改变评估参数时，不会重新采样用户和测试数据，
    从而保证不同配置之间的公平对比。
    
    Args:
        num_users: 用户数量
        test_length: 测试集长度
        history_time_start/end: 历史时间范围（影响用户筛选条件）
        test_time_start/end: 测试时间范围（影响测试数据采样）
        seed: 随机种子
        input_data_path: 输入数据路径
    
    Returns:
        8位哈希签名
    """
    # 只包含影响用户选择和测试数据采样的参数
    # 注意：max_history_tokens, max_history_days, history_scene_filter 不再影响签名
    config_str = f"{num_users}|{test_length}|" \
                 f"{history_time_start or ''}|{history_time_end or ''}|" \
                 f"{test_time_start or ''}|{test_time_end or ''}|" \
                 f"{seed}|{input_data_path or ''}"
    
    # 生成 MD5 哈希，取前8位
    hash_obj = hashlib.md5(config_str.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def generate_descriptive_dirname(
    num_users: int,
    test_length: int,
    history_time_start: str = None,
    history_time_end: str = None,
    test_time_start: str = None,
    test_time_end: str = None,
    seed: int = 42,
    signature: str = None
) -> str:
    """
    生成具有描述性的目录名
    
    格式示例：200u_30t_h~0930_t1001~1130_s42_abc12345
    
    注意：目录名只包含影响采样的核心参数，不包含评估参数
    （max_history_tokens, max_history_days, history_scene_filter）
    这样可以确保相同采样配置的数据集使用同一个目录。
    
    Args:
        num_users: 用户数量
        test_length: 测试集长度
        history_time_start/end: 历史时间范围
        test_time_start/end: 测试时间范围
        seed: 随机种子
        signature: 配置签名
    
    Returns:
        描述性目录名
    """
    # 基础信息：用户数_测试数
    parts = [
        f"{num_users}u",
        f"{test_length}t",
    ]
    
    # 历史时间范围（简化格式：MMDD）
    if history_time_start or history_time_end:
        h_start = history_time_start[5:7] + history_time_start[8:10] if history_time_start else ""
        h_end = history_time_end[5:7] + history_time_end[8:10] if history_time_end else ""
        parts.append(f"h{h_start}~{h_end}")
    
    # 测试时间范围（简化格式：MMDD）
    if test_time_start or test_time_end:
        t_start = test_time_start[5:7] + test_time_start[8:10] if test_time_start else ""
        t_end = test_time_end[5:7] + test_time_end[8:10] if test_time_end else ""
        parts.append(f"t{t_start}~{t_end}")
    
    # 随机种子
    parts.append(f"s{seed}")
    
    # 配置签名（用于唯一标识）
    if signature:
        parts.append(signature)
    
    return "_".join(parts)


def find_existing_dataset(
    output_dir: str,
    signature: str
) -> Optional[str]:
    """
    查找已存在的相同配置的数据集
    
    Args:
        output_dir: 数据集根目录（如 dataset/）
        signature: 配置签名
    
    Returns:
        如果找到匹配的数据集，返回主 JSON 文件路径；否则返回 None
    """
    if not os.path.exists(output_dir):
        return None
    
    # 在 output_dir 下查找所有包含签名的目录
    # 目录结构：dataset/YYYY-MM-DD/描述性目录名/
    pattern = os.path.join(output_dir, "*", f"*_{signature}", "*.json")
    
    matching_files = glob.glob(pattern)
    
    # 过滤掉分析文件和分布文件，只保留主数据文件
    main_files = [
        f for f in matching_files 
        if not f.endswith("_analysis.json") 
        and not f.endswith("_test_distribution.json")
        and not os.path.basename(f).endswith("_sampled_actions.json")
    ]
    
    if main_files:
        # 返回最新的匹配文件
        main_files.sort(key=os.path.getmtime, reverse=True)
        return main_files[0]
    
    return None


# ============================================================
# 高价值动作定义
# Key: 领域/场景类型
# Value: 该领域下的高价值动作集合
# ============================================================
HIGH_VALUE_ACTIONS = {
    "视频浏览": {"like", "collect", "share", "comment"},
    "直播间": {"send_gift", "add_to_cart", "follow"},
    "商城购物": {"purchase", "order_success", "add_to_cart"},
    "广告推荐": {"conversion", "activation", "purchase", "submit", "click"},
    "电商客服对话": {"purchase", "positive_feedback"},
}


def _get_action_summary(action: Dict, max_length: int = 100) -> str:
    """
    获取行为的摘要信息
    
    Args:
        action: 行为记录
        max_length: 摘要最大长度
    
    Returns:
        行为摘要字符串
    """
    action_type = action.get("type", "")
    context = action.get("context", {})
    
    if action_type == "视频浏览":
        caption = context.get("caption", "")
        if caption:
            return caption[:max_length] + ("..." if len(caption) > max_length else "")
        return "无标题视频"
    
    elif action_type == "直播间":
        live_title = context.get("live_title", "")
        live_category = context.get("live_category", "")
        if live_title:
            return f"[{live_category}] {live_title}"[:max_length]
        return f"[{live_category}] 直播间"
    
    elif action_type == "商城购物":
        product_name = context.get("product_name", context.get("item_name", ""))
        if product_name:
            return product_name[:max_length]
        return "商品浏览"
    
    elif action_type == "广告推荐":
        ad_title = context.get("ad_title", context.get("title", ""))
        if ad_title:
            return ad_title[:max_length]
        return "广告内容"
    
    else:
        # 尝试获取任何可用的文本描述
        for key in ["title", "name", "content", "description", "caption"]:
            if key in context and context[key]:
                return str(context[key])[:max_length]
        return f"{action_type}行为"


def is_high_value_action(item: Dict) -> bool:
    """
    判断一个行为是否为高价值行为
    
    逻辑：
        根据行为的领域（type字段），检查其包含的动作列表（action字段）中
        是否存在定义的"高价值动作"。
    
    Args:
        item: 单个行为记录，包含 'type' 和 'action' 字段
    
    Returns:
        bool: True 表示高价值，False 表示低价值
    """
    domain = item.get("type", "")
    if domain not in HIGH_VALUE_ACTIONS:
        # 如果领域未定义高价值动作，默认视为低价值
        return False
    
    target_actions = HIGH_VALUE_ACTIONS[domain]
    
    # 遍历Item的所有动作
    actions = item.get("action", [])
    if not isinstance(actions, list):
        return False
    
    for act in actions:
        if act.get("type") in target_actions:
            return True
    
    return False


def has_sufficient_context_for_prediction(action: Dict) -> bool:
    """
    检查行为的 context 是否有足够的信息用于预测
    
    对于某些场景，如果 context 中缺乏关键文本字段，无法准确预测用户行为，
    应该排除在测试集之外。
    
    Args:
        action: 行为记录
    
    Returns:
        True 表示 context 信息充足，可以用于预测
        False 表示 context 信息不足，应排除
    """
    action_type = action.get("type", "")
    context = action.get("context", {})
    
    # 视频浏览场景：caption、ocr、asr 至少有一个非空
    if action_type == "视频浏览":
        caption = context.get("caption", "")
        ocr = context.get("ocr", "")
        asr = context.get("asr", "")
        # 三个字段都为空才过滤
        if (not caption or not caption.strip()) and \
           (not ocr or not ocr.strip()) and \
           (not asr or not asr.strip()):
            return False
    
    # 直播间场景：必须有直播标题 (live_title) 或直播类别 (live_category)
    elif action_type == "直播间":
        live_title = context.get("live_title", "")
        live_category = context.get("live_category", "")
        # 至少要有一个关键信息
        if (not live_title or not live_title.strip()) and (not live_category or not live_category.strip()):
            return False
    
    return True


def split_user_actions_by_tokens(
    user_data: Dict,
    test_length: int,
    history_time_start: str = None,
    history_time_end: str = None,
    test_time_start: str = None,
    test_time_end: str = None,
    seed: int = None
) -> Tuple[Dict, List[Dict], int, List[Dict]]:
    """
    基于时间范围分割用户行为，支持历史和测试集分离的时间范围
    
    采样策略：使用均衡采样（时间均衡 + 领域均衡 + 高低价值均衡）
    只跳过 context 信息不足的行为，确保凑满 test_length 个行为
    
    重要：
    1. 为了支持滚动预测，会保存完整的原始行为时间线
    2. history_scene_filter 和 max_history_tokens 不在此阶段处理，
       它们只在评估时影响 prompt 构建，不影响数据采样。
       这样可以确保不同配置能对比相同的数据集。
    
    Args:
        user_data: 用户数据（包含 action_history）
        test_length: 测试行为数量
        history_time_start: 历史行为时间范围起始（格式：YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS）
        history_time_end: 历史行为时间范围结束
        test_time_start: 测试集时间范围起始
        test_time_end: 测试集时间范围结束
        seed: 随机种子，确保每次运行结果一致
    
    Returns:
        (history_data, test_actions_with_index, actual_tokens, skipped_actions_info)
        - history_data: 包含user_profile和初始历史行为的字典
        - test_actions_with_index: 待预测的行为列表，每个包含 action 和 original_index
        - actual_tokens: 初始历史行为的 Token 数（估算值）
        - skipped_actions_info: 被跳过的行为信息列表（用于记录）
    """
    all_original_actions = user_data["action_history"]
    
    # 判断是否使用分离时间范围模式
    use_separated_time_range = (history_time_start or history_time_end or test_time_start or test_time_end)
    
    # 筛选基础历史行为（只根据历史时间范围，不按场景过滤）
    # 场景过滤在评估时进行，这样可以对比不同场景过滤配置的效果
    base_history_pool = []
    for action in all_original_actions:
        timestamp = action.get("timestamp", "")
        in_range = True
        if history_time_start and timestamp < history_time_start:
            in_range = False
        if history_time_end and timestamp > history_time_end:
            in_range = False
        if in_range:
            base_history_pool.append(action)
    
    # 筛选测试时间范围内的所有行为（如10-11月份）
    # 这些行为会用于滚动预测时的增量历史
    test_time_all_actions = []
    for action in all_original_actions:
        timestamp = action.get("timestamp", "")
        in_range = True
        if test_time_start and timestamp < test_time_start:
            in_range = False
        if test_time_end and timestamp > test_time_end:
            in_range = False
        if in_range:
            test_time_all_actions.append(action)
    
    # 从测试时间范围中筛选符合条件的待预测行为
    # 条件：context 信息充足（不再筛选场景类型）
    test_candidates_all = []
    skipped_actions_info = []  # 记录被跳过的行为信息
    
    for idx, action in enumerate(test_time_all_actions):
        action_type = action.get("type", "")
        
        # 检查 context 信息是否充足
        if not has_sufficient_context_for_prediction(action):
            skipped_actions_info.append({
                "action": action,
                "skip_reason": "context信息不足",
                "action_type": action_type,
                "timestamp": action.get("timestamp", "")
            })
            continue
        
        # 检查是否会在评估时被过滤（如 play_duration 为 0）
        if should_filter_action(action):
            skipped_actions_info.append({
                "action": action,
                "skip_reason": "评估时会被过滤（如play_duration为0）",
                "action_type": action_type,
                "timestamp": action.get("timestamp", "")
            })
            continue
        
        # test_time_index: 在测试时间范围内的索引（用于滚动预测时确定之前的行为）
        test_candidates_all.append({"action": action, "test_time_index": idx})
    
    # 始终使用均衡采样：时间均衡 + 领域均衡 + 高低价值均衡
    test_actions_with_index = _balanced_sample(
        test_candidates_all, 
        test_length,
        m_buckets=10,  # 时间分桶数量
        seed=seed  # 传入随机种子确保一致性
    )
    
    # 基础历史池用于统计
    filtered_base_history = base_history_pool
    
    # 计算基础历史的 Token 数（用于统计）
    base_history_tokens = 0
    for action in filtered_base_history:
        timestamp = action.get("timestamp", "未知时间")
        action_type = action.get("type", "未知行为")
        context_str = format_action_context(action)
        result_str = format_action_result(action)
        
        action_text = (
            f"【行为】时间：{timestamp}\n"
            f"  场景：{action_type}\n"
            f"  详情：{context_str}\n"
            f"  反应：{result_str}\n"
        )
        base_history_tokens += estimate_token_count(action_text)
    
    # 返回数据结构
    # - base_history_pool: 基础历史行为（如9月份的所有行为）
    # - test_time_all_actions: 测试时间范围内的所有行为（用于滚动预测时构建增量历史）
    # - test_actions_with_index: 被采样的待预测行为
    # - skipped_actions_info: 被跳过的行为信息
    result_data = {
        "user_profile": user_data["user_profile"],
        "base_history": base_history_pool,  # 基础历史（如9月份）
        "test_time_all_actions": test_time_all_actions,  # 测试时间范围内所有行为
    }
    
    return result_data, test_actions_with_index, base_history_tokens, skipped_actions_info


def _balanced_sample(
    candidates_all: List[Dict],
    total_count: int,
    m_buckets: int = 10,
    seed: int = None
) -> List[Dict]:
    """
    均衡采样：时间均衡 + 领域均衡 + 高低价值均衡
    
    采样逻辑：
    1. 将候选行为按时间切分为M个桶，保证时间维度的均衡性
    2. 在每个桶内进行分层采样：
       - 领域均衡：轮询各个领域（视频浏览、直播间等）
       - 价值均衡：50%概率选高价值，50%概率选低价值
    
    Args:
        candidates_all: 所有符合条件的候选行为（已按时间排序，包含 test_time_index）
        total_count: 要采样的数量
        m_buckets: 时间分桶数量，默认10
        seed: 随机种子，确保每次运行结果一致
    
    Returns:
        均衡采样后的行为列表
    """
    # 设置随机种子，确保每次运行结果一致
    if seed is not None:
        random.seed(seed)
    if not candidates_all or total_count <= 0:
        return []
    
    n = len(candidates_all)
    
    # 如果候选数量不足，全部返回
    if n <= total_count:
        return candidates_all.copy()
    
    # 动态调整桶数量：如果数据量较少，减少桶数
    actual_buckets = min(m_buckets, n // 2, total_count)
    if actual_buckets < 1:
        actual_buckets = 1
    
    # 计算每个桶的目标采样数
    base_sample_per_bucket = total_count // actual_buckets
    remainder = total_count % actual_buckets
    
    # 将候选数据切分为 M 个桶
    bucket_data_size = n // actual_buckets
    buckets = []
    
    start_idx = 0
    for i in range(actual_buckets):
        if i == actual_buckets - 1:
            end_idx = n
        else:
            end_idx = start_idx + bucket_data_size
        buckets.append(candidates_all[start_idx:end_idx])
        start_idx = end_idx
    
    sampled_results = []
    
    # 在每个桶内进行采样
    for i, bucket_items in enumerate(buckets):
        if not bucket_items:
            continue
        
        # 当前桶需要采样的数量
        target_k = base_sample_per_bucket + (1 if i < remainder else 0)
        if target_k <= 0:
            continue
        
        # 如果桶内数据量少于目标量，全取
        if len(bucket_items) <= target_k:
            sampled_results.extend(bucket_items)
            continue
        
        # --- 桶内采样逻辑 ---
        # 将Item按领域(domain)和价值(value)分组
        # domain_pools: { domain_name: { 'high': [items], 'low': [items] } }
        domain_pools = defaultdict(lambda: {'high': [], 'low': []})
        
        for item in bucket_items:
            action = item.get("action", item)  # 兼容两种格式
            domain = action.get("type", "unknown")
            is_high = is_high_value_action(action)
            key = 'high' if is_high else 'low'
            domain_pools[domain][key].append(item)
        
        # 获取所有存在的领域列表，用于轮询
        active_domains = list(domain_pools.keys())
        random.shuffle(active_domains)
        
        bucket_sampled = []
        domain_idx = 0
        
        while len(bucket_sampled) < target_k and active_domains:
            # 轮询领域
            current_domain = active_domains[domain_idx]
            pool = domain_pools[current_domain]
            
            # 检查是否有剩余Item
            has_high = len(pool['high']) > 0
            has_low = len(pool['low']) > 0
            
            if not has_high and not has_low:
                # 该领域已空
                active_domains.pop(domain_idx)
                if active_domains:
                    domain_idx = domain_idx % len(active_domains)
                continue
            
            # 决定优先采样的类别（50%概率）
            if has_high and has_low:
                pick_high = random.random() < 0.5
            elif has_high:
                pick_high = True
            else:
                pick_high = False
            
            # 执行采样
            if pick_high:
                pop_idx = random.randint(0, len(pool['high']) - 1)
                selected_item = pool['high'].pop(pop_idx)
            else:
                pop_idx = random.randint(0, len(pool['low']) - 1)
                selected_item = pool['low'].pop(pop_idx)
            
            bucket_sampled.append(selected_item)
            
            # 移动到下一个领域
            if active_domains:
                domain_idx = (domain_idx + 1) % len(active_domains)
        
        sampled_results.extend(bucket_sampled)
    
    # 按 test_time_index 排序，保持时间顺序
    sampled_results.sort(key=lambda x: x.get("test_time_index", 0))
    
    return sampled_results


def analyze_sampled_actions(experiment_data: Dict, output_path: str):
    """
    对采样的行为进行详细分析，输出分析报告
    
    分析内容：
    1. 时间范围分析
    2. type（场景类型）种类分布比例
    3. action（用户行为）类型分布
    4. 高价值/低价值行为分布
    5. 每个用户的采样统计
    
    Args:
        experiment_data: 实验数据
        output_path: 分析报告输出路径
    """
    analysis = {
        "summary": {},
        "time_analysis": {},
        "type_distribution": {},
        "action_distribution": {},
        "value_distribution": {},
        "per_user_stats": []
    }
    
    # 收集所有采样的行为
    all_sampled_actions = []
    all_timestamps = []
    type_counts = defaultdict(int)
    action_counts = defaultdict(int)
    high_value_count = 0
    low_value_count = 0
    
    for user_entry in experiment_data["users"]:
        user_id = user_entry["user_id"]
        test_actions = user_entry.get("test_actions", [])
        
        user_type_counts = defaultdict(int)
        user_action_counts = defaultdict(int)
        user_high_value = 0
        user_low_value = 0
        user_timestamps = []
        
        for test_item in test_actions:
            action = test_item["action"]
            all_sampled_actions.append(action)
            
            # 时间戳
            timestamp = action.get("timestamp", "")
            if timestamp:
                all_timestamps.append(timestamp)
                user_timestamps.append(timestamp)
            
            # type 统计
            action_type = action.get("type", "未知")
            type_counts[action_type] += 1
            user_type_counts[action_type] += 1
            
            # action 统计（用户行为类型）
            actions_list = action.get("action", [])
            if isinstance(actions_list, list):
                for act in actions_list:
                    act_type = act.get("type", "unknown")
                    action_counts[act_type] += 1
                    user_action_counts[act_type] += 1
            
            # 高价值/低价值统计
            if is_high_value_action(action):
                high_value_count += 1
                user_high_value += 1
            else:
                low_value_count += 1
                user_low_value += 1
        
        # 用户统计
        user_stats = {
            "user_id": user_id,
            "sampled_count": len(test_actions),
            "time_range": {
                "earliest": min(user_timestamps) if user_timestamps else None,
                "latest": max(user_timestamps) if user_timestamps else None
            },
            "type_distribution": dict(user_type_counts),
            "action_distribution": dict(user_action_counts),
            "high_value_count": user_high_value,
            "low_value_count": user_low_value,
            "high_value_ratio": user_high_value / len(test_actions) if test_actions else 0
        }
        analysis["per_user_stats"].append(user_stats)
    
    # 汇总统计
    total_sampled = len(all_sampled_actions)
    analysis["summary"] = {
        "total_users": len(experiment_data["users"]),
        "total_sampled_actions": total_sampled,
        "avg_actions_per_user": total_sampled / len(experiment_data["users"]) if experiment_data["users"] else 0,
        "seed": experiment_data["metadata"].get("seed", "unknown"),
        "sampling_strategy": experiment_data["metadata"].get("sampling_strategy", "balanced")
    }
    
    # 时间分析
    if all_timestamps:
        sorted_timestamps = sorted(all_timestamps)
        analysis["time_analysis"] = {
            "earliest": sorted_timestamps[0],
            "latest": sorted_timestamps[-1],
            "total_count": len(sorted_timestamps),
            # 按月统计
            "by_month": {}
        }
        # 按月统计
        month_counts = defaultdict(int)
        for ts in all_timestamps:
            month = ts[:7] if len(ts) >= 7 else "unknown"
            month_counts[month] += 1
        analysis["time_analysis"]["by_month"] = dict(sorted(month_counts.items()))
        
        # 按日统计（只统计前10天和后10天）
        day_counts = defaultdict(int)
        for ts in all_timestamps:
            day = ts[:10] if len(ts) >= 10 else "unknown"
            day_counts[day] += 1
        sorted_days = sorted(day_counts.items())
        analysis["time_analysis"]["by_day_sample"] = {
            "first_10_days": dict(sorted_days[:10]),
            "last_10_days": dict(sorted_days[-10:]) if len(sorted_days) > 10 else {}
        }
    
    # type（场景类型）分布
    analysis["type_distribution"] = {
        "counts": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "ratios": {
            k: round(v / total_sampled * 100, 2) 
            for k, v in sorted(type_counts.items(), key=lambda x: -x[1])
        } if total_sampled > 0 else {}
    }
    
    # action（用户行为）分布
    analysis["action_distribution"] = {
        "counts": dict(sorted(action_counts.items(), key=lambda x: -x[1])),
        "ratios": {
            k: round(v / sum(action_counts.values()) * 100, 2) 
            for k, v in sorted(action_counts.items(), key=lambda x: -x[1])
        } if action_counts else {}
    }
    
    # 高价值/低价值分布
    analysis["value_distribution"] = {
        "high_value_count": high_value_count,
        "low_value_count": low_value_count,
        "high_value_ratio": round(high_value_count / total_sampled * 100, 2) if total_sampled > 0 else 0,
        "low_value_ratio": round(low_value_count / total_sampled * 100, 2) if total_sampled > 0 else 0
    }
    
    # 保存分析报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print(f"\n" + "=" * 80)
    print("采样行为分析报告")
    print("=" * 80)
    print(f"\n【基本信息】")
    print(f"  总用户数: {analysis['summary']['total_users']}")
    print(f"  总采样行为数: {analysis['summary']['total_sampled_actions']}")
    print(f"  平均每用户: {analysis['summary']['avg_actions_per_user']:.1f} 条")
    print(f"  随机种子: {analysis['summary']['seed']}")
    
    print(f"\n【时间范围】")
    if analysis["time_analysis"]:
        print(f"  最早: {analysis['time_analysis']['earliest']}")
        print(f"  最晚: {analysis['time_analysis']['latest']}")
        print(f"  按月分布:")
        for month, count in analysis["time_analysis"]["by_month"].items():
            print(f"    {month}: {count} 条")
    
    print(f"\n【场景类型分布】")
    for action_type, count in analysis["type_distribution"]["counts"].items():
        ratio = analysis["type_distribution"]["ratios"].get(action_type, 0)
        print(f"  {action_type}: {count} 条 ({ratio}%)")
    
    print(f"\n【用户行为(action)分布】Top 10")
    for i, (act_type, count) in enumerate(list(analysis["action_distribution"]["counts"].items())[:10]):
        ratio = analysis["action_distribution"]["ratios"].get(act_type, 0)
        print(f"  {act_type}: {count} 次 ({ratio}%)")
    
    print(f"\n【高价值/低价值分布】")
    print(f"  高价值行为: {analysis['value_distribution']['high_value_count']} 条 ({analysis['value_distribution']['high_value_ratio']}%)")
    print(f"  低价值行为: {analysis['value_distribution']['low_value_count']} 条 ({analysis['value_distribution']['low_value_ratio']}%)")
    
    print(f"\n分析报告已保存到: {output_path}")
    print("=" * 80)
    
    return analysis


def _uniform_sample_by_time(
    candidates_all: List[Dict],
    total_count: int
) -> List[Dict]:
    """
    按时间均匀采样（简单版本，在时间范围内均匀分布采样点）
    
    采样逻辑：
    - total_count=1: 取中间位置的数据
    - total_count=2: 取两端的数据
    - total_count=N: 在时间范围内均匀分布 N 个采样点
    
    Args:
        candidates_all: 所有符合条件的候选行为（已按时间排序）
        total_count: 要采样的数量
    
    Returns:
        均匀采样后的行为列表
    """
    if not candidates_all or total_count <= 0:
        return []
    
    n = len(candidates_all)
    
    # 如果候选数量不足，全部返回
    if n <= total_count:
        return candidates_all.copy()
    
    result = []
    
    if total_count == 1:
        # 取中间位置
        target_idx = n // 2
        result.append(candidates_all[target_idx])
    else:
        # 均匀分布采样点
        for i in range(total_count):
            target_idx = int(i * (n - 1) / (total_count - 1))
            result.append(candidates_all[target_idx])
    
    return result






def select_users(
    all_users: Dict,
    num_users: int,
    test_length: int,
    history_time_start: str = None,
    history_time_end: str = None,
    test_time_start: str = None,
    test_time_end: str = None,
    min_history_actions: int = 1,
    seed: int = 42
) -> List[str]:
    """
    选择符合条件的用户
    
    筛选条件：
    1. 测试时间范围内有足够的 context 信息充足的行为（>= test_length）
    2. 历史时间范围内有足够的历史行为（>= min_history_actions）
    
    注意：history_scene_filter 不再影响用户选择，它只在评估时过滤历史行为。
    这样可以确保不同的 history_scene_filter 配置能对比相同的用户集合。
    
    Args:
        all_users: 所有用户数据
        num_users: 需要选择的用户数量
        test_length: 测试集长度
        history_time_start: 历史行为时间范围起始
        history_time_end: 历史行为时间范围结束
        test_time_start: 测试集时间范围起始
        test_time_end: 测试集时间范围结束
        min_history_actions: 最少需要的历史行为数量（默认1）
        seed: 随机种子
    
    Returns:
        选中的用户ID列表
    """
    random.seed(seed)
    
    print(f"\n筛选用户...")
    print(f"  历史行为: 需要至少 {min_history_actions} 条")
    print(f"  测试集: 需要至少 {test_length} 条 context 信息充足的行为")
    print(f"  注: 视频浏览需要有标题(caption)，直播间需要有标题(live_title)或类别(live_category)")
    
    if history_time_start or history_time_end:
        print(f"  历史时间范围: {history_time_start or '不限'} ~ {history_time_end or '不限'}")
    if test_time_start or test_time_end:
        print(f"  测试时间范围: {test_time_start or '不限'} ~ {test_time_end or '不限'}")
    
    # 筛选满足条件的用户
    valid_users = []
    
    for user_id, user_data in all_users.items():
        actions = user_data.get("action_history", [])
        
        # 筛选历史行为（只根据时间范围，不按场景过滤）
        history_actions = []
        for action in actions:
            timestamp = action.get("timestamp", "")
            in_range = True
            if history_time_start and timestamp < history_time_start:
                in_range = False
            if history_time_end and timestamp > history_time_end:
                in_range = False
            if in_range:
                history_actions.append(action)
        
        # 筛选测试行为（根据时间范围）
        test_actions = []
        for action in actions:
            timestamp = action.get("timestamp", "")
            in_range = True
            if test_time_start and timestamp < test_time_start:
                in_range = False
            if test_time_end and timestamp > test_time_end:
                in_range = False
            if in_range:
                test_actions.append(action)
        
        # 检查测试集条件：只检查 context 信息是否充足（不再筛选场景类型）
        test_candidate_actions = [
            action for action in test_actions
            if has_sufficient_context_for_prediction(action)
        ]
        
        # 用户需要满足：测试集足够 + 有历史数据
        if len(test_candidate_actions) >= test_length and len(history_actions) >= min_history_actions:
            valid_users.append(user_id)
    
    print(f"  找到 {len(valid_users)} 个符合条件的用户")
    
    # 重要：对用户ID排序，确保每次运行时遍历顺序一致
    # 这样相同的随机种子会选择相同的用户
    valid_users.sort()
    
    # 选择用户
    if len(valid_users) <= num_users:
        selected = valid_users
        print(f"  可用用户数 ≤ 目标数量，选择全部 {len(selected)} 个用户")
    else:
        # 随机选择
        selected = random.sample(valid_users, num_users)
        print(f"  随机选择 {num_users} 个用户")
    
    # 重要：对选中的用户排序，确保后续处理顺序一致
    # 这样每个用户的 user_seed 会保持一致
    selected.sort()
    
    return selected


def load_user_ids_from_file(file_path: str) -> List[str]:
    """
    从文件加载用户ID列表
    
    支持两种格式：
    1. JSON文件（包含 users 数组，每个元素有 user_id 字段）
    2. 纯文本文件（每行一个用户ID）
    
    Args:
        file_path: 用户ID文件路径
        
    Returns:
        用户ID列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # 尝试解析为 JSON
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "users" in data:
            # 格式：{"users": [{"user_id": "123", ...}, ...]}
            return [u["user_id"] for u in data["users"]]
        elif isinstance(data, list):
            # 格式：["123", "456", ...] 或 [{"user_id": "123"}, ...]
            if data and isinstance(data[0], dict):
                return [u["user_id"] for u in data]
            else:
                return [str(uid) for uid in data]
    except json.JSONDecodeError:
        pass
    
    # 按纯文本解析（每行一个用户ID）
    return [line.strip() for line in content.split('\n') if line.strip()]


def extract_users_from_existing_dataset(
    source_dataset_path: str,
    target_user_ids: List[str] = None,
    num_users: int = None,
    output_path: str = None,
    seed: int = 42,
) -> Dict:
    """
    从已有的实验数据集中提取指定用户的数据
    
    这个函数确保从主实验数据集中提取的用户，其待预测行为与主实验完全一致。
    这对于需要从大规模实验中挑选子集进行快速测试或对比分析非常有用。
    
    使用场景：
    1. 主实验生成了 199 个用户的完整数据集
    2. 现在只想用其中 50 个用户进行快速预测
    3. 确保这 50 个用户的待预测行为与 199 用户实验完全一致
    
    Args:
        source_dataset_path: 已有的主实验数据集路径（JSON文件）
        target_user_ids: 要提取的用户ID列表，如果为 None 则随机选择
        num_users: 如果 target_user_ids 为 None，则随机选择的用户数量
        output_path: 输出文件路径，如果为 None 则自动生成
        seed: 随机种子（用于随机选择用户时）
    
    Returns:
        提取后的实验数据
    """
    # 1. 加载源数据集
    print("=" * 80)
    print("📦 从已有主实验数据集提取用户子集")
    print("=" * 80)
    print(f"\n【源数据集信息】")
    print(f"  路径: {source_dataset_path}")
    
    if not os.path.exists(source_dataset_path):
        raise FileNotFoundError(f"源数据集文件不存在: {source_dataset_path}")
    
    with open(source_dataset_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    source_users = source_data.get("users", [])
    source_metadata = source_data.get("metadata", {})
    
    print(f"  用户数量: {len(source_users)}")
    print(f"  每用户测试行为数: {source_metadata.get('test_length', 'N/A')}")
    print(f"  配置签名: {source_metadata.get('config_signature', 'N/A')}")
    
    # 显示时间范围信息
    if source_metadata.get('history_time_start') or source_metadata.get('history_time_end'):
        print(f"  历史时间范围: {source_metadata.get('history_time_start', '不限')} ~ {source_metadata.get('history_time_end', '不限')}")
    if source_metadata.get('test_time_start') or source_metadata.get('test_time_end'):
        print(f"  测试时间范围: {source_metadata.get('test_time_start', '不限')} ~ {source_metadata.get('test_time_end', '不限')}")
    
    # 构建用户ID到用户数据的映射
    user_data_map = {user["user_id"]: user for user in source_users}
    available_user_ids = list(user_data_map.keys())
    
    # 2. 确定要提取的用户
    print(f"\n【用户提取】")
    if target_user_ids:
        # 使用指定的用户ID
        print(f"  提取方式: 从指定的用户ID列表中提取")
        print(f"  指定用户数: {len(target_user_ids)}")
        
        selected_user_ids = []
        missing_users = []
        for uid in target_user_ids:
            if uid in user_data_map:
                selected_user_ids.append(uid)
            else:
                missing_users.append(uid)
        
        if missing_users:
            print(f"\n  ⚠️ 警告: {len(missing_users)} 个用户ID在源数据集中未找到")
            if len(missing_users) <= 5:
                print(f"     未找到: {missing_users}")
            else:
                print(f"     未找到（前5个）: {missing_users[:5]}...")
            print(f"     这些用户可能不在主实验的 {len(source_users)} 个用户中")
        
        print(f"\n  ✅ 成功匹配: {len(selected_user_ids)} / {len(target_user_ids)} 个用户")
    else:
        # 随机选择用户
        if num_users is None or num_users <= 0:
            raise ValueError("必须指定 target_user_ids 或 num_users")
        
        print(f"  提取方式: 从源数据集中随机选择")
        print(f"  目标数量: {num_users}")
        print(f"  随机种子: {seed}")
        
        random.seed(seed)
        available_user_ids.sort()  # 先排序确保一致性
        
        if num_users >= len(available_user_ids):
            selected_user_ids = available_user_ids
            print(f"\n  ⚠️ 请求数量({num_users}) >= 源数据集用户数({len(available_user_ids)})")
            print(f"  ✅ 将提取全部 {len(selected_user_ids)} 个用户")
        else:
            selected_user_ids = random.sample(available_user_ids, num_users)
            print(f"\n  ✅ 随机选择了 {num_users} 个用户")
    
    # 排序以保持一致性
    selected_user_ids.sort()
    
    # 3. 提取用户数据
    extracted_users = [user_data_map[uid] for uid in selected_user_ids]
    
    # 显示提取的用户样本
    print(f"\n【提取的用户样本】（前5个）")
    for i, uid in enumerate(selected_user_ids[:5], 1):
        user_entry = user_data_map[uid]
        test_count = len(user_entry.get("test_actions", []))
        history_count = user_entry.get("stats", {}).get("base_history_count", "?")
        print(f"  {i}. 用户 {uid}: {test_count} 条待预测行为, {history_count} 条历史")
    if len(selected_user_ids) > 5:
        print(f"  ... 共 {len(selected_user_ids)} 个用户")
    
    # 4. 构建新的 metadata
    # 复制源 metadata，更新用户数量相关信息
    new_metadata = source_metadata.copy()
    new_metadata["num_users"] = len(extracted_users)
    new_metadata["source_dataset"] = source_dataset_path
    new_metadata["source_num_users"] = len(source_users)
    new_metadata["extraction_seed"] = seed
    new_metadata["created_at"] = __import__('datetime').datetime.now().isoformat()
    new_metadata["is_subset_extraction"] = True
    
    # 重新计算统计信息
    action_type_stats = defaultdict(int)
    for user_entry in extracted_users:
        for test_item in user_entry.get("test_actions", []):
            action = test_item["action"]
            action_type = action.get("type", "未知")
            action_type_stats[action_type] += 1
    
    new_metadata["action_type_statistics"] = dict(action_type_stats)
    new_metadata["covered_action_types"] = list(action_type_stats.keys())
    new_metadata["total_test_actions"] = sum(action_type_stats.values())
    
    # 5. 生成新的配置签名
    # 基于提取的用户ID列表生成签名
    user_ids_str = ",".join(sorted(selected_user_ids))
    user_ids_hash = hashlib.md5(user_ids_str.encode('utf-8')).hexdigest()[:8]
    
    # 子集签名 = 源签名 + 用户数 + 用户ID哈希
    source_signature = source_metadata.get("config_signature", "unknown")
    subset_signature = f"sub{source_signature[:4]}_{len(selected_user_ids)}u_{user_ids_hash[:4]}"
    new_metadata["config_signature"] = subset_signature
    new_metadata["source_config_signature"] = source_signature
    
    # 生成描述性目录名
    descriptive_dirname = generate_descriptive_dirname(
        num_users=len(selected_user_ids),
        test_length=source_metadata.get("test_length", 30),
        history_time_start=source_metadata.get("history_time_start"),
        history_time_end=source_metadata.get("history_time_end"),
        test_time_start=source_metadata.get("test_time_start"),
        test_time_end=source_metadata.get("test_time_end"),
        seed=seed,
        signature=subset_signature
    )
    new_metadata["descriptive_dirname"] = descriptive_dirname
    
    # 6. 构建结果数据
    extracted_data = {
        "metadata": new_metadata,
        "users": extracted_users
    }
    
    # 7. 保存结果
    if output_path:
        # 使用上海时区
        shanghai_tz = pytz.timezone('Asia/Shanghai')
        now_shanghai = datetime.now(shanghai_tz)
        run_date = now_shanghai.strftime("%Y-%m-%d")
        
        output_base_dir = os.path.dirname(output_path)
        filename = os.path.basename(output_path)
        timestamp_output_dir = os.path.join(output_base_dir, run_date, descriptive_dirname)
        timestamped_output_path = os.path.join(timestamp_output_dir, filename)
        
        os.makedirs(timestamp_output_dir, exist_ok=True)
        with open(timestamped_output_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n【保存位置】")
        print(f"  {timestamped_output_path}")
        
        # 生成分析报告
        analysis_output_path = os.path.join(timestamp_output_dir, "sampled_actions_analysis.json")
        analyze_sampled_actions(extracted_data, analysis_output_path)
        
        # 输出实际路径供脚本捕获
        print(f"\nACTUAL_OUTPUT_PATH={timestamped_output_path}")
    
    # 8. 打印统计信息
    print("\n" + "=" * 80)
    print("📊 提取结果统计")
    print("=" * 80)
    
    print(f"\n【数据概览】")
    print(f"  源数据集: {len(source_users)} 用户 → 提取后: {len(extracted_users)} 用户")
    print(f"  每用户测试行为: {source_metadata.get('test_length', 'N/A')} 条")
    print(f"  总测试行为数: {sum(action_type_stats.values())} 条")
    
    print(f"\n【场景类型分布】")
    for action_type, count in sorted(action_type_stats.items(), key=lambda x: -x[1]):
        percentage = count / sum(action_type_stats.values()) * 100
        print(f"  {action_type:20s}: {count:5d} 条 ({percentage:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ 用户子集提取完成！")
    print("=" * 80)
    print(f"\n💡 关键保证:")
    print(f"   ✓ 提取的 {len(extracted_users)} 个用户的待预测行为（test_actions）与主实验完全一致")
    print(f"   ✓ 历史数据（base_history, test_time_all_actions）完整保留")
    print(f"   ✓ 可直接用于对比实验，确保公平性")
    print("=" * 80)
    
    return extracted_data


def prepare_fixed_experiment_data(
    input_data_path: str,
    output_path: str,
    num_users: int = 50,
    test_length: int = 20,
    max_history_tokens: int = 32000,
    max_history_days: int = None,
    history_time_start: str = None,
    history_time_end: str = None,
    test_time_start: str = None,
    test_time_end: str = None,
    history_scene_filter: str = None,
    seed: int = 42,
    force_resample: bool = False,
    user_ids_file: str = None,
    specified_user_ids: List[str] = None,
):
    """
    准备固定的实验数据集（均衡采样策略）
    
    采样策略：时间均衡 + 领域均衡 + 高低价值均衡
    只跳过 context 信息不足的行为，确保凑满 test_length 个行为
    
    缓存复用：如果相同配置的数据集已存在，直接复用（除非 force_resample=True）
    
    Args:
        input_data_path: 输入数据文件路径
        output_path: 输出文件路径
        num_users: 选择多少个用户（当指定用户ID时会被忽略）
        test_length: 测试行为数量
        max_history_tokens: 历史行为的最大 Token 数限制（用于控制历史行为数量）
        max_history_days: 只保留近 N 天的历史行为，如果为 None 则不限制天数
        history_time_start: 历史行为时间范围起始（格式：YYYY-MM-DD）
        history_time_end: 历史行为时间范围结束
        test_time_start: 测试集时间范围起始
        test_time_end: 测试集时间范围结束
        history_scene_filter: 历史行为场景过滤（如"视频浏览"、"直播间"等），为空则不过滤
        seed: 随机种子
        force_resample: 是否强制重新采样（忽略缓存）
        user_ids_file: 指定用户ID的文件路径（JSON或纯文本格式）
        specified_user_ids: 直接指定的用户ID列表（优先级高于 user_ids_file）
    """
    # 处理指定用户ID的情况
    use_specified_users = False
    target_user_ids = None
    
    if specified_user_ids:
        target_user_ids = specified_user_ids
        use_specified_users = True
        print(f"📋 使用直接指定的 {len(target_user_ids)} 个用户ID")
    elif user_ids_file:
        if not os.path.exists(user_ids_file):
            raise FileNotFoundError(f"用户ID文件不存在: {user_ids_file}")
        target_user_ids = load_user_ids_from_file(user_ids_file)
        use_specified_users = True
        print(f"📋 从文件加载了 {len(target_user_ids)} 个用户ID: {user_ids_file}")
    
    # 如果指定了用户ID，更新 num_users
    if use_specified_users:
        num_users = len(target_user_ids)
    # 生成配置签名（只基于影响采样的参数）
    # 注意：max_history_tokens, max_history_days, history_scene_filter 不再影响签名
    # 这样改变这些评估参数时，不会重新采样用户和测试数据
    if use_specified_users:
        # 如果指定了用户ID，签名基于用户ID列表的哈希
        user_ids_str = ",".join(sorted(target_user_ids))
        user_ids_hash = hashlib.md5(user_ids_str.encode('utf-8')).hexdigest()[:8]
        config_signature = generate_config_signature(
            num_users=num_users,
            test_length=test_length,
            history_time_start=history_time_start,
            history_time_end=history_time_end,
            test_time_start=test_time_start,
            test_time_end=test_time_end,
            seed=seed,
            input_data_path=f"specified_users_{user_ids_hash}"  # 用用户ID哈希替代路径
        )
    else:
        config_signature = generate_config_signature(
            num_users=num_users,
            test_length=test_length,
            history_time_start=history_time_start,
            history_time_end=history_time_end,
            test_time_start=test_time_start,
            test_time_end=test_time_end,
            seed=seed,
            input_data_path=input_data_path
        )
    
    # 生成描述性目录名（同样只基于影响采样的参数）
    descriptive_dirname = generate_descriptive_dirname(
        num_users=num_users,
        test_length=test_length,
        history_time_start=history_time_start,
        history_time_end=history_time_end,
        test_time_start=test_time_start,
        test_time_end=test_time_end,
        seed=seed,
        signature=config_signature
    )
    
    output_dir = os.path.dirname(output_path)
    
    print("=" * 80)
    print("准备固定实验数据集（均衡采样策略）")
    print("=" * 80)
    print(f"配置参数:")
    print(f"\n【采样参数】（影响用户选择和数据采样，相同参数复用数据集）")
    if use_specified_users:
        print(f"  用户选择: 使用指定的 {num_users} 个用户ID")
        if user_ids_file:
            print(f"  用户ID文件: {user_ids_file}")
    else:
        print(f"  用户数量: {num_users}（随机选择）")
    print(f"  测试集长度: {test_length} 条待预测行为")
    print(f"  采样策略: 时间均衡 + 领域均衡 + 高低价值均衡")
    print(f"  随机种子: {seed}")
    print(f"  配置签名: {config_signature}")
    
    if history_time_start or history_time_end:
        print(f"  历史时间范围: {history_time_start or '不限'} ~ {history_time_end or '不限'}")
    
    if test_time_start or test_time_end:
        print(f"  测试时间范围: {test_time_start or '不限'} ~ {test_time_end or '不限'}")
    
    print(f"\n【评估参数】（不影响采样，只在评估时生效，可安全修改后对比）")
    print(f"  历史行为最大Token数: {max_history_tokens} (评估时控制历史长度)")
    if max_history_days is not None and max_history_days > 0:
        print(f"  历史行为最大天数: {max_history_days} 天")
    if history_scene_filter:
        print(f"  历史场景过滤: {history_scene_filter}")
    else:
        print(f"  历史场景过滤: 无（使用全部场景）")
    
    # 检查是否存在相同配置的数据集（缓存复用）
    if not force_resample:
        existing_dataset = find_existing_dataset(output_dir, config_signature)
        if existing_dataset:
            print(f"\n" + "=" * 80)
            print("✅ 发现已存在相同配置的数据集，直接复用！")
            print("=" * 80)
            print(f"\n已有数据集路径: {existing_dataset}")
            print(f"配置签名匹配: {config_signature}")
            print(f"\n💡 如需强制重新采样，请使用 --force 参数")
            print("=" * 80)
            
            # 输出实际路径供脚本捕获
            print(f"\nACTUAL_OUTPUT_PATH={existing_dataset}")
            
            # 加载已有数据
            with open(existing_dataset, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # 更新 metadata 中的评估参数为当前命令行传入的值
            # 这些参数不影响采样，但会影响评估时的 prompt 构建
            old_max_history_tokens = cached_data.get("metadata", {}).get("max_history_tokens")
            old_max_history_days = cached_data.get("metadata", {}).get("max_history_days")
            old_history_scene_filter = cached_data.get("metadata", {}).get("history_scene_filter")
            
            if "metadata" in cached_data:
                cached_data["metadata"]["max_history_tokens"] = max_history_tokens
                cached_data["metadata"]["max_history_days"] = max_history_days
                cached_data["metadata"]["history_scene_filter"] = history_scene_filter
            
            # 如果评估参数有变化，打印提示
            params_changed = []
            if old_max_history_tokens != max_history_tokens:
                params_changed.append(f"max_history_tokens: {old_max_history_tokens} → {max_history_tokens}")
            if old_max_history_days != max_history_days:
                params_changed.append(f"max_history_days: {old_max_history_days} → {max_history_days}")
            if old_history_scene_filter != history_scene_filter:
                params_changed.append(f"history_scene_filter: {old_history_scene_filter} → {history_scene_filter}")
            
            if params_changed:
                print(f"\n📝 评估参数已更新（不影响采样数据）:")
                for change in params_changed:
                    print(f"   - {change}")
            
            return cached_data
    else:
        print(f"  强制重新采样: 是")
    
    print(f"\n数据分割说明:")
    if history_time_start or history_time_end or test_time_start or test_time_end:
        print(f"  - 历史行为: 从历史时间范围内筛选，根据 Token 数限制选取")
        print(f"  - 测试集: 从测试时间范围内均衡采样 {test_length} 条（跳过 context 信息不足的行为）")
    else:
        print(f"  从用户最新的行为往前推:")
        print(f"  - 测试集: 均衡采样 {test_length} 条（跳过 context 信息不足的行为）")
        print(f"  - 历史上下文: 从测试集之前筛选，根据 Token 数限制选取")
    
    # 1. 加载数据
    if use_specified_users:
        # 如果指定了用户ID，直接加载这些用户的数据
        print(f"\n加载指定的 {len(target_user_ids)} 个用户数据...")
        all_users = load_user_data(
            input_data_path, 
            min_actions=0,  # 不过滤最小行为数
            target_users=0  # 加载全部数据（会在下面过滤）
        )
        
        # 过滤出指定的用户
        specified_users = {}
        missing_users = []
        for uid in target_user_ids:
            if uid in all_users:
                specified_users[uid] = all_users[uid]
            else:
                missing_users.append(uid)
        
        if missing_users:
            print(f"  ⚠️ 警告: {len(missing_users)} 个用户ID在数据中未找到")
            if len(missing_users) <= 10:
                print(f"     未找到的用户ID: {missing_users}")
            else:
                print(f"     未找到的用户ID（前10个）: {missing_users[:10]}...")
        
        all_users = specified_users
        print(f"  成功加载 {len(all_users)} 个指定用户的数据")
        
        # 直接使用指定的用户ID（已排序）
        selected_user_ids = sorted([uid for uid in target_user_ids if uid in all_users])
        print(f"\n使用指定的 {len(selected_user_ids)} 个用户")
    else:
        # 常规模式：加载数据并随机选择用户
        min_required = test_length * 2  # 保守估计，加载更多数据
        all_users = load_user_data(
            input_data_path, 
            min_actions=min_required,
            target_users=num_users * 3  # 多加载一些用户以便筛选
        )
        
        # 2. 选择用户
        # 注意：history_scene_filter 不再影响用户选择，它只在评估时过滤历史行为
        selected_user_ids = select_users(
            all_users,
            num_users=num_users,
            test_length=test_length,
            history_time_start=history_time_start,
            history_time_end=history_time_end,
            test_time_start=test_time_start,
            test_time_end=test_time_end,
            min_history_actions=1,  # 只要有历史数据即可，具体数量由 token 限制控制
            seed=seed
        )
    
    # 判断是否使用分离时间范围模式
    use_separated_time_range = (history_time_start or history_time_end or test_time_start or test_time_end)
    
    # 3. 准备实验数据
    print(f"\n准备实验数据...")
    experiment_data = {
        "metadata": {
            "num_users": len(selected_user_ids),
            "test_length": test_length,
            "max_history_tokens": max_history_tokens,
            "max_history_days": max_history_days,  # 只保留近 N 天的历史行为
            "history_time_start": history_time_start,
            "history_time_end": history_time_end,
            "test_time_start": test_time_start,
            "test_time_end": test_time_end,
            "history_scene_filter": history_scene_filter,  # 历史行为场景过滤
            "sampling_strategy": "balanced",  # 均衡采样：时间+领域+价值
            "use_separated_time_range": use_separated_time_range,  # 标记是否使用分离时间范围
            "use_specified_users": use_specified_users,  # 是否使用指定的用户ID
            "user_ids_file": user_ids_file if use_specified_users else None,  # 用户ID文件路径
            "seed": seed,
            "created_at": __import__('datetime').datetime.now().isoformat(),
        },
        "users": []
    }
    
    # 统计场景类型
    action_type_stats = defaultdict(int)
    
    # 新增：用于统计每个用户历史行为的token数和时间分布
    user_history_tokens = []
    all_history_timestamps = []
    all_test_timestamps = []
    
    # 用于保存每个用户的采样行为文件
    user_sampled_actions_dir = None
    
    for user_idx, user_id in enumerate(selected_user_ids):
        user_data = all_users[user_id]
        # 为每个用户生成固定的随机种子（基于全局种子和用户ID的稳定哈希）
        # 这样确保每次运行时，相同的用户ID会得到相同的采样结果
        # 无论用户在列表中的位置如何变化
        user_seed = seed + get_stable_user_seed_offset(user_id)
        
        # 使用均衡采样的分割函数
        # 注意：max_history_tokens 和 history_scene_filter 不再传递给采样函数
        # 它们只在评估时影响 prompt 构建，不影响数据采样
        result_data, test_actions_with_index, base_history_tokens, skipped_actions_info = split_user_actions_by_tokens(
            user_data, 
            test_length,
            history_time_start=history_time_start,
            history_time_end=history_time_end,
            test_time_start=test_time_start,
            test_time_end=test_time_end,
            seed=user_seed  # 传入用户专属随机种子
        )
        
        # 收集基础历史行为的时间戳用于分布统计
        for action in result_data["base_history"]:
            timestamp = action.get("timestamp", "未知时间")
            if timestamp and timestamp != "未知时间":
                all_history_timestamps.append(timestamp)
        
        user_history_tokens.append(base_history_tokens)
        
        # 收集测试集时间戳
        for test_item in test_actions_with_index:
            action = test_item["action"]
            timestamp = action.get("timestamp", "")
            if timestamp:
                all_test_timestamps.append(timestamp)
        
        # 构建测试行为列表（包含 test_time_index 用于滚动预测）
        test_actions_data = []
        for test_item in test_actions_with_index:
            action = test_item["action"]
            test_time_index = test_item["test_time_index"]  # 在测试时间范围内的索引
            test_actions_data.append({
                "action": action,
                "test_time_index": test_time_index,  # 用于确定该测试行为之前的行为
            })
            
            # 统计场景类型
            action_type = action.get("type", "未知")
            action_type_stats[action_type] += 1
        
        experiment_data["users"].append({
            "user_id": user_id,
            "user_profile": result_data["user_profile"],
            "base_history": result_data["base_history"],  # 基础历史（如9月份的行为）
            "test_time_all_actions": result_data["test_time_all_actions"],  # 测试时间范围内所有行为
            "test_actions": test_actions_data,  # 包含 test_time_index
            "skipped_actions": skipped_actions_info,  # 被跳过的行为信息
            "stats": {
                "base_history_count": len(result_data["base_history"]),
                "test_time_all_actions_count": len(result_data["test_time_all_actions"]),
                "test_count": len(test_actions_with_index),
                "skipped_count": len(skipped_actions_info),
                "base_history_tokens": base_history_tokens
            }
        })
    
    # 4. 添加统计信息
    experiment_data["metadata"]["action_type_statistics"] = dict(action_type_stats)
    experiment_data["metadata"]["covered_action_types"] = list(action_type_stats.keys())
    experiment_data["metadata"]["total_test_actions"] = sum(action_type_stats.values())
    
    # 新增：用户历史行为token数统计
    if user_history_tokens:
        import statistics
        experiment_data["metadata"]["history_token_statistics"] = {
            "avg_tokens_per_user": sum(user_history_tokens) / len(user_history_tokens),
            "min_tokens": min(user_history_tokens),
            "max_tokens": max(user_history_tokens),
            "median_tokens": statistics.median(user_history_tokens),
            "total_tokens": sum(user_history_tokens),
        }
    
    # 新增：时间分布区间统计
    time_distribution = {}
    if all_history_timestamps:
        sorted_history = sorted(all_history_timestamps)
        time_distribution["history"] = {
            "earliest": sorted_history[0],
            "latest": sorted_history[-1],
            "count": len(sorted_history)
        }
    if all_test_timestamps:
        sorted_test = sorted(all_test_timestamps)
        time_distribution["test"] = {
            "earliest": sorted_test[0],
            "latest": sorted_test[-1],
            "count": len(sorted_test)
        }
    if time_distribution:
        experiment_data["metadata"]["time_distribution"] = time_distribution
    
    # 5. 保存（按日期和描述性目录名分类）
    # 使用上海时区
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    now_shanghai = datetime.now(shanghai_tz)
    run_date = now_shanghai.strftime("%Y-%m-%d")
    
    # 添加配置签名到 metadata
    experiment_data["metadata"]["config_signature"] = config_signature
    experiment_data["metadata"]["descriptive_dirname"] = descriptive_dirname
    
    # 修改输出路径，添加日期/描述性目录名两级子目录
    # 目录名格式：200u_30t_32k_h~0930_t1001~1130_s42_abc12345
    output_base_dir = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    timestamp_output_dir = os.path.join(output_base_dir, run_date, descriptive_dirname)
    timestamped_output_path = os.path.join(timestamp_output_dir, filename)
    
    os.makedirs(timestamp_output_dir, exist_ok=True)
    with open(timestamped_output_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n实验数据已保存到: {timestamped_output_path}")
    
    # 5.1 为每个用户保存单独的采样行为文件
    user_sampled_actions_dir = os.path.join(timestamp_output_dir, "sampled_actions_distribution")
    os.makedirs(user_sampled_actions_dir, exist_ok=True)
    
    print(f"\n保存每个用户的采样行为文件...")
    for user_entry in experiment_data["users"]:
        user_id = user_entry["user_id"]
        user_sampled_file = os.path.join(user_sampled_actions_dir, f"{user_id}_sampled_actions.json")
        
        # 构建用户采样行为的详细信息
        user_sampled_data = {
            "user_id": user_id,
            "test_length": test_length,
            "actual_sampled_count": len(user_entry["test_actions"]),
            "skipped_count": user_entry["stats"]["skipped_count"],
            "sampled_actions": [],
            "skipped_actions": user_entry.get("skipped_actions", [])
        }
        
        # 添加采样的行为详情
        for i, test_item in enumerate(user_entry["test_actions"], 1):
            action = test_item["action"]
            user_sampled_data["sampled_actions"].append({
                "序号": i,
                "时间": action.get("timestamp", ""),
                "场景类型": action.get("type", ""),
                "内容摘要": _get_action_summary(action),
                "完整行为": action
            })
        
        # 保存用户采样文件
        with open(user_sampled_file, 'w', encoding='utf-8') as f:
            json.dump(user_sampled_data, f, ensure_ascii=False, indent=2)
    
    print(f"  用户采样行为文件已保存到: {user_sampled_actions_dir}")
    print(f"  共 {len(experiment_data['users'])} 个用户文件")
    
    # 5.2 生成采样行为分析报告
    analysis_output_path = os.path.join(timestamp_output_dir, "sampled_actions_analysis.json")
    analyze_sampled_actions(experiment_data, analysis_output_path)
    
    # 5.3 生成采样行为可视化图表（时间分布、场景分布、行为分布）
    try:
        from plot.plot_sampled_actions import generate_sampled_actions_charts
        generate_sampled_actions_charts(experiment_data, user_sampled_actions_dir)
    except ImportError:
        print("  ⚠️  plot_sampled_actions.py not found, skipping sampled actions chart generation")
    except Exception as e:
        print(f"  ⚠️  Sampled actions chart generation failed: {e}")
    
    # 5.4 生成历史行为分布可视化图表（base_history + test_time_all_actions）
    history_distribution_dir = os.path.join(timestamp_output_dir, "history_distribution")
    try:
        from plot.plot_history_distribution import generate_history_charts
        generate_history_charts(experiment_data, history_distribution_dir)
    except ImportError:
        print("  ⚠️  plot_history_distribution.py not found, skipping history distribution chart generation")
    except Exception as e:
        print(f"  ⚠️  History distribution chart generation failed: {e}")
    
    # 6. 生成测试数据分布统计（用于后续绘制PR-AUC曲线）
    distribution_output_path = timestamped_output_path.replace(".json", "_test_distribution.json")
    analyze_test_data_distribution(experiment_data, distribution_output_path)
    
    # 7. 生成统计报告
    print("\n" + "=" * 80)
    print("实验数据统计")
    print("=" * 80)
    
    total_test_actions = sum(action_type_stats.values())
    
    print(f"\n基本信息:")
    print(f"  用户数量: {len(selected_user_ids)}")
    print(f"  总测试行为数: {total_test_actions}")
    print(f"  场景类型数: {len(action_type_stats)}")
    
    if len(selected_user_ids) > 0:
        print(f"  平均每用户: {total_test_actions / len(selected_user_ids):.1f} 个测试行为")
    
    # 显示基础历史Token统计
    if user_history_tokens:
        import statistics
        avg_tokens = sum(user_history_tokens) / len(user_history_tokens)
        median_tokens = statistics.median(user_history_tokens)
        print(f"\n基础历史Token统计:")
        print(f"  每用户平均Token数: {avg_tokens:,.0f}")
        print(f"  中位数Token数: {median_tokens:,.0f}")
        print(f"  最小Token数: {min(user_history_tokens):,}")
        print(f"  最大Token数: {max(user_history_tokens):,}")
        print(f"  总Token数: {sum(user_history_tokens):,}")
    
    # 显示时间分布区间
    if all_history_timestamps or all_test_timestamps:
        print(f"\n时间分布区间:")
        if all_history_timestamps:
            sorted_history = sorted(all_history_timestamps)
            print(f"  基础历史时间范围:")
            print(f"    最早: {sorted_history[0]}")
            print(f"    最晚: {sorted_history[-1]}")
            print(f"    样本数: {len(sorted_history)}")
        if all_test_timestamps:
            sorted_test = sorted(all_test_timestamps)
            print(f"  测试集时间范围:")
            print(f"    最早: {sorted_test[0]}")
            print(f"    最晚: {sorted_test[-1]}")
            print(f"    样本数: {len(sorted_test)}")
    
    print(f"\n场景类型分布:")
    for action_type, count in sorted(action_type_stats.items(), key=lambda x: -x[1]):
        percentage = count / sum(action_type_stats.values()) * 100
        print(f"  {action_type:20s}: {count:5d} 个样本 ({percentage:5.1f}%)")
    
    # 显示用户样本（前3个）
    print(f"\n用户样本（前3个）:")
    for i, user in enumerate(experiment_data["users"][:3], 1):
        print(f"  {i}. 用户 {user['user_id']}")
        print(f"     基础历史: {user['stats']['base_history_count']} 条 ({user['stats']['base_history_tokens']:,} tokens)")
        print(f"     测试时间范围行为: {user['stats']['test_time_all_actions_count']} 条")
        print(f"     待预测行为: {user['stats']['test_count']} 条")
    
    print("\n" + "=" * 80)
    print("✅ 实验数据准备完成！")
    print("=" * 80)
    print(f"\n后续使用:")
    print(f"  python evaluator.py --use-fixed-data {timestamped_output_path}")
    print("=" * 80)
    
    # 输出实际路径供脚本捕获（特殊格式，方便脚本解析）
    print(f"\nACTUAL_OUTPUT_PATH={timestamped_output_path}")
    
    return experiment_data


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="准备固定的实验数据集（均衡采样策略）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：

  # 基本用法：选择10个用户，20条测试，历史由token数控制
  python prepare_experiment_data.py --num-users 10 --test-length 20
  
  # 指定历史行为最大Token数（128K）
  python prepare_experiment_data.py --num-users 10 --max-history-tokens 128000
  
  # 分离的时间范围：历史从9月，测试从10-11月
  python prepare_experiment_data.py \\
    --num-users 10 \\
    --test-length 30 \\
    --history-time-start 2025-09-01 \\
    --history-time-end 2025-09-30 \\
    --test-time-start 2025-10-01 \\
    --test-time-end 2025-11-30 \\
    --output dataset/experiment_data.json

  # 使用指定的用户ID文件（跳过随机选择，直接使用文件中的用户）
  python prepare_experiment_data.py \\
    --user-ids-file user_stats_filtered_output.json \\
    --test-length 30 \\
    --output dataset/specified_users.json

  # 【新功能】从已有主实验数据集中提取用户子集
  # 确保提取的用户待预测行为与主实验完全一致
  python prepare_experiment_data.py \\
    --source-dataset dataset/2026-02-03/199u_30t_h~0930_t1001~1130_s42_882d82f7/experiment_data.json \\
    --extract-num-users 50 \\
    --output dataset/experiment_data.json

  # 从主实验中提取指定的用户ID
  python prepare_experiment_data.py \\
    --source-dataset dataset/2026-02-03/199u_30t_h~0930_t1001~1130_s42_882d82f7/experiment_data.json \\
    --user-ids-file my_selected_users.txt \\
    --output dataset/experiment_data.json

用户ID文件格式：
  支持以下格式：
  1. JSON格式（包含 users 数组）：
     {"users": [{"user_id": "123", ...}, {"user_id": "456", ...}]}
  2. JSON数组格式：
     ["123", "456", "789"]
  3. 纯文本格式（每行一个用户ID）：
     123
     456
     789

采样策略说明：
  采用均衡采样策略（时间均衡 + 领域均衡 + 高低价值均衡）：
  1. 时间均衡：将测试时间范围按时间切分为M个桶，保证时间维度的均衡性
  2. 领域均衡：轮询各个领域（视频浏览、直播间等），保证各领域样本数量尽可能一致
  3. 价值均衡：在选定领域内，以50%概率采样高价值（如点赞、购买）或低价值Item
  4. 只跳过 context 信息不足的行为（如无标题的视频），确保凑满 test_length 个

数据分割逻辑：
  分离时间范围模式：
  - 历史行为：从 history-time-start ~ history-time-end 范围内采集
  - 测试集：从 test-time-start ~ test-time-end 范围内均衡采样
  
  传统模式（不指定时间范围）：
  - 从用户最新的行为往前推
  - 均衡采样 test_length 条行为作为测试集
  - 从测试集之前的行为中按 Token 数限制选取历史上下文

子集提取模式（使用 --source-dataset）：
  - 从已有的主实验数据集中提取用户子集
  - 确保提取的用户待预测行为与主实验完全一致
  - 适用于快速测试或对比分析
        """
    )
    
    # 子集提取参数
    parser.add_argument("--source-dataset", type=str, default=None,
                       help="从已有数据集提取用户子集的源文件路径（使用此参数时进入提取模式）")
    parser.add_argument("--extract-num-users", type=int, default=None,
                       help="从源数据集中随机提取的用户数量（与 --source-dataset 配合使用）")
    
    # 常规参数
    parser.add_argument("--num-users", type=int, default=50, 
                       help="选择多少个用户（默认50，使用--user-ids-file时忽略此参数）")
    parser.add_argument("--user-ids-file", type=str, default=None,
                       help="指定用户ID的文件路径（JSON或纯文本格式），使用此参数时跳过随机用户选择")
    parser.add_argument("--test-length", type=int, default=20,
                       help="测试行为数量（默认20）")
    parser.add_argument("--max-history-tokens", type=int, default=MAX_HISTORY_TOKENS,
                       help=f"历史行为的最大Token数限制（默认{MAX_HISTORY_TOKENS}）")
    parser.add_argument("--max-history-days", type=int, default=None,
                       help="只保留近 N 天的历史行为（默认不限制，即使用全部历史）")
    parser.add_argument("--history-time-start", type=str, default=None,
                       help="历史行为时间范围起始（格式：YYYY-MM-DD）")
    parser.add_argument("--history-time-end", type=str, default=None,
                       help="历史行为时间范围结束（格式：YYYY-MM-DD）")
    parser.add_argument("--test-time-start", type=str, default=None,
                       help="测试集时间范围起始（格式：YYYY-MM-DD）")
    parser.add_argument("--test-time-end", type=str, default=None,
                       help="测试集时间范围结束（格式：YYYY-MM-DD）")
    parser.add_argument("--history-scene-filter", type=str, default=None,
                       help="历史行为场景过滤，只选择指定场景的历史行为（可选值：视频浏览、直播间、商城购物、广告推荐、电商客服对话）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子（默认42）")
    parser.add_argument("--output", type=str, default="./work_data/experiment_data.json",
                       help="输出文件路径（默认dataset/experiment_data.json）")
    parser.add_argument("--force", action="store_true",
                       help="强制重新采样，即使存在相同配置的数据集")
    
    args = parser.parse_args()
    
    # 根据参数决定使用哪种模式
    if args.source_dataset:
        # 子集提取模式：从已有数据集中提取用户
        target_user_ids = None
        if args.user_ids_file:
            if not os.path.exists(args.user_ids_file):
                raise FileNotFoundError(f"用户ID文件不存在: {args.user_ids_file}")
            target_user_ids = load_user_ids_from_file(args.user_ids_file)
            print(f"📋 从文件加载了 {len(target_user_ids)} 个用户ID: {args.user_ids_file}")
        
        extract_users_from_existing_dataset(
            source_dataset_path=args.source_dataset,
            target_user_ids=target_user_ids,
            num_users=args.extract_num_users,
            output_path=args.output,
            seed=args.seed,
        )
    else:
        # 常规模式：生成新的实验数据集
        prepare_fixed_experiment_data(
            "./work_data/user_jsons",
            args.output,
            num_users=args.num_users,
            test_length=args.test_length,
            max_history_tokens=args.max_history_tokens,
            max_history_days=args.max_history_days,
            history_time_start=args.history_time_start,
            history_time_end=args.history_time_end,
            test_time_start=args.test_time_start,
            test_time_end=args.test_time_end,
            history_scene_filter=args.history_scene_filter,
            seed=args.seed,
            force_resample=args.force,
            user_ids_file=args.user_ids_file,
        )


if __name__ == "__main__":
    main()

