"""
Model Caller: Unified interface for calling various LLM APIs
"""
import json
import re
import time
import os
import math
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Union
import requests
from openai import OpenAI


def stable_hash(key: str) -> int:
    """生成稳定的哈希值（跨运行一致）"""
    return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)


def get_endpoints(model_config: Dict) -> List[Dict]:
    """
    从模型配置中提取端点列表
    
    Returns:
        [{"url": str, "max_workers": int}, ...]
    """
    endpoints = model_config.get("endpoints")
    base_urls = model_config.get("base_urls")
    base_url = model_config.get("base_url")
    default_workers = model_config.get("max_workers", 10)
    
    result = []
    if endpoints:
        for ep in endpoints:
            if isinstance(ep, dict):
                result.append({
                    "url": ep["url"],
                    "max_workers": ep.get("max_workers", default_workers)
                })
            elif isinstance(ep, str):
                result.append({"url": ep, "max_workers": default_workers})
    elif base_urls:
        for url in base_urls:
            result.append({"url": url, "max_workers": default_workers})
    elif base_url:
        result.append({"url": base_url, "max_workers": default_workers})
    
    return result


def assign_users_to_endpoints(user_ids: List[str], endpoints: List[Dict]) -> Dict[str, List[str]]:
    """
    将用户分配到各端点（基于稳定哈希，按 max_workers 加权分配）
    
    并发数大的端点会分配到更多用户，同一用户始终分配到同一端点（利用前缀缓存）
    
    Returns:
        {endpoint_url: [user_id1, user_id2, ...], ...}
    """
    assignment = {ep["url"]: [] for ep in endpoints}
    
    if not endpoints or not user_ids:
        return assignment
    
    # 计算各端点的权重（基于 max_workers）
    total_workers = sum(ep["max_workers"] for ep in endpoints)
    
    # 使用整数计算避免浮点精度问题
    # 构建累积权重列表（使用整数表示）
    cumulative_workers = []
    cumsum = 0
    for ep in endpoints:
        cumsum += ep["max_workers"]
        cumulative_workers.append(cumsum)
    
    for user_id in user_ids:
        # 用哈希值映射到 [0, total_workers) 区间（整数运算）
        h = stable_hash(user_id) % total_workers
        
        # 找到对应的端点（落在哪个区间）
        for i, threshold in enumerate(cumulative_workers):
            if h < threshold:
                assignment[endpoints[i]["url"]].append(user_id)
                break
        else:
            # 兜底：分配到最后一个端点（理论上不会到这里）
            assignment[endpoints[-1]["url"]].append(user_id)
    
    return assignment


class DynamicTaskQueue:
    """
    动态任务队列：实现跨端点的工作窃取（Work Stealing）负载均衡
    
    设计目标：
    1. 优先保持同一用户的任务在同一端点执行（利用前缀缓存）
    2. 当某端点任务耗尽时，允许从其他端点"窃取"任务
    3. 确保所有端点始终有活可干，避免资源浪费
    
    工作原理：
    - 每个端点维护一个本地任务队列（初始分配的任务）
    - 当本地队列为空时，从全局待处理队列获取任务
    - 全局队列中的任务可被任意空闲端点获取
    """
    
    def __init__(self, endpoints: List[Dict], tasks: List[Dict], user_tasks: Dict[str, List[Dict]]):
        """
        初始化动态任务队列
        
        Args:
            endpoints: 端点列表 [{"url": str, "max_workers": int}, ...]
            tasks: 所有任务列表
            user_tasks: 按用户分组的任务 {user_id: [task1, task2, ...], ...}
        """
        self.endpoints = endpoints
        self.user_tasks = user_tasks
        self.lock = threading.Lock()
        
        # 初始分配：按哈希将用户分配到端点（保持局部性）
        user_ids = list(user_tasks.keys())
        initial_assignment = assign_users_to_endpoints(user_ids, endpoints)
        
        # 每个端点的本地任务队列（优先处理）
        # 使用 deque 以支持高效的双端操作
        from collections import deque
        self.endpoint_queues: Dict[str, deque] = {}
        for ep in endpoints:
            url = ep["url"]
            users = initial_assignment.get(url, [])
            # 将分配给该端点的用户的所有任务放入本地队列
            local_tasks = []
            for uid in users:
                local_tasks.extend(user_tasks.get(uid, []))
            self.endpoint_queues[url] = deque(local_tasks)
        
        # 统计信息
        self.total_tasks = len(tasks)
        self.completed_count = 0
        self.stolen_count = 0  # 被窃取的任务数
        self.endpoint_stats = {ep["url"]: {"local": 0, "stolen": 0} for ep in endpoints}
        
        # 打印初始分配情况
        print(f"\n📊 动态负载均衡初始化完成:")
        for ep in endpoints:
            url = ep["url"]
            queue_size = len(self.endpoint_queues[url])
            print(f"   [{url}] 初始任务: {queue_size}, 并发: {ep['max_workers']}")
    
    def get_task(self, endpoint_url: str) -> Optional[Dict]:
        """
        为指定端点获取一个任务
        
        优先级：
        1. 从本地队列获取（保持前缀缓存效果）
        2. 从其他端点队列窃取（避免空闲）
        
        Args:
            endpoint_url: 请求任务的端点 URL
            
        Returns:
            任务字典，如果没有可用任务则返回 None
        """
        with self.lock:
            # 1. 优先从本地队列获取
            local_queue = self.endpoint_queues.get(endpoint_url)
            if local_queue and len(local_queue) > 0:
                task = local_queue.popleft()
                self.endpoint_stats[endpoint_url]["local"] += 1
                return task
            
            # 2. 本地队列为空，尝试从其他端点窃取
            # 优先从任务最多的端点窃取（负载均衡）
            max_queue_url = None
            max_queue_size = 0
            
            for ep in self.endpoints:
                url = ep["url"]
                if url != endpoint_url:
                    queue = self.endpoint_queues[url]
                    if len(queue) > max_queue_size:
                        max_queue_size = len(queue)
                        max_queue_url = url
            
            # 如果找到有任务的队列，窃取一个任务
            if max_queue_url and max_queue_size > 0:
                task = self.endpoint_queues[max_queue_url].popleft()
                self.stolen_count += 1
                self.endpoint_stats[endpoint_url]["stolen"] += 1
                return task
            
            # 3. 所有队列都空了，没有任务可做
            return None
    
    def mark_completed(self):
        """标记一个任务完成"""
        with self.lock:
            self.completed_count += 1
    
    def get_remaining_count(self) -> int:
        """获取剩余未完成的任务数"""
        with self.lock:
            total_in_queues = sum(len(q) for q in self.endpoint_queues.values())
            return total_in_queues
    
    def is_all_done(self) -> bool:
        """检查是否所有任务都已分配完成"""
        with self.lock:
            return all(len(q) == 0 for q in self.endpoint_queues.values())
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            return {
                "total_tasks": self.total_tasks,
                "completed": self.completed_count,
                "stolen_count": self.stolen_count,
                "stolen_rate": self.stolen_count / self.total_tasks * 100 if self.total_tasks > 0 else 0,
                "endpoint_stats": dict(self.endpoint_stats),
                "remaining_per_endpoint": {url: len(q) for url, q in self.endpoint_queues.items()}
            }


class ModelCaller:
    """统一的模型调用接口"""
    
    
    def __init__(self, model_config: Dict, base_url_override: str = None, error_log_dir: str = None):
        """
        初始化模型调用器
        
        Args:
            model_config: 模型配置字典
            base_url_override: 覆盖配置中的 base_url（用于多端点并发时指定具体端点）
            error_log_dir: 错误日志保存目录，如果为 None 则不保存
        """
        if not model_config:
            raise ValueError("必须提供 model_config 参数")
        
        self.config = model_config
        self.model_type = model_config.get("type", "openai_compatible")
        self.model_name = model_config.get("name", model_config.get("model", "unknown"))
        self.use_logprobs = model_config.get("use_logprobs", False)
        self.use_cache_control = model_config.get("use_cache_control", False)
        self.rpm = model_config.get("rpm", 0)  # 每分钟最大请求数（0表示不限制）
        self.tpm = model_config.get("tpm", 0)  # 每分钟最大 token 数（0表示不限制）
        self.error_log_dir = error_log_dir
        
        # 确定使用的 base_url
        if base_url_override:
            self.base_url = base_url_override
        else:
            endpoints = get_endpoints(model_config)
            self.base_url = endpoints[0]["url"] if endpoints else None
        
        self._first_call_logged = False
        self._direct_mapping_debug_done = False
        self._request_lock = threading.Lock()  # 用于限流的锁
        self._error_log_lock = threading.Lock() # 用于写错误日志的锁
        self._request_timestamps = []  # 记录过去一分钟内的请求时间戳（滑动窗口）
        self._token_records = []  # 记录过去一分钟内的 token 使用量 [(timestamp, tokens), ...]

    def set_error_log_dir(self, error_log_dir: str):
        """设置错误日志目录"""
        self.error_log_dir = error_log_dir

    def _log_error(self, prompt: str, raw_output: str, error_msg: str):
        """记录错误日志"""
        if not self.error_log_dir:
            return
            
        try:
            log_file = os.path.join(self.error_log_dir, "failed_prompts.jsonl")
            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": self.model_name,
                "error": error_msg,
                "raw_output": raw_output,
                "prompt": prompt
            }
            
            # 使用锁确保多线程写文件安全
            with self._error_log_lock:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"FAILED to write error log: {e}")

    def _wait_for_rate_limit(self):
        """等待限流（检查 RPM 和 TPM）"""
        if self.rpm <= 0 and self.tpm <= 0:
            return
        
        with self._request_lock:
            current_time = time.time()
            window_start = current_time - 60  # 一分钟的滑动窗口
            
            # 清理窗口外的旧请求记录
            self._request_timestamps = [t for t in self._request_timestamps if t > window_start]
            self._token_records = [(t, tokens) for t, tokens in self._token_records if t > window_start]
            
            # RPM 限流：检查请求数
            if self.rpm > 0:
                while len(self._request_timestamps) >= self.rpm:
                    # 计算需要等待的时间：最早的请求过期时间
                    oldest_request = self._request_timestamps[0]
                    wait_time = oldest_request + 60 - current_time + 0.1
                    if wait_time > 0:
                        time.sleep(wait_time)
                    
                    # 重新获取当前时间并清理
                    current_time = time.time()
                    window_start = current_time - 60
                    self._request_timestamps = [t for t in self._request_timestamps if t > window_start]
                    self._token_records = [(t, tokens) for t, tokens in self._token_records if t > window_start]
            
            # TPM 限流：检查 token 使用量
            if self.tpm > 0:
                current_tokens = sum(tokens for _, tokens in self._token_records)
                while current_tokens >= self.tpm and self._token_records:
                    # 计算需要等待的时间：最早的 token 记录过期时间
                    oldest_record = self._token_records[0][0]
                    wait_time = oldest_record + 60 - current_time + 0.1
                    if wait_time > 0:
                        time.sleep(wait_time)
                    
                    # 重新获取当前时间并清理
                    current_time = time.time()
                    window_start = current_time - 60
                    self._request_timestamps = [t for t in self._request_timestamps if t > window_start]
                    self._token_records = [(t, tokens) for t, tokens in self._token_records if t > window_start]
                    current_tokens = sum(tokens for _, tokens in self._token_records)
            
            # 记录本次请求时间
            self._request_timestamps.append(current_time)
    
    def _record_token_usage(self, tokens: int):
        """记录本次请求的 token 使用量（用于 TPM 限流）"""
        if self.tpm <= 0 or tokens <= 0:
            return
        
        with self._request_lock:
            self._token_records.append((time.time(), tokens))

    
    def call(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        调用模型进行预测
        
        Args:
            prompt: 输入的prompt
            max_retries: 最大重试次数
            
        Returns:
            模型的响应文本，如果失败则返回None
        """
        for attempt in range(max_retries):
            try:
                # 支持新旧两种类型标识
                if self.model_type in ["openai", "deepseek", "qwen", "openai_compatible"]:
                    response_data = self._call_openai_compatible(prompt)
                    response = response_data["content"]  # 提取内容
                elif self.model_type == "anthropic":
                    response = self._call_anthropic(prompt)
                elif self.model_type == "aws_claude":
                    response = self._call_aws_claude(prompt)
                else:
                    raise ValueError(f"不支持的模型类型: {self.model_type}")
                
                return response
            
            except Exception as e:
                print(f"[{self.model_name}] 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return None
        
        return None
    
    def _call_openai_compatible(self, prompt: str, enable_logprobs: bool = False, max_tokens: int = None) -> Union[str, Dict]:
        """
        调用OpenAI兼容的API（使用 openai SDK）
        
        Args:
            prompt: 输入的prompt
            enable_logprobs: 是否启用 logprobs 返回（用于二分类计算）
            max_tokens: 最大生成 token 数，如果为 None 则不限制（二分类时会覆盖为 1）
        
        Returns:
            如果 enable_logprobs=False，返回字符串响应
            如果 enable_logprobs=True，返回字典 {"content": str, "logprobs": logprobs_data, "usage": usage_data}
        """
        # 创建 OpenAI 客户端
        client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.base_url,
            timeout=3600
        )
        
        # 构建 message content
        if self.use_cache_control:
            message_content = [
                {
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        else:
            message_content = prompt
        
        # 构建请求参数
        request_params = {
            "model": self.config["model"],
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            # "temperature": self.config.get("temperature", 0.1),
        }
        
        # 如果需要 logprobs，添加相关参数（二分类模式），针对开源模型
        if enable_logprobs:
            request_params["logprobs"] = True
            request_params["top_logprobs"] = 5
        
        # 检测是否为推理模型（GPT-5、GPT-5.2、o1、o3 等）
        # 这些模型不支持 max_tokens，需要使用 max_completion_tokens，并禁用思考模式
        model_name_lower = self.model_name.lower()
        is_reasoning_model = self.config.get("is_reasoning_model", False) or any(
            name in model_name_lower for name in ["gpt-5", "gpt5", "o1", "o3", "reasoning"]
        )
        
        # 设置 max_tokens / max_completion_tokens（由调用方根据任务类型指定）
        if max_tokens is not None:
            if is_reasoning_model:
                # 推理模型使用 max_completion_tokens
                request_params["max_completion_tokens"] = max_tokens
            else:
                request_params["max_tokens"] = max_tokens
        
        # 推理模型需要禁用/最小化思考模式
        if is_reasoning_model:
            request_params["reasoning_effort"] = self.config.get("reasoning_effort", "minimal")
        
        # 如果配置了 extra_body，添加到 extra_body 参数中
        extra_body = self.config.get("extra_body")
        if extra_body:
            request_params["extra_body"] = extra_body
        
        # 限流：等待间隔
        self._wait_for_rate_limit()
        
        # 调用 API
        response = client.chat.completions.create(**request_params)
        
        # 提取 content（使用对象属性访问）
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message') and choice.message:
                content = choice.message.content or ""
            else:
                content = ""
        else:
            content = ""
        
        # 调试：首次调用时打印原始 API 返回数据
        if not self._first_call_logged:
            self._first_call_logged = True
            # 使用 tqdm.write 避免被进度条覆盖
            try:
                from tqdm import tqdm
                tqdm.write("\n" + "=" * 60)
                tqdm.write(f"🔍 [调试] {self.model_name} 首次 API 调用返回:")
                tqdm.write("=" * 60)
                tqdm.write(f"   response 类型: {type(response).__name__}")
                content_display = content[:100] + '...' if len(content) > 100 else content
                tqdm.write(f"   content: {content_display if content else 'N/A'}")
                tqdm.write(f"   usage: {response.usage}")
                tqdm.write("=" * 60 + "\n")
            except ImportError:
                content_display = content[:100] + '...' if len(content) > 100 else content
                print(f"\n[调试] content: {content_display if content else 'N/A'}", flush=True)
        
        # 提取 token 使用量信息
        usage_data = {}
        if response.usage:
            # 将 usage 对象转换为字典
            try:
                if hasattr(response.usage, 'model_dump'):
                    usage_data = response.usage.model_dump()
                elif hasattr(response.usage, '__dict__'):
                    usage_data = dict(response.usage.__dict__)
                else:
                    # 手动提取常见字段
                    usage_data = {
                        "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                        "total_tokens": getattr(response.usage, 'total_tokens', 0),
                    }
            except Exception:
                usage_data = {}
        
        # 提取缓存命中的 tokens（如果有）
        # OpenAI/Qwen API 可能返回 prompt_tokens_details.cached_tokens
        cached_tokens = 0
        prompt_tokens_details = usage_data.get("prompt_tokens_details") or {}
        if isinstance(prompt_tokens_details, dict) and prompt_tokens_details:
            cached_tokens = prompt_tokens_details.get("cached_tokens", 0) or 0
        usage_data["cached_tokens"] = cached_tokens
        
        # 记录 token 使用量（用于 TPM 限流）
        total_tokens = usage_data.get("total_tokens", 0) or usage_data.get("prompt_tokens", 0) + usage_data.get("completion_tokens", 0)
        self._record_token_usage(total_tokens)
        
        if enable_logprobs:
            # 提取 logprobs 信息（从 response 对象中获取）
            logprobs_data = None
            if response.choices and len(response.choices) > 0 and response.choices[0].logprobs:
                # 将 logprobs 对象转换为字典
                try:
                    if hasattr(response.choices[0].logprobs, 'model_dump'):
                        logprobs_data = response.choices[0].logprobs.model_dump()
                    elif hasattr(response.choices[0].logprobs, '__dict__'):
                        logprobs_data = dict(response.choices[0].logprobs.__dict__)
                    else:
                        logprobs_data = response.choices[0].logprobs
                except Exception:
                    logprobs_data = None
            return {
                "content": content,
                "logprobs": logprobs_data,
                "usage": usage_data
            }
        
        return {"content": content, "usage": usage_data}
    
    def _call_anthropic(self, prompt: str) -> str:
        """调用Anthropic API"""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config['api_key'],
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.config["model"],
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.max_tokens or 1024,
            "temperature": self.config.get("temperature", 0.7),
        }
        
        # 限流：等待间隔
        self._wait_for_rate_limit()
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result["content"][0]["text"]
    
    def _call_aws_claude(self, prompt: str) -> str:
        """调用AWS Bedrock Claude"""
        # 设置 AWS 凭证环境变量
        if 'aws_access_key_id' in self.config:
            os.environ["AWS_ACCESS_KEY_ID"] = self.config['aws_access_key_id']
        if 'aws_secret_access_key' in self.config:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.config['aws_secret_access_key']
        
        # 导入 AWS Claude 调用函数
        from aws_claude_caller import call_with_messages_aws_claude
        
        # 准备消息
        messages = [{'role': 'user', 'content': prompt}]
        
        # 限流：等待间隔
        self._wait_for_rate_limit()
        
        # 调用 AWS Claude
        response = call_with_messages_aws_claude(
            messages,
            max_tokens=self.max_tokens or 10000,
            temperature=self.config.get('temperature', 0.7),
            mode='normal'
        )
        
        return response
    
    def _call_aws_claude_binary(self, prompt: str) -> str:
        """调用AWS Bedrock Claude进行二分类预测（只输出Yes/No）"""
        # 设置 AWS 凭证环境变量
        if 'aws_access_key_id' in self.config:
            os.environ["AWS_ACCESS_KEY_ID"] = self.config['aws_access_key_id']
        if 'aws_secret_access_key' in self.config:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.config['aws_secret_access_key']
        
        # 导入 AWS Claude 调用函数
        from aws_claude_caller import call_with_messages_aws_claude
        
        # 准备消息
        messages = [{'role': 'user', 'content': prompt}]
        
        # 限流：等待间隔
        self._wait_for_rate_limit()
        
        # 调用 AWS Claude，限制输出长度
        response = call_with_messages_aws_claude(
            messages,
            max_tokens=5,  # 二分类只需要输出 Yes/No
            temperature=self.config.get('temperature', 0.7),
            mode='normal'
        )
        
        return response
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """检测是否是限流错误"""
        error_str = str(error).lower()
        rate_limit_keywords = [
            "rate limit", "ratelimit", "rate_limit", "RateLimitError",
            "429", "too many requests", "quota exceeded",
            "throttl", "limit exceeded", "请求过于频繁"
        ]
        return any(keyword in error_str for keyword in rate_limit_keywords)
    
    def call_binary_classification(self, prompt: str, max_retries: int = 5) -> Dict:
        """
        调用模型进行二分类预测（Yes/No）
        
        对于开源模型（use_logprobs=True）：使用 logprobs 计算 softmax 归一化概率
        对于闭源模型（use_logprobs=False）：直接映射 Yes->1, No->0
        
        Args:
            prompt: 输入的prompt（应该让模型只输出 Yes 或 No）
            max_retries: 最大重试次数，默认 5 次
            
        Returns:
            {
                "success": bool,  # 是否成功获取有效预测
                "prediction": float,  # 预测值（0-1之间的概率）
                "raw_output": str,  # 模型原始输出
                "method": str,  # "logprobs" 或 "direct_mapping"
                "retry_count": int,  # 重试次数
                "error": str or None,  # 错误信息（如果有）
                "logprob_yes": float or None,  # Yes 的 logprob（仅 logprobs 模式）
                "logprob_no": float or None,  # No 的 logprob（仅 logprobs 模式）
                "prompt_tokens": int,  # API 返回的 prompt token 数
                "completion_tokens": int,  # API 返回的 completion token 数
                "cached_tokens": int,  # 缓存命中的 token 数
            }
        """
        result = {
            "success": False,
            "prediction": None,
            "predicted_label": None,  # 基于YES/NO直接判定的标签（0或1）
            "raw_output": "",
            "method": "logprobs" if self.use_logprobs else "direct_mapping",
            "retry_count": 0,
            "error": None,
            "logprob_yes": None,
            "logprob_no": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
        
        for attempt in range(max_retries):
            result["retry_count"] = attempt
            
            try:
                if self.use_logprobs:
                    # 开源模型：使用 logprobs 计算概率
                    prediction_result = self._call_binary_with_logprobs(prompt)
                    if prediction_result["success"]:
                        result.update(prediction_result)
                        return result
                    else:
                        result["raw_output"] = prediction_result.get("raw_output", "")
                        result["error"] = prediction_result.get("error", "Unknown error")
                else:
                    # 闭源模型：直接映射 Yes/No
                    prediction_result = self._call_binary_direct_mapping(prompt)
                    if prediction_result["success"]:
                        result.update(prediction_result)
                        return result
                    else:
                        result["raw_output"] = prediction_result.get("raw_output", "")
                        result["error"] = prediction_result.get("error", "Unknown error")
                
                # 模型输出不是 Yes/No，等待后重试
                if attempt < max_retries - 1:
                    # 每 3 次重试打印一次日志
                    if attempt == 0 or (attempt + 1) % 3 == 0:
                        raw_output_preview = result.get("raw_output", "")[:50]
                        print(f"[{self.model_name}] 二分类输出无效 (尝试 {attempt + 1}/{max_retries}): '{raw_output_preview}...'")
                    
            except Exception as e:
                result["error"] = str(e)
                
                if attempt < max_retries - 1:
                    # 检测是否是限流错误
                    if self._is_rate_limit_error(e):
                        wait_time = 10  # 限流错误等待10秒
                        print(f"[{self.model_name}] 遇到限流错误，等待 {wait_time} 秒后重试: {e}")
                    else:
                        wait_time = 1  # 其他错误等待1秒
                        # 始终打印错误信息
                        print(f"[{self.model_name}] 二分类调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    
                    time.sleep(wait_time)
        
        # 所有重试都失败，打印详细原因
        result["retry_count"] = max_retries
        error_info = result.get("error", "未知错误")
        raw_output_preview = result.get("raw_output", "")[:100] if result.get("raw_output") else "无输出"
        print(f"[{self.model_name}] 二分类调用失败，已达到最大重试次数 {max_retries}，原因: {error_info}，输出: {raw_output_preview}")
        return result
    
    def call_continuous_prediction(self, prompt: str, max_retries: int = 5) -> Dict:
        """
        调用模型进行连续值预测（如观看时长）
        
        模型需要输出一个数字，我们解析并返回该数字作为预测值。
        
        Args:
            prompt: 输入的prompt（应该让模型输出一个数字）
            max_retries: 最大重试次数，默认 5 次
            
        Returns:
            {
                "success": bool,  # 是否成功获取有效预测
                "prediction": float,  # 预测的数值
                "raw_output": str,  # 模型原始输出
                "retry_count": int,  # 重试次数
                "error": str or None,  # 错误信息（如果有）
                "prompt_tokens": int,  # API 返回的 prompt token 数
                "completion_tokens": int,  # API 返回的 completion token 数
                "cached_tokens": int,  # 缓存命中的 token 数
            }
        """
        result = {
            "success": False,
            "prediction": None,
            "raw_output": "",
            "method": "continuous",
            "retry_count": 0,
            "error": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
        
        for attempt in range(max_retries):
            result["retry_count"] = attempt
            
            try:
                # 调用模型（连续值预测只需要输出一个数字，限制 max_tokens 避免超时）
                if self.model_type in ["openai", "deepseek", "qwen", "openai_compatible"]:
                    response_data = self._call_openai_compatible(prompt, enable_logprobs=False)
                elif self.model_type == "aws_claude":
                    response_data = self._call_aws_claude(prompt)
                elif self.model_type == "anthropic":
                    response_data = self._call_anthropic(prompt)
                else:
                    result["error"] = f"不支持的模型类型: {self.model_type}"
                    return result
                
                content = response_data["content"]
                usage_data = response_data.get("usage", {})
                result["raw_output"] = content
                
                # 提取 token 使用量
                result["prompt_tokens"] = usage_data.get("prompt_tokens", 0)
                result["completion_tokens"] = usage_data.get("completion_tokens", 0)
                result["cached_tokens"] = usage_data.get("cached_tokens", 0)
                
                # 从输出中提取数字
                number = self._extract_number_from_text(content)
                
                if number is not None:
                    result["success"] = True
                    result["prediction"] = number
                    return result
                else:
                    content_preview = content[:100] + '...' if len(content) > 100 else content
                    result["error"] = f"无法从模型输出中提取数字: '{content_preview}'"
                
                # 无法提取数字，等待后重试
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    # 每 3 次重试打印一次日志
                    if attempt == 0 or (attempt + 1) % 3 == 0:
                        raw_output_preview = result.get("raw_output", "")[:50]
                        print(f"[{self.model_name}] 连续值输出无效 (尝试 {attempt + 1}/{max_retries}): '{raw_output_preview}...'")
                    time.sleep(wait_time)
                    
            except Exception as e:
                result["error"] = str(e)
                
                if attempt < max_retries - 1:
                    # 检测是否是限流错误
                    if self._is_rate_limit_error(e):
                        wait_time = 10  # 限流错误等待10秒
                        print(f"[{self.model_name}] 遇到限流错误，等待 {wait_time} 秒后重试: {e}")
                    else:
                        wait_time = 1  # 其他错误等待1秒
                        # 始终打印错误信息
                        print(f"[{self.model_name}] 连续值预测调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    
                    time.sleep(wait_time)
        
        # 所有重试都失败，打印详细原因
        result["retry_count"] = max_retries
        error_info = result.get("error", "未知错误")
        raw_output_preview = result.get("raw_output", "")[:100] if result.get("raw_output") else "无输出"
        print(f"[{self.model_name}] 连续值预测调用失败，已达到最大重试次数 {max_retries}，原因: {error_info}，输出: {raw_output_preview}")
        return result
    
    def call_text_prediction(self, prompt: str, max_retries: int = 5) -> Dict:
        """
        调用模型进行文本预测（如搜索关键词、用户回复等）
        
        模型需要输出一段文本，我们直接返回该文本作为预测值。
        
        Args:
            prompt: 输入的prompt（应该让模型输出文本内容）
            max_retries: 最大重试次数，默认 5 次
            
        Returns:
            {
                "success": bool,  # 是否成功获取有效预测
                "prediction": str,  # 预测的文本
                "raw_output": str,  # 模型原始输出
                "retry_count": int,  # 重试次数
                "error": str or None,  # 错误信息（如果有）
                "prompt_tokens": int,  # API 返回的 prompt token 数
                "completion_tokens": int,  # API 返回的 completion token 数
                "cached_tokens": int,  # 缓存命中的 token 数
            }
        """
        result = {
            "success": False,
            "prediction": None,
            "raw_output": "",
            "method": "text",
            "retry_count": 0,
            "error": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
        
        for attempt in range(max_retries):
            result["retry_count"] = attempt
            
            try:
                # 调用模型（文本预测如搜索关键词）
                if self.model_type in ["openai", "deepseek", "qwen", "openai_compatible"]:
                    response_data = self._call_openai_compatible(prompt, enable_logprobs=False)
                elif self.model_type == "aws_claude":
                    response_data = self._call_aws_claude(prompt)
                elif self.model_type == "anthropic":
                    response_data = self._call_anthropic(prompt)
                else:
                    result["error"] = f"不支持的模型类型: {self.model_type}"
                    return result
                
                content = response_data["content"]
                usage_data = response_data.get("usage", {})
                result["raw_output"] = content
                
                # 提取 token 使用量
                result["prompt_tokens"] = usage_data.get("prompt_tokens", 0)
                result["completion_tokens"] = usage_data.get("completion_tokens", 0)
                result["cached_tokens"] = usage_data.get("cached_tokens", 0)
                
                # 清理文本输出
                cleaned_text = self._clean_text_output(content)
                
                if cleaned_text:
                    result["success"] = True
                    result["prediction"] = cleaned_text
                    return result
                else:
                    content_preview = content[:100] + '...' if len(content) > 100 else content
                    result["error"] = f"模型输出为空或无效: '{content_preview}'"
                
                # 输出为空，等待后重试
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    # 每 3 次重试打印一次日志
                    if attempt == 0 or (attempt + 1) % 3 == 0:
                        raw_output_preview = result.get("raw_output", "")[:50]
                        print(f"[{self.model_name}] 文本输出无效 (尝试 {attempt + 1}/{max_retries}): '{raw_output_preview}...'")
                    time.sleep(wait_time)
                    
            except Exception as e:
                result["error"] = str(e)
                
                if attempt < max_retries - 1:
                    # 检测是否是限流错误
                    if self._is_rate_limit_error(e):
                        wait_time = 10  # 限流错误等待10秒
                        print(f"[{self.model_name}] 遇到限流错误，等待 {wait_time} 秒后重试: {e}")
                    else:
                        wait_time = 1  # 其他错误等待1秒
                        # 始终打印错误信息
                        print(f"[{self.model_name}] 文本预测调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    
                    time.sleep(wait_time)
        
        # 所有重试都失败，打印详细原因
        result["retry_count"] = max_retries
        error_info = result.get("error", "未知错误")
        raw_output_preview = result.get("raw_output", "")[:100] if result.get("raw_output") else "无输出"
        print(f"[{self.model_name}] 文本预测调用失败，已达到最大重试次数 {max_retries}，原因: {error_info}，输出: {raw_output_preview}")
        return result
    
    def _clean_text_output(self, text: str) -> str:
        """
        清理文本输出
        
        移除引号、多余的空白符等
        
        Args:
            text: 模型输出的文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        text = text.strip()
        
        # 移除开头和结尾的引号（中文或英文）
        quotes = ['"', "'", '"', '"', ''', ''', '「', '」', '『', '』']
        for quote in quotes:
            if text.startswith(quote) and text.endswith(quote):
                text = text[1:-1].strip()
                break
            elif text.startswith(quote):
                text = text[1:].strip()
            elif text.endswith(quote):
                text = text[:-1].strip()
        
        # 移除"搜索关键词："等前缀
        prefixes_to_remove = [
            "搜索关键词：", "搜索关键词:", "关键词：", "关键词:",
            "搜索：", "搜索:", "Search:", "Keyword:", "Keywords:",
        ]
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                break
        
        return text
    
    def _extract_number_from_text(self, text: str) -> Optional[float]:
        """
        从文本中提取数字
        
        支持多种格式：
        - 纯数字: "42", "3.14"
        - 带单位: "42秒", "3.5分钟"
        - 句子中的数字: "我预测观看时长为42秒"
        
        Args:
            text: 模型输出的文本
            
        Returns:
            提取的数字，如果无法提取则返回 None
        """
        if not text:
            return None
        
        text = text.strip()
        
        # 首先尝试直接解析为数字
        try:
            return float(text)
        except ValueError:
            pass
        
        # 尝试提取数字（包括小数）
        # 优先匹配整数或小数
        patterns = [
            r'(\d+\.?\d*)\s*秒',  # 匹配 "42秒" 或 "42.5秒"
            r'(\d+\.?\d*)\s*分钟',  # 匹配 "3分钟" 或 "3.5分钟"（需要转换为秒）
            r'(\d+\.?\d*)\s*s(?:econds?)?',  # 匹配 "42s" 或 "42 seconds"
            r'(\d+\.?\d*)\s*min(?:utes?)?',  # 匹配 "3min" 或 "3 minutes"（需要转换为秒）
            r'^(\d+\.?\d*)$',  # 匹配纯数字
            r'(\d+\.?\d*)',  # 匹配任意数字（最后的fallback）
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                number = float(match.group(1))
                # 如果是分钟，转换为秒
                if i == 1 or i == 3:  # 分钟模式
                    number *= 60
                return number
        
        return None
    
    def _call_binary_with_logprobs(self, prompt: str) -> Dict:
        """
        使用 logprobs 进行二分类预测（开源模型）
        
        Args:
            prompt: 输入的prompt
        
        Returns:
            {
                "success": bool,
                "prediction": float,  # softmax 归一化后的 P(Yes)
                "raw_output": str,
                "logprob_yes": float or None,
                "logprob_no": float or None,
                "error": str or None,
                "prompt_tokens": int,  # API 返回的 prompt token 数
                "completion_tokens": int,  # API 返回的 completion token 数
                "cached_tokens": int,  # 缓存命中的 token 数
            }
        """
        result = {
            "success": False,
            "prediction": None,
            "predicted_label": None,  # 基于YES/NO直接判定的标签（0或1）
            "raw_output": "",
            "logprob_yes": None,
            "logprob_no": None,
            "error": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
        
        try:
            # 支持新旧两种类型标识
            if self.model_type in ["openai", "deepseek", "qwen", "openai_compatible"]:
                response_data = self._call_openai_compatible(prompt, enable_logprobs=True)
            else:
                # 不支持 logprobs 的模型类型，降级为直接映射
                result["error"] = f"模型类型 {self.model_type} 不支持 logprobs"
                return result
            
            content = response_data["content"]
            logprobs_data = response_data["logprobs"]
            usage_data = response_data.get("usage", {})
            result["raw_output"] = content
            
            # 提取 token 使用量
            result["prompt_tokens"] = usage_data.get("prompt_tokens", 0)
            result["completion_tokens"] = usage_data.get("completion_tokens", 0)
            result["cached_tokens"] = usage_data.get("cached_tokens", 0)
            
            # 检查输出是否为 Yes 或 No
            content_stripped = content.strip()
            if content_stripped not in ["Yes", "No"]:
                result["error"] = f"模型输出不是 Yes/No: '{content_stripped}'"
                return result
            
            # 提取 logprobs
            if logprobs_data and "content" in logprobs_data and len(logprobs_data["content"]) > 0:
                top_logprobs = logprobs_data["content"][0].get("top_logprobs", [])
                
                logprob_yes = None
                logprob_no = None
                
                for item in top_logprobs:
                    token = item.get("token", "")
                    logprob = item.get("logprob", None)
                    
                    if token == "Yes":
                        logprob_yes = logprob
                    elif token == "No":
                        logprob_no = logprob
                    
                    if logprob_yes is not None and logprob_no is not None:
                        break
                
                result["logprob_yes"] = logprob_yes
                result["logprob_no"] = logprob_no
                
                # 基于YES/NO直接判定标签
                result["predicted_label"] = 1 if content_stripped == "Yes" else 0
                
                # 计算 softmax 归一化
                if logprob_yes is not None and logprob_no is not None:
                    exp_yes = math.exp(logprob_yes)
                    exp_no = math.exp(logprob_no)
                    p_yes = exp_yes / (exp_yes + exp_no)
                    
                    result["success"] = True
                    result["prediction"] = p_yes
                else:
                    # 如果无法找到两个 logprob，降级为直接映射
                    result["error"] = "无法在 top_logprobs 中找到 Yes 和 No 的 logprob"
                    result["success"] = True
                    result["prediction"] = 1.0 if content_stripped == "Yes" else 0.0
            else:
                # logprobs 数据不完整，降级为直接映射
                result["error"] = "logprobs 数据不完整"
                result["predicted_label"] = 1 if content_stripped == "Yes" else 0
                result["success"] = True
                result["prediction"] = 1.0 if content_stripped == "Yes" else 0.0
                    
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _call_binary_direct_mapping(self, prompt: str) -> Dict:
        """
        直接映射 Yes/No 进行二分类预测（闭源模型）
        
        Args:
            prompt: 输入的prompt
        
        Returns:
            {
                "success": bool,
                "prediction": float,  # Yes->1.0, No->0.0
                "raw_output": str,
                "error": str or None,
                "prompt_tokens": int,  # API 返回的 prompt token 数
                "completion_tokens": int,  # API 返回的 completion token 数
                "cached_tokens": int,  # 缓存命中的 token 数
            }
        """
        result = {
            "success": False,
            "prediction": None,
            "predicted_label": None,  # 基于YES/NO直接判定的标签（0或1）
            "raw_output": "",
            "error": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
        
        try:
            # 调试：在 API 调用前打印（仅首次）
            if not self._direct_mapping_debug_done:
                try:
                    from tqdm import tqdm
                    tqdm.write(f"\n[调试] 准备调用 API, model_type={self.model_type}")
                except ImportError:
                    print(f"\n[调试] 准备调用 API, model_type={self.model_type}", flush=True)
            
            # 支持新旧两种类型标识
            if self.model_type in ["openai", "deepseek", "qwen", "openai_compatible"]:
                # 二分类只需要输出 Yes/No，限制 max_tokens
                response_data = self._call_openai_compatible(prompt, enable_logprobs=False)
                response = response_data["content"]
                usage_data = response_data.get("usage", {})
                
                # 调试：检查 usage_data 的实际内容（仅首次）
                if not self._direct_mapping_debug_done:
                    self._direct_mapping_debug_done = True
                    try:
                        from tqdm import tqdm
                        tqdm.write(f"[调试] API 调用成功!")
                        tqdm.write(f"[调试] response_data keys: {list(response_data.keys())}")
                        tqdm.write(f"[调试] usage_data 类型: {type(usage_data)}")
                        tqdm.write(f"[调试] usage_data 内容: {usage_data}")
                    except ImportError:
                        print(f"[调试] usage_data: {usage_data}", flush=True)
                
                result["prompt_tokens"] = usage_data.get("prompt_tokens", 0) or 0
                result["completion_tokens"] = usage_data.get("completion_tokens", 0) or 0
                result["cached_tokens"] = usage_data.get("cached_tokens", 0) or 0
            elif self.model_type == "anthropic":
                response = self._call_anthropic(prompt)
            elif self.model_type == "aws_claude":
                response = self._call_aws_claude_binary(prompt)
            else:
                result["error"] = f"不支持的模型类型: {self.model_type}"
                return result
            
            result["raw_output"] = response
            
            # 清理输出：移除markdown标记、引号等
            content_cleaned = self._clean_text_output(response)
            # 移除所有空白字符以便匹配
            content_compact = "".join(content_cleaned.split()).lower()
            
            # 策略1：精确匹配（忽略大小写和空白）
            if content_compact == "yes":
                result["success"] = True
                result["prediction"] = 1.0
                result["predicted_label"] = 1
                return result
            elif content_compact == "no":
                result["success"] = True
                result["prediction"] = 0.0
                result["predicted_label"] = 0
                return result
            
            # 策略2：前缀匹配（处理 "Yes." 或 "Yes, I think..."）
            if content_cleaned.lower().startswith("yes"):
                result["success"] = True
                result["prediction"] = 1.0
                result["predicted_label"] = 1
                return result
            elif content_cleaned.lower().startswith("no"):
                result["success"] = True
                result["prediction"] = 0.0
                result["predicted_label"] = 0
                return result
                
            # 策略3：关键词搜索（处理 "Answer: Yes" 或 markdown "**Yes**"）
            # 只在开头部分搜索，避免匹配到长文本中间的无关单词
            head_part = content_cleaned[:20].lower()
            if "yes" in head_part and "no" not in head_part:
                result["success"] = True
                result["prediction"] = 1.0
                result["predicted_label"] = 1
                return result
            elif "no" in head_part and "yes" not in head_part:
                result["success"] = True
                result["prediction"] = 0.0
                result["predicted_label"] = 0
                return result
            
            # 策略4：更宽松的搜索（最后尝试）
            import re
            # 匹配独立的单词 Yes/No
            if re.search(r'\b(yes|YES|Yes)\b', content_cleaned):
                 result["success"] = True
                 result["prediction"] = 1.0
                 result["predicted_label"] = 1
            elif re.search(r'\b(no|NO|No)\b', content_cleaned):
                 result["success"] = True
                 result["prediction"] = 0.0
                 result["predicted_label"] = 0
            else:
                error_msg = f"模型输出不是 Yes/No: '{content_cleaned[:50]}...'"
                result["error"] = error_msg
                self._log_error(prompt, response, error_msg)
                
        except Exception as e:
            result["error"] = str(e)
            self._log_error(prompt, response if 'response' in locals() else "NO_RESPONSE", str(e))
            # 调试：打印异常信息
            try:
                from tqdm import tqdm
                tqdm.write(f"\n[调试] API 调用异常: {type(e).__name__}: {e}")
            except ImportError:
                print(f"\n[调试] API 调用异常: {type(e).__name__}: {e}", flush=True)
        
        return result


def parse_model_response(response_text: str, num_questions: int, debug: bool = False) -> Dict:
    """
    解析模型响应，提取预测值（支持数值和文本）
    
    Args:
        response_text: 模型的原始响应
        num_questions: 问题数量
        debug: 是否打印调试信息
        
    Returns:
        Dict mapping "answer_1", "answer_2", ... to predicted values (可能是float或str)
        如果某个问题解析失败，该键的值为 None (而不是0.0)
    """
    if debug:
        print(f"\n[DEBUG] 原始响应:\n{response_text[:500]}...\n")
    
    # 策略1: 尝试提取markdown包裹的JSON
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    
    # 策略2: 尝试提取普通的JSON（不带markdown标记）
    if not json_match:
        json_match = re.search(r'```\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    
    # 策略3: 直接查找JSON对象
    if not json_match:
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
    
    if json_match:
        try:
            json_str = json_match.group(1) if json_match.lastindex and json_match.lastindex >= 1 else json_match.group(0)
            
            if debug:
                print(f"[DEBUG] 提取的JSON字符串:\n{json_str}\n")
            
            # 清理JSON字符串（移除可能的注释和多余空格）
            json_str = re.sub(r'//.*?\n', '\n', json_str)  # 移除单行注释
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)  # 移除多行注释
            
            result = json.loads(json_str)
            
            # 标准化键名，保持原始类型（可能是数字或文本）
            parsed = {}
            for i in range(1, num_questions + 1):
                key = f"answer_{i}"
                if key in result:
                    value = result[key]
                    # 如果是字符串，保持字符串；如果是数字，转换为float
                    if isinstance(value, str):
                        parsed[key] = value
                    else:
                        try:
                            parsed[key] = float(value)
                        except (ValueError, TypeError):
                            parsed[key] = str(value)
                else:
                    # 如果键不存在，给 None
                    if debug:
                        print(f"[DEBUG] 警告: 缺少 {key}")
                    parsed[key] = None
            
            if debug:
                print(f"[DEBUG] 解析成功: {parsed}\n")
            
            return parsed
            
        except json.JSONDecodeError as e:
            if debug:
                print(f"[DEBUG] JSON解析失败: {e}\n")
    
    # 如果JSON解析失败，尝试正则提取（备用方案）
    if debug:
        print("[DEBUG] 使用备用方案：正则提取\n")
    
    parsed = {}
    for i in range(1, num_questions + 1):
        # 先尝试提取字符串值
        pattern_str = rf'"?answer_{i}"?\s*:\s*"([^"]+)"'
        match = re.search(pattern_str, response_text)
        if match:
            parsed[f"answer_{i}"] = match.group(1)
            if debug:
                print(f"[DEBUG] answer_{i} = {match.group(1)} (字符串)")
        else:
            # 再尝试提取数字值
            pattern_num = rf'"?answer_{i}"?\s*:\s*([0-9.]+)'
            match = re.search(pattern_num, response_text)
            if match:
                try:
                    parsed[f"answer_{i}"] = float(match.group(1))
                    if debug:
                        print(f"[DEBUG] answer_{i} = {match.group(1)} (数字)")
                except ValueError:
                    parsed[f"answer_{i}"] = None
                    if debug:
                        print(f"[DEBUG] answer_{i} = None (转换失败)")
            else:
                parsed[f"answer_{i}"] = None
                if debug:
                    print(f"[DEBUG] answer_{i} = None (未找到)")
    
    return parsed
