#!/usr/bin/env python3
"""
History action processor: unified entry point

Features:
- Select RAG or Summary processor based on config
- RAG mode: vector retrieval on history actions, returns most relevant actions
- Summary mode: summarize history actions (not yet implemented)

Usage:
1. Configure HISTORY_PROCESS_MODE = "rag" or "summary" in config.py
2. Call process_history() from prompt_builder.py

Dependencies (only needed when using RAG):
    pip install langchain langchain-text-splitters langchain-community langchain-openai faiss-cpu tiktoken
"""

from typing import List, Dict, Optional, Tuple

from config import HISTORY_PROCESS_MODE, RAG_CONFIG


_rag_processor = None
_summary_processor = None


def _get_rag_processor():
    global _rag_processor
    if _rag_processor is None:
        try:
            import rag_processor
            _rag_processor = rag_processor.get_rag_processor()
        except Exception as e:
            print(f"⚠️ Failed to load RAG processor: {e}")
            _rag_processor = None
    return _rag_processor


def _get_summary_processor():
    global _summary_processor
    if _summary_processor is None:
        try:
            import summary_processor
            _summary_processor = summary_processor.get_summary_processor()
        except Exception as e:
            print(f"⚠️ Failed to load Summary processor: {e}")
            _summary_processor = None
    return _summary_processor


def _ensure_langchain_imported() -> bool:
    """
    Check if LangChain is available (for backward compatibility)

    Returns:
        bool: whether available
    """
    try:
        import rag_processor
        return rag_processor._ensure_langchain_imported()
    except Exception:
        return False


def get_history_processor(mode: str = None):
    """
    Get history processor instance

    Args:
        mode: processing mode, one of "rag", "summary", "none"
              If None, uses HISTORY_PROCESS_MODE config

    Returns:
        Processor instance, or None if mode is "none"
    """
    mode = mode or HISTORY_PROCESS_MODE

    if mode == "none":
        return None
    elif mode == "rag":
        return _get_rag_processor()
    elif mode == "summary":
        return _get_summary_processor()
    else:
        print(f"Unknown mode: {mode}, falling back to 'none'")
        return None


def process_history(
    action_history: List[Dict],
    current_action: Dict = None,
    mode: str = None,
    max_output_tokens: int = None
) -> Tuple[str, bool]:
    """
    Convenience function to process history actions

    Args:
        action_history: list of history actions
        current_action: current action to predict (needed for RAG mode)
        mode: processing mode
        max_output_tokens: maximum output token count

    Returns:
        (processed history text, whether a processor was used)
    """
    processor = get_history_processor(mode)

    if processor is None:
        return None, False

    try:
        result = processor.process(
            action_history,
            current_action,
            max_output_tokens
        )
        return result, True
    except Exception as e:
        print(f"History processing failed: {e}")
        return None, False


def precompute_all_embeddings(user_data_list: List[Dict], mode: str = "rag"):
    """
    Pre-compute embeddings for all users' histories (RAG mode only)

    Args:
        user_data_list: list of user data
        mode: processing mode, only "rag" is supported
    """
    if mode != "rag":
        print("⚠️ Only RAG mode is supported")
        return

    processor = _get_rag_processor()
    if processor is None:
        print("⚠️ Failed to create RAG processor")
        return

    print("⚠️ precompute_all_embeddings is deprecated, use precompute_user_indices instead")


def precompute_user_indices(eval_data: List[Dict], show_progress: bool = True, mode: str = None, max_workers: int = None):
    """
    Pre-build per-user indices (RAG) or summaries (Summary) for each user

    Args:
        eval_data: evaluation data list
        show_progress: whether to show progress
        mode: processing mode, one of "rag", "summary". If None, uses HISTORY_PROCESS_MODE config
        max_workers: concurrent worker threads (only effective for summary mode), default None (serial). Recommended: 4-8
    """
    mode = mode or HISTORY_PROCESS_MODE

    if mode == "rag":
        processor = _get_rag_processor()
        if processor is None:
            print("⚠️ Failed to create RAG processor")
            return

        try:
            processor.precompute_user_indices(eval_data, show_progress=show_progress)
        except Exception as e:
            print(f"⚠️ Failed to precompute user indices: {e}")

    elif mode == "summary":
        processor = _get_summary_processor()
        if processor is None:
            print("⚠️ Failed to create Summary processor")
            return

        try:
            processor.precompute_user_summaries(eval_data, show_progress=show_progress, max_workers=max_workers)
        except Exception as e:
            print(f"⚠️ Failed to precompute user summaries: {e}")

    else:
        print(f"⚠️ Unknown mode: {mode}, skipping precomputation")


def precompute_from_data_file(data_path: str, mode: str = "rag"):
    """
    Precompute embeddings from a data file (deprecated)

    Args:
        data_path: path to data file (JSON format)
        mode: processing mode
    """
    print("⚠️ precompute_from_data_file is deprecated")


def configure_rag(
    embedding_model: str = None,
    api_key: str = None,
    base_url: str = None,
    top_k: int = None,
    chunk_size: int = None,
    use_cache: bool = None,
    cache_dir: str = None
):
    """Configure the RAG processor"""
    if embedding_model:
        RAG_CONFIG["embedding_model"] = embedding_model
    if api_key:
        RAG_CONFIG["api_key"] = api_key
    if base_url:
        RAG_CONFIG["base_url"] = base_url
    if top_k:
        RAG_CONFIG["top_k"] = top_k
    if chunk_size:
        RAG_CONFIG["chunk_size"] = chunk_size
    if use_cache is not None:
        RAG_CONFIG["use_cache"] = use_cache
    if cache_dir:
        RAG_CONFIG["cache_dir"] = cache_dir

    global _rag_processor
    _rag_processor = None


def set_history_process_mode(mode: str):
    """Set the history processing mode"""
    global HISTORY_PROCESS_MODE, _rag_processor, _summary_processor

    if mode not in ["none", "rag", "summary"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'none', 'rag' or 'summary'")

    HISTORY_PROCESS_MODE = mode
    _rag_processor = None
    _summary_processor = None


def format_action_for_embedding(action: Dict) -> str:
    """
    Compatibility function: delegates to the implementation in rag_processor

    Note: This function is for backward compatibility only; new code should import rag_processor directly
    """
    try:
        import rag_processor
        return rag_processor.format_action_for_embedding(action)
    except Exception:
        timestamp = action.get("timestamp", "unknown time")
        action_type = action.get("type", "unknown action")
        return f"time: {timestamp} | scene: {action_type}"


if __name__ == "__main__":
    print("=" * 60)
    print("History action processor test")
    print("=" * 60)

    test_actions = [
        {
            "timestamp": "2024-01-15 10:30:00",
            "type": "Video",
            "context": {
                "caption": "Food exploration: Chengdu hotpot",
                "duration": 120,
                "like_cnt": 1000
            },
            "action": [{"type": "like"}, {"type": "collect"}]
        },
        {
            "timestamp": "2024-01-15 11:00:00",
            "type": "Video",
            "context": {
                "caption": "Japan travel vlog",
                "duration": 300,
                "like_cnt": 5000
            },
            "action": [{"type": "watch complete"}]
        },
    ]

    current_action = {
        "timestamp": "2024-01-15 15:00:00",
        "type": "Video",
        "context": {
            "caption": "Sichuan home cooking",
            "duration": 180
        }
    }

    print("\nTest format_action_for_embedding:")
    print("=" * 60)
    for action in test_actions:
        print(format_action_for_embedding(action))
        print()

    print("\nCurrent mode:", HISTORY_PROCESS_MODE)
    print("\nTo test RAG mode, make sure:")
    print("1. Install dependencies: pip install langchain langchain-community langchain-openai faiss-cpu")
    print("2. Set HISTORY_PROCESS_MODE = 'rag' in config.py")
    print("3. Call: result, used = process_history(test_actions, current_action)")
