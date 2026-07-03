"""
Prompt Builder: Convert user data into LLM-friendly prompts
"""
import json
from typing import Dict, List, Optional
from config import *

# 尝试导入历史处理器（RAG/Summary）
try:
    import history_processor as _history_processor
    process_history = _history_processor.process_history
    # 注意：不要直接导入 HISTORY_PROCESS_MODE，因为它可能在运行时被修改
    # 使用函数获取当前值，避免导入时的值副本问题
    def _get_history_process_mode():
        return _history_processor.HISTORY_PROCESS_MODE
    HISTORY_PROCESSOR_AVAILABLE = True
except ImportError:
    HISTORY_PROCESSOR_AVAILABLE = False
    _history_processor = None
    def _get_history_process_mode():
        return "none"
    def process_history(*args, **kwargs):
        return None, False

# 尝试导入Qwen tokenizer用于准确的token计数
try:
    from transformers import AutoTokenizer
    QWEN_TOKENIZER = AutoTokenizer.from_pretrained("/path/to/Qwen3-8B", trust_remote_code=True)
    TOKENIZER_AVAILABLE = True
except Exception as e:
    QWEN_TOKENIZER = None
    TOKENIZER_AVAILABLE = False
    print(f"Warning: Qwen tokenizer not available ({e}), using approximate token counting")


def format_action_context(action: Dict) -> str:
    """格式化单个行为的上下文信息"""
    context = action.get("context", {})
    action_type = action.get("type", "")
    
    # 根据不同的行为类型，提取关键信息
    if action_type == "视频浏览":
        parts = []
        
        # 视频标题
        if "caption" in context and context["caption"]:
            parts.append(f"这是一个标题为《{context['caption']}》的视频")
        else:
            parts.append("这是一个视频")
        
        # 作者信息 - 粉丝数和认证状态
        if "fans_user_num" in context and context["fans_user_num"] > 0:
            fans_num = context["fans_user_num"]
            is_verified = context.get("is_verified", False)
            verification_type = context.get("verification_type", "")
            
            if fans_num >= 10000:
                fans_str = f"{fans_num / 10000:.1f}万"
            else:
                fans_str = f"{fans_num}"
            
            author_info = f"作者拥有 {fans_str} 粉丝"
            if is_verified and verification_type:
                author_info += f"，并且通过了{verification_type}认证"
            elif is_verified:
                author_info += "，并且已通过平台认证"
            parts.append(author_info)
        elif context.get("is_verified", False):
            verification_type = context.get("verification_type", "")
            if verification_type:
                parts.append(f"作者已通过{verification_type}认证")
            else:
                parts.append("作者已通过平台认证")
        
        # 视频时长
        if "duration" in context and context["duration"]:
            duration_minutes = context['duration'] // 60
            duration_seconds = context['duration'] % 60
            if duration_minutes > 0:
                parts.append(f"视频时长为 {duration_minutes} 分 {duration_seconds} 秒")
            else:
                parts.append(f"视频时长为 {duration_seconds} 秒")
        
        # 视频热度数据
        popularity = []
        # 展示次数
        if "show_cnt" in context and context["show_cnt"] > 0:
            popularity.append(f"{context['show_cnt']}次曝光")
        # 播放次数
        if "play_cnt" in context and context["play_cnt"] > 0:
            popularity.append(f"{context['play_cnt']}次播放")
        # 完播次数
        if "complete_play_cnt" in context and context["complete_play_cnt"] > 0:
            popularity.append(f"{context['complete_play_cnt']}次完整播放")
        # 点赞数
        if "like_cnt" in context and context["like_cnt"] > 0:
            popularity.append(f"{context['like_cnt']}个赞")
        # 评论数
        if "comment_cnt" in context and context["comment_cnt"] > 0:
            popularity.append(f"{context['comment_cnt']}条评论")
        # 分享数
        if "share_cnt" in context and context["share_cnt"] > 0:
            popularity.append(f"{context['share_cnt']}次分享")
        # 收藏数
        if "collect_cnt" in context and context["collect_cnt"] > 0:
            popularity.append(f"{context['collect_cnt']}次收藏")
        # 下载数
        if "download_cnt" in context and context["download_cnt"] > 0:
            popularity.append(f"{context['download_cnt']}次下载")
        if "follow_cnt" in context and context["follow_cnt"] > 0:
            popularity.append(f"{context['follow_cnt']}次关注")
        
        if popularity:
            parts.append("该视频目前收获了 " + "、".join(popularity))
        
        # OCR文本内容（视频画面中的文字）
        # TODO: 构建练习题的时候把这两个东西去掉
        if "ocr" in context and context["ocr"]:
            ocr_text = context['ocr']
            if len(ocr_text) > 10000:
                parts.append(f"视频画面中识别到的文字内容为：{ocr_text[:10000]}...")
            else:
                parts.append(f"视频画面中识别到的文字内容为：{ocr_text}")
        
        # ASR语音文本内容
        # TODO: 构建练习题的时候把这两个东西去掉
        if "asr" in context and context["asr"]:
            asr_text = context['asr']
            if len(asr_text) > 10000:
                parts.append(f"视频中的语音内容为：{asr_text[:10000]}...")
            else:
                parts.append(f"视频中的语音内容为：{asr_text}")
        
        # 上传时间
        if "upload_timestamp" in context and context["upload_timestamp"]:
            parts.append(f"发布时间是 {context['upload_timestamp']}")
        
        return "。".join(parts) + "。" if parts else "视频信息缺失。"
    
    elif action_type == "电影评分":
        return f"这个电影名称为{context['movie_title']}, 上映于{context['release_year']}, 它的类型是{context['genres']}"

    elif action_type == "商城购物":
        parts = []
        
        # 商品名称
        if "item_title" in context and context["item_title"]:
            parts.append(f"这是一件商品，名称为{context['item_title']}")
        else:
            parts.append("这是一件商品")
        
        # 商品描述
        if "item_desc" in context and context["item_desc"]:
            desc = context["item_desc"]
            if len(desc) > 10000:
                parts.append(f"商品描述：{desc[:10000]}...")
            else:
                parts.append(f"商品描述：{desc}")
        
        # 商品来源
        if "product_source" in context and context["product_source"]:
            parts.append(f"商品来自{context['product_source']}")
        
        # 价格
        if "item_price" in context and context["item_price"]:
            price = context['item_price']
            if price > 0:
                parts.append(f"售价为 {price} 元")
            else:
                parts.append("售价未知")
        
        # 类别信息（三级类别）
        category_info = []
        if "category_level1_name" in context and context["category_level1_name"]:
            category_info.append(context["category_level1_name"])
        if "category_level2_name" in context and context["category_level2_name"]:
            category_info.append(context["category_level2_name"])
        if "category_level3_name" in context and context["category_level3_name"]:
            category_info.append(context["category_level3_name"])
        
        if category_info:
            parts.append(f"商品具体类别为：{'，'.join(category_info)}")
        
        # 行业分类（三级行业）
        industry_info = []
        if "industry_level1_name" in context and context["industry_level1_name"]:
            industry_info.append(context["industry_level1_name"])
        if "industry_level2_name" in context and context["industry_level2_name"]:
            industry_info.append(context["industry_level2_name"])
        if "industry_level3_name" in context and context["industry_level3_name"]:
            industry_info.append(context["industry_level3_name"])
        
        if industry_info:
            parts.append(f"所属行业为：{'，'.join(industry_info)}")
        
        # 销量和评价数据
        sales_info = []
        if "item_num_180d_sales" in context and context["item_num_180d_sales"] > 0:
            sales = context["item_num_180d_sales"]
            if sales >= 10000:
                sales_info.append(f"近180天已售出 {sales/10000:.1f}万件")
            else:
                sales_info.append(f"近180天已售出 {sales} 件")
        
        if "item_num_good_assess" in context and context["item_num_good_assess"] > 0:
            assess = context["item_num_good_assess"]
            if assess >= 10000:
                sales_info.append(f"{assess/10000:.1f}万条好评")
            else:
                sales_info.append(f"{assess}条好评")
        
        if "item_num_180d_buyer" in context and context["item_num_180d_buyer"] > 0:
            buyer = context["item_num_180d_buyer"]
            if buyer >= 10000:
                sales_info.append(f"{buyer/10000:.1f}万人购买")
            else:
                sales_info.append(f"{buyer}人购买")
        
        if "item_num_180d_rebuyer" in context and context["item_num_180d_rebuyer"] > 0:
            rebuyer = context["item_num_180d_rebuyer"]
            if rebuyer >= 10000:
                sales_info.append(f"{rebuyer/10000:.1f}万人复购")
            else:
                sales_info.append(f"{rebuyer}人复购")
        
        if sales_info:
            parts.append("该商品" + "、".join(sales_info))
        
        # 加购数据
        if "item_num_180d_cart" in context and context["item_num_180d_cart"] > 0:
            cart_num = context["item_num_180d_cart"]
            if cart_num >= 10000:
                parts.append(f"近180天有 {cart_num/10000:.1f}万 人加入购物车")
            else:
                parts.append(f"近180天有 {cart_num} 人加入购物车")
        
        return "。".join(parts) + "。" if parts else "商品信息缺失。"
    
    elif action_type == "广告推荐":
        parts = []
        
        # 产品和行业
        if "product" in context and context["product"]:
            product_info = f"这是一条推广{context['product']}的广告"
            if "industry_main" in context and context["industry_main"]:
                product_info += f"，属于{context['industry_main']}"
                if "industry_sub" in context and context["industry_sub"] and context["industry_sub"] != context.get("industry_main"):
                    product_info += f"行业中的{context['industry_sub']}品类"
                else:
                    product_info += "行业"
            parts.append(product_info)
        else:
            parts.append("这是一条广告")
        
        # 广告主认证信息
        if "is_verified" in context and context["is_verified"]:
            if "verification_type" in context and context["verification_type"]:
                parts.append(f"广告主已认证，认证类型为{context['verification_type']}")
            else:
                parts.append("广告主已认证")
        
        # 粉丝数
        if "fans_user_num" in context and context["fans_user_num"]:
            parts.append(f"广告主粉丝数为 {context['fans_user_num']}")
        elif "fans_user_num_op_range" in context and context["fans_user_num_op_range"]:
            parts.append(f"广告主粉丝数范围为{context['fans_user_num_op_range']}")
        
        # 广告标题/描述
        if "caption" in context and context["caption"]:
            caption = context['caption']
            if len(caption) > 500:
                parts.append(f"广告标题为：{caption[:500]}...")
            else:
                parts.append(f"广告标题为：{caption}")
        
        # 广告类型
        if "photo_type" in context and context["photo_type"]:
            parts.append(f"广告类型为{context['photo_type']}")
        
        # 广告时长
        if "duration" in context and context["duration"]:
            duration = context["duration"]
            if duration >= 60:
                minutes = duration // 60
                seconds = duration % 60
                if seconds > 0:
                    parts.append(f"广告时长为 {minutes} 分 {seconds} 秒")
                else:
                    parts.append(f"广告时长为 {minutes} 分钟")
            else:
                parts.append(f"广告时长为 {duration} 秒")
        
        # 上传时间
        if "upload_timestamp" in context and context["upload_timestamp"]:
            parts.append(f"广告上传时间为 {context['upload_timestamp']}")
        
        # 曝光次数
        if "show_cnt" in context and context["show_cnt"] and context["show_cnt"] > 0:
            parts.append(f"该广告已经向用户曝光过 {context['show_cnt']} 次")
        
        # 播放次数
        if "play_cnt" in context and context["play_cnt"] and context["play_cnt"] > 0:
            parts.append(f"广告播放次数为 {context['play_cnt']}")
        
        # 完整播放次数
        if "complete_play_cnt" in context and context["complete_play_cnt"] and context["complete_play_cnt"] > 0:
            parts.append(f"广告被完整播放 {context['complete_play_cnt']} 次")
        
        # 互动数据
        interaction_parts = []
        if "like_cnt" in context and context["like_cnt"] and context["like_cnt"] > 0:
            interaction_parts.append(f"获得点赞 {context['like_cnt']} 次")
        if "comment_cnt" in context and context["comment_cnt"] and context["comment_cnt"] > 0:
            interaction_parts.append(f"获得评论 {context['comment_cnt']} 次")
        if "share_cnt" in context and context["share_cnt"] and context["share_cnt"] > 0:
            interaction_parts.append(f"被分享 {context['share_cnt']} 次")
        if "collect_cnt" in context and context["collect_cnt"] and context["collect_cnt"] > 0:
            interaction_parts.append(f"被收藏 {context['collect_cnt']} 次")
        if "download_cnt" in context and context["download_cnt"] and context["download_cnt"] > 0:
            interaction_parts.append(f"被下载 {context['download_cnt']} 次")
        if "follow_cnt" in context and context["follow_cnt"] and context["follow_cnt"] > 0:
            interaction_parts.append(f"被关注 {context['follow_cnt']} 次")
        if interaction_parts:
            parts.append(f"广告互动数据：{', '.join(interaction_parts)}")
        
        # 负面反馈数据
        negative_parts = []
        if "report_cnt" in context and context["report_cnt"] and context["report_cnt"] > 0:
            negative_parts.append(f"被举报 {context['report_cnt']} 次")
        if "reduce_similar_cnt" in context and context["reduce_similar_cnt"] and context["reduce_similar_cnt"] > 0:
            negative_parts.append(f"减少类似推荐 {context['reduce_similar_cnt']} 次")
        if negative_parts:
            parts.append(f"广告负面反馈：{', '.join(negative_parts)}")
        
        # OCR文本内容（广告文案）
        if "ocr_text" in context and context["ocr_text"]:
            ocr_text = context['ocr_text']
            if len(ocr_text) > 10000:
                parts.append(f"广告画面中的文字内容为：{ocr_text[:10000]}")
            else:
                parts.append(f"广告画面中的文字内容为：{ocr_text}")
        
        # ASR语音文本内容
        if "asr_text" in context and context["asr_text"]:
            asr_text = context['asr_text']
            if len(asr_text) > 10000:
                parts.append(f"广告中的语音内容为：{asr_text[:10000]}")
            else:
                parts.append(f"广告中的语音内容为：{asr_text}")
        
        return "。".join(parts) + "。" if parts else "广告信息缺失。"
    
    elif action_type == "直播间":
        parts = []
        
        # 直播标题和类别
        if "live_title" in context and context["live_title"]:
            live_intro = f"这是一个标题为{context['live_title']}的直播间"
        else:
            live_intro = "这是一个直播间"
        
        if "live_category" in context and context["live_category"]:
            live_intro += f"，直播类型是{context['live_category']} 场景"
        
        # 游戏直播的游戏名称
        if "live_game_name" in context and context["live_game_name"]:
            live_intro += f"，正在直播《{context['live_game_name']}》"
        
        if "is_shop_live" in context:
            if context['is_shop_live']:
                live_intro += "，主播正在进行带货直播"
            else:
                live_intro += "，这是一场娱乐性的非带货直播"
        
        parts.append(live_intro)
        
        # 人气数据
        popularity_info = []
        
        # 总用户数
        if "live_total_user_cnt" in context and context["live_total_user_cnt"]:
            user_cnt = context['live_total_user_cnt']
            if user_cnt >= 10000:
                popularity_info.append(f"累计 {user_cnt/10000:.1f}万 人次观看")
            else:
                popularity_info.append(f"累计 {user_cnt} 人次观看")
        
        # 总观看次数
        if "live_total_view_cnt" in context and context["live_total_view_cnt"]:
            view_cnt = context['live_total_view_cnt']
            if view_cnt >= 10000:
                popularity_info.append(f"总观看量达到 {view_cnt/10000:.1f}万")
            else:
                popularity_info.append(f"总观看量 {view_cnt}")
        
        if popularity_info:
            parts.append("该直播已有 " + "、".join(popularity_info))
        
        # 直播总时长（总观看时长）
        if "live_total_view_duration" in context and context["live_total_view_duration"]:
            duration_seconds = context['live_total_view_duration']
            duration_hours = duration_seconds / 3600
            if duration_hours >= 10000:
                parts.append(f"累计观看时长达到 {duration_hours/10000:.1f}万 小时")
            elif duration_hours >= 1:
                parts.append(f"累计观看时长 {duration_hours:.1f} 小时")
            else:
                duration_minutes = duration_seconds / 60
                parts.append(f"累计观看时长 {duration_minutes:.0f} 分钟")

        if "live_cover_content" in context and context["live_cover_content"]:
            parts.append(f"直播封面的内容是：{context['live_cover_content']}")
        
        # 互动数据
        interaction_info = []
        if "live_like_cnt" in context and context["live_like_cnt"] > 0:
            like_cnt = context['live_like_cnt']
            if like_cnt >= 10000:
                interaction_info.append(f"{like_cnt/10000:.1f}万个赞")
            else:
                interaction_info.append(f"{like_cnt}个赞")
        
        if "live_comment_cnt" in context and context["live_comment_cnt"] > 0:
            comment_cnt = context['live_comment_cnt']
            if comment_cnt >= 10000:
                interaction_info.append(f"{comment_cnt/10000:.1f}万条评论")
            else:
                interaction_info.append(f"{comment_cnt}条评论")
        
        if interaction_info:
            parts.append("直播间目前有 " + "、".join(interaction_info))
        
        # 商品信息（如果是带货直播）
        if "items" in context and context["items"] and len(context["items"]) > 0:
            items = context["items"]
            item_details = []
            for item in items:
                item_title = item.get("title", "")
                item_price = item.get("price", "")
                if item_title and item_price:
                    item_details.append(f"{item_title}售价 {item_price} 元")
                elif item_title:
                    item_details.append(f"{item_title}")
            if item_details:
                if len(item_details) == 1:
                    parts.append(f"主播正在推荐 1 件商品：{item_details[0]}")
                else:
                    parts.append(f"主播正在推荐 {len(item_details)} 件商品：{', '.join(item_details)}")
        
        return "。".join(parts) + "。" if parts else "直播信息缺失。"
    
    elif action_type == "搜索行为":
        parts = []
        
        # 从action中获取搜索关键词（因为有些数据在action里）
        keyword = None
        query_category = None
        
        # 优先从action中获取
        action_list = action.get("action", [])
        for act in action_list:
            if act.get("type") == "search":
                keyword = act.get("keyword")
                query_category = act.get("query_category")
                break
        
        # 如果action中没有，再从context获取
        if not keyword and "keyword" in context:
            keyword = context["keyword"]
        if not query_category and "query_category" in context:
            query_category = context["query_category"]
        
        if keyword:
            search_info = f"用户在搜索框中输入了关键词 {keyword}"
            if query_category:
                if query_category == "查询型":
                    search_info += "，希望查找相关信息"
                elif query_category == "浏览型":
                    search_info += "，想要浏览相关内容"
                else:
                    search_info += f"，搜索意图是{query_category}"
            parts.append(search_info)
        else:
            parts.append("用户进行了搜索")
        
        return "。".join(parts) + "。" if parts else "搜索信息缺失。"
    
    elif action_type == "电商客服对话":
        parts = []
        
        # 咨询类型
        if "ticket_category" in context and context["ticket_category"]:
            parts.append(f"这是一次{context['ticket_category']}类型的咨询")
        else:
            parts.append("这是一次客服对话")
        
        # 商品信息
        order_details = []
        if "product_name" in context and context["product_name"]:
            order_details.append(f"涉及商品为《{context['product_name']}》")
        
        if "product_category_info" in context and context["product_category_info"]:
            order_details.append(f"商品类别是{context['product_category_info']}")
        
        # 订单详情
        price_info = []
        if "item_price" in context and context["item_price"]:
            price_info.append(f"单价 {context['item_price']} 元")
        
        if "item_qty" in context and context["item_qty"]:
            qty = context['item_qty']
            try:
                if isinstance(qty, (int, float)):
                    if qty == int(qty):  # 如果是整数
                        price_info.append(f"购买数量 {int(qty)} 件")
                    else:
                        price_info.append(f"购买数量 {qty} 件")
                else:
                    price_info.append(f"购买数量 {qty} 件")
            except:
                price_info.append(f"购买数量 {qty} 件")
        
        if "express_fee" in context and context["express_fee"]:
            try:
                fee = float(context["express_fee"])
                if fee > 0:
                    price_info.append(f"运费 {fee} 元")
            except:
                pass
        
        if price_info:
            order_details.append("，".join(price_info))
        
        if order_details:
            parts.append("；".join(order_details))
        
        # 下单时间
        if "pay_order_time" in context and context["pay_order_time"]:
            parts.append(f"该订单的下单时间是 {context['pay_order_time']}")
        
        return "。".join(parts) + "。" if parts else "客服对话信息缺失。"
    
    return str(context)


def format_action_result(action: Dict) -> str:
    """格式化单个行为的结果（按场景类型分类处理）"""
    action_type = action.get("type", "")
    action_list = action.get("action", [])
    results = []
    
    # 视频浏览场景
    if action_type == "视频浏览":
        for act in action_list:
            act_type = act.get("type", "")
            
            if act_type == "watch":
                # 处理新格式：play_duration 可能是 "194秒" 这样的字符串，也可能是数字（如 11.75）
                watch_details = []
                
                if "play_duration" in act:
                    play_duration = act["play_duration"]
                    # 如果是字符串格式（如 "194秒"）
                    if isinstance(play_duration, str):
                        watch_details.append(f"观看了 {play_duration}")
                    # 如果是数字格式（秒数）
                    elif isinstance(play_duration, (int, float)):
                        watch_sec = play_duration
                        if watch_sec >= 60:
                            minutes = int(watch_sec // 60)
                            seconds = int(watch_sec % 60)
                            if seconds > 0:
                                watch_details.append(f"观看了 {minutes} 分 {seconds} 秒")
                            else:
                                watch_details.append(f"观看了 {minutes} 分钟")
                        else:
                            # 如果是整数就不显示小数
                            if watch_sec == int(watch_sec):
                                watch_details.append(f"观看了 {int(watch_sec)} 秒")
                            else:
                                watch_details.append(f"观看了 {watch_sec:.1f} 秒")
                elif "watch_seconds" in act:
                    watch_sec = act["watch_seconds"]
                    if watch_sec >= 60:
                        minutes = int(watch_sec // 60)
                        seconds = int(watch_sec % 60)
                        if seconds > 0:
                            results.append(f"观看了 {minutes} 分 {seconds} 秒")
                        else:
                            results.append(f"观看了 {minutes} 分钟")
                
                # 观看特征
                watch_features = []
                # 循环播放
                if act.get("played_loop_cnt"):
                    watch_features.append(f"循环播放了 {act.get('played_loop_cnt')}")
                # 快进播放
                if act.get("is_fast_forward_play"):
                    watch_features.append("使用了快进播放操作")
                # 回退播放
                if act.get("is_backward_play"):
                    watch_features.append("进行了回退观看操作")
                # 放大播放
                if act.get("is_enlarge_play"):
                    watch_features.append("放大了视频画面操作")
                # 完播标记：支持 completed 和 is_complete_play 两种字段
                if act.get("completed") or act.get("is_complete_play"):
                    watch_features.append("完整看完了整个视频")
                
                if watch_features:
                    watch_details.append("，".join(watch_features))
                
                if watch_details:
                    results.append("、".join(watch_details))
            
            elif act_type == "like":
                # 点赞行为（只要有 like action 就认为点赞了）
                results.append("对视频进行点赞操作")
            
            elif act_type == "comment":
                # 评论行为（只要有 comment action 就认为评论了）
                comment_detail = act.get("comment_detail_list", "")
                if comment_detail:
                    results.append(f"发表了评论：{comment_detail}")
                else:
                    results.append("发表了评论")
                # 评论停留时长
                if act.get("comment_stay_duration"):
                    results.append(f"在评论区停留了 {act.get('comment_stay_duration')}")
                if act.get("is_at_friend_in_comment"):
                    results.append("在评论区提及了好友")
            
            elif act_type == "share":
                # 分享行为（只要有 share action 就认为分享了）
                share_cnt = act.get("share_cnt")
                if share_cnt:
                    results.append(f"分享给朋友，成功分享了{share_cnt}次")
                else:
                    results.append("分享了视频")
            
            elif act_type == "collect":
                # 收藏行为（只要有 collect action 就认为收藏了）
                collect_cnt = act.get("collect_cnt")
                if collect_cnt:
                    results.append(f"收藏了该视频，成功收藏了{collect_cnt}次")
                else:
                    results.append("收藏了该视频")
            
            elif act_type == "download":
                # 下载行为（只要有 download action 就认为下载了）
                download_cnt = act.get("download_cnt")
                if download_cnt:
                    results.append(f"下载了该视频，成功下载了{download_cnt}次")
                else:
                    results.append("下载了该视频")
            
            elif act_type == "follow":
                # 关注视频作者（只要有 follow action 就认为关注了）
                results.append("关注了视频作者")
            
            elif act_type == "unfollow":
                # 取消关注视频作者
                if act.get("is_unfollow_action"):
                    results.append("取消关注了视频作者")
            
            elif act_type == "dislike":
                # 不喜欢（减少类似内容推荐）
                if act.get("reduce_similar_cnt") or act.get("reduce_simliar_cnt"):
                    results.append("选择了不感兴趣")
            
            elif act_type == "report":
                # 举报
                if act.get("report_cnt"):
                    results.append("进行了举报")
    
    elif action_type == "电影评分":
        for act in action_list:
            results.append(f"用户对此电影打分是 {act['rating']}")

    # 商城购物场景
    elif action_type == "商城购物":
        for act in action_list:
            act_type = act.get("type", "")
            
            if act_type == "cart":
                # is_add_to_cart 布尔值
                if act.get("is_add_to_cart"):
                    results.append("把商品加入了购物车")
                else:
                    results.append("浏览了商品但未加入购物车")
            
            elif act_type == "purchase":
                # 支持多个可能的字段：order_success, paid, is_pay
                if act.get("order_success") or act.get("paid") or act.get("is_pay"):
                    results.append("成功下单购买")
                else:
                    results.append("未购买该商品")
    
    # 广告推荐场景
    elif action_type == "广告推荐":
        for act in action_list:
            act_type = act.get("type", "")
            
            if act_type == "watch":
                # 观看时长
                watch_seconds = 0
                if "watch_seconds" in act:
                    watch_seconds = act.get("watch_seconds", 0)
                elif "play_duration" in act:
                    play_duration = act.get("play_duration")
                    if isinstance(play_duration, str):
                        try:
                            watch_seconds = float(play_duration.replace("秒", "").strip())
                        except:
                            watch_seconds = 0
                    elif isinstance(play_duration, (int, float)):
                        watch_seconds = float(play_duration)
                
                watch_features = []
                if watch_seconds > 0:
                    watch_features.append(f"观看了 {watch_seconds:.1f} 秒")
                
                # 循环播放
                if act.get("played_loop_cnt"):
                    watch_features.append(f"循环播放了 {act.get('played_loop_cnt')} 次")
                
                # 完播标记
                if act.get("is_complete_play"):
                    watch_features.append("看完了整个广告")
                
                if watch_features:
                    results.append("、".join(watch_features))
            
            elif act_type == "like":
                if act.get("like_cnt"):
                    results.append(f"为广告点赞 {act.get('like_cnt')} 次")
                else:
                    results.append("为广告点赞")
            
            elif act_type == "comment":
                # 只要有 comment action 就认为评论了
                results.append("在广告下发表了评论")
            
            elif act_type == "share":
                # 只要有 share action 就认为分享了
                share_cnt = act.get("share_cnt")
                if share_cnt:
                    results.append(f"分享了广告 {share_cnt} 次")
                else:
                    results.append("分享了广告")
            
            elif act_type == "conversion":
                if act.get("conversion_cnt"):
                    results.append(f"点击了广告产生转化，共 {act.get('conversion_cnt')} 次")
            
            elif act_type == "activation":
                if act.get("activation_cnt"):
                    results.append(f"激活了应用或服务，共 {act.get('activation_cnt')} 次")
            
            elif act_type == "purchase":
                # 广告推荐场景的购买行为
                purchase_info = []
                if act.get("pay_cnt"):
                    purchase_info.append(f"支付了 {act.get('pay_cnt')} 次")
                if act.get("pay_purchase_amt"):
                    purchase_info.append(f"支付金额 {act.get('pay_purchase_amt')} 元")
                if purchase_info:
                    results.append("通过广告" + "、".join(purchase_info))
            
            elif act_type == "submit":
                # 表单提交
                if act.get("form_submit_total_cnt"):
                    results.append(f"提交了表单 {act.get('form_submit_total_cnt')} 次")
            
            elif act_type == "follow":
                results.append("关注了广告主")
            
            elif act_type == "unfollow":
                results.append("取消关注了广告主")
            
            elif act_type == "dislike":
                if act.get("report_cnt"):
                    results.append("举报了广告")
    
    # 直播间场景
    elif action_type == "直播间":
        for act in action_list:
            act_type = act.get("type", "")
            
            if act_type == "watch":
                # 停留时长
                play_duration = act.get("play_duration", 0)
                if isinstance(play_duration, str):
                    try:
                        watch_seconds = float(play_duration.replace("秒", "").strip())
                    except:
                        watch_seconds = 0
                elif isinstance(play_duration, (int, float)):
                    watch_seconds = float(play_duration)
                else:
                    watch_seconds = 0
                
                watch_info = []
                if watch_seconds > 0:
                    if watch_seconds >= 60:
                        minutes = int(watch_seconds // 60)
                        seconds = int(watch_seconds % 60)
                        if seconds > 0:
                            watch_info.append(f"在直播间停留了 {minutes} 分 {seconds} 秒")
                        else:
                            watch_info.append(f"在直播间停留了 {minutes} 分钟")
                    else:
                        watch_info.append(f"在直播间停留了 {int(watch_seconds)} 秒")
                
                # 播放次数
                if act.get("play_count"):
                    watch_info.append(f"观看了 {act.get('play_count')} 次")
                
                if watch_info:
                    results.append("，".join(watch_info))
            
            elif act_type == "like":
                if act.get("like_cnt"):
                    like_cnt = act.get("like_cnt")
                    if like_cnt >= 10000:
                        results.append(f"为主播点赞 {like_cnt/10000:.1f}万 次")
                    else:
                        results.append(f"为主播点赞 {like_cnt} 次")
                else:
                    results.append("为主播点赞")
            
            elif act_type == "comment":
                comment_info = []
                if act.get("comment_cnt"):
                    cnt = act.get("comment_cnt")
                    comment_info.append(f"发送了 {cnt} 条弹幕")
                
                # 评论内容
                if act.get("comment_content"):
                    comments = act.get("comment_content")
                    if isinstance(comments, list) and len(comments) > 0:
                        # 显示前3条评论
                        display_comments = comments[:3]
                        comment_text = "、".join([f"「{c}」" for c in display_comments])
                        if len(comments) > 3:
                            comment_info.append(f"评论内容包括：{comment_text} 等")
                        else:
                            comment_info.append(f"评论内容：{comment_text}")
                
                if comment_info:
                    results.append("在直播间" + "，".join(comment_info))
                else:
                    results.append("在直播间发送了弹幕")
            
            elif act_type == "send_gift":
                if act.get("gift_amount"):
                    gift_amount = act.get("gift_amount")
                    results.append(f"给主播送了价值 {gift_amount} 元的礼物")
            
            elif act_type == "follow":
                if act.get("is_follow_action"):
                    results.append("关注了主播")
            
            elif act_type == "unfollow":
                if act.get("is_unfollow_action"):
                    results.append("取消关注了主播")
            
            elif act_type == "share":
                if act.get("share_cnt"):
                    results.append(f"分享了直播间 {act.get('share_cnt')} 次")
            
            elif act_type == "dislike":
                dislike_info = []
                if act.get("report_cnt"):
                    dislike_info.append("举报了直播间")
                if act.get("reduce_simliar_cnt"):
                    dislike_info.append("选择了不感兴趣")
                if dislike_info:
                    results.append("、".join(dislike_info))
            
            elif act_type == "click_cart":
                # 直播间点击购物车（加购）
                if act.get("is_click_cart_action"):
                    item_title = act.get("item_title", "")
                    item_price = act.get("item_price", "")
                    if item_title and item_price:
                        results.append(f"把商品 {item_title}（售价{item_price}元）加入了购物车")
                    elif item_title:
                        results.append(f"把商品 {item_title} 加入了购物车")
                    else:
                        results.append("点击了购物车")
            
            elif act_type == "join_group":
                if act.get("join_fans_group_cnt"):
                    cnt = act.get("join_fans_group_cnt")
                    results.append(f"加入了粉丝团（{cnt}次）")
    
    # 搜索行为场景
    elif action_type == "搜索行为":
        for act in action_list:
            act_type = act.get("type", "")
            
            if act_type == "search":
                # 搜索行为已在context中处理，这里不重复
                pass
            
            elif act_type == "show":
                # 展示搜索结果
                if act.get("show_cnt"):
                    results.append(f"系统展示了 {act.get('show_cnt')} 条搜索结果")
            
            elif act_type == "click":
                if act.get("click_cnt"):
                    results.append(f"点击了搜索结果 {act.get('click_cnt')} 次")
                else:
                    results.append("点击了搜索结果")
    
    # 电商客服对话场景
    elif action_type == "电商客服对话":
        for act in action_list:
            act_type = act.get("type", "")
            
            if act_type == "dialogue":
                # 对话轮次统计
                if act.get("content"):
                    dialogue_list = act.get("content")
                    if isinstance(dialogue_list, list):
                        user_msgs = sum(1 for msg in dialogue_list if msg.get("role") == "user")
                        assistant_msgs = sum(1 for msg in dialogue_list if msg.get("role") == "assistant")
                        results.append(f"与客服进行了对话（用户发送 {user_msgs} 条消息，客服回复 {assistant_msgs} 条）")
            
            elif act_type == "purchase":
                # 支付状态
                if act.get("paid"):
                    results.append("最终完成了支付")
                else:
                    results.append("未完成支付")
    
    # 其他场景或通用处理
    else:
        for act in action_list:
            act_type = act.get("type", "")
            
            if act_type == "watch":
                if "play_duration" in act or "watch_seconds" in act:
                    results.append("观看了内容")
            elif act_type == "like":
                results.append("点赞")
            elif act_type == "comment":
                results.append("评论")
            elif act_type == "share":
                results.append("分享")
            elif act_type == "collect":
                results.append("收藏")
            elif act_type == "download":
                results.append("下载")
            elif act_type == "purchase":
                results.append("购买")
            elif act_type == "click":
                results.append("点击")
    
    return "，".join(results) if results else "仅浏览未进行其他操作"


def should_filter_action(action: Dict) -> bool:
    """
    判断是否应该过滤掉这个行为
    
    过滤条件：
    1. watch行为的play_duration为0
    2. 广告推荐的show_cnt为0（展示次数为0说明用户没看到）
    3. 直播间场景中，context只有live_comment_cnt和live_like_cnt，且都是0（无效数据）
    
    Args:
        action: 行为记录
    
    Returns:
        True 表示应该过滤掉（不使用），False 表示保留
    """
    action_type = action.get("type", "")
    context = action.get("context", {})
    action_list = action.get("action", [])
    
    # 检查广告推荐的show_cnt
    if action_type == "广告推荐":
        show_cnt = context.get("show_cnt")
        if show_cnt is not None and show_cnt == 0:
            return True
    
    # 检查直播间场景的无效数据
    if action_type == "直播间":
        # 如果context只有live_comment_cnt和live_like_cnt这两个字段，且都是0，则过滤
        if (len(context) == 2 and 
            'live_comment_cnt' in context and 
            'live_like_cnt' in context and 
            context.get('live_comment_cnt') == 0 and 
            context.get('live_like_cnt') == 0):
            return True
    
    # 检查watch行为的play_duration
    for act in action_list:
        if act.get("type") == "watch":
            play_duration = act.get("play_duration")
            
            # 检查 play_duration 是否为 0
            if play_duration is not None:
                # 如果是字符串格式（如 "0秒"）
                if isinstance(play_duration, str):
                    try:
                        duration_value = float(play_duration.replace("秒", "").strip())
                        if duration_value == 0:
                            return True
                    except (ValueError, AttributeError):
                        pass
                # 如果是数字格式
                elif isinstance(play_duration, (int, float)):
                    if play_duration == 0:
                        return True
            
            # 也检查 watch_seconds 字段
            watch_seconds = act.get("watch_seconds")
            if watch_seconds is not None and watch_seconds == 0:
                return True
    
    return False


def estimate_token_count(text: str) -> int:
    """
    估算文本的token数量
    
    如果Qwen tokenizer可用，使用Qwen的tokenizer进行准确计数
    否则使用简单估算：中文约1.5 token/字符，英文约1.3 token/单词
    """
    if TOKENIZER_AVAILABLE and QWEN_TOKENIZER is not None:
        try:
            return len(QWEN_TOKENIZER.encode(text))
        except Exception as e:
            print(f"Warning: Qwen tokenizer encoding failed: {e}, using approximation")
    
    # 简单估算方法
    # 中文字符
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    # 英文单词（粗略估计：非中文字符 / 5）
    other_chars = len(text) - chinese_chars
    english_words = other_chars / 5
    
    # 中文：1.5 token/字符，英文：1.3 token/单词
    estimated_tokens = int(chinese_chars * 1.5 + english_words * 1.3)
    return estimated_tokens


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """
    将文本截断到指定的 token 数量限制
    
    使用二分查找找到合适的截断点，确保不超过 max_tokens
    """
    if not text or max_tokens <= 0:
        return text
    
    current_tokens = estimate_token_count(text)
    if current_tokens <= max_tokens:
        return text
    
    # 使用二分查找找到合适的截断点
    left, right = 0, len(text)
    best_length = 0
    
    while left < right:
        mid = (left + right + 1) // 2
        truncated = text[:mid]
        tokens = estimate_token_count(truncated)
        
        if tokens <= max_tokens:
            best_length = mid
            left = mid
        else:
            right = mid - 1
    
    # 在最后一个完整句子或行处截断（如果可能）
    truncated_text = text[:best_length]
    
    # 尝试在句号、换行符处截断以保持完整性
    last_break = -1
    for sep in ['\n\n', '\n', '。', '！', '？', '.', '!', '?']:
        pos = truncated_text.rfind(sep)
        if pos > last_break and pos > best_length * 0.7:  # 至少保留 70% 的内容
            last_break = pos + len(sep)
    
    if last_break > 0:
        truncated_text = truncated_text[:last_break]
    
    # 添加截断提示
    truncated_text = truncated_text.rstrip() + "\n\n[... 内容已截断 ...]"
    
    return truncated_text


def get_actual_used_history(
    action_history: List[Dict], 
    max_history_tokens: int = None,
    max_history_days: int = None,
    reference_timestamp: str = None
) -> Dict:
    """
    获取实际用于 prompt 的历史行为列表（经过过滤和截断）
    
    支持两种截断方式（可同时使用，同时满足两个条件）：
    1. 基于 token 数限制（max_history_tokens）
    2. 基于天数限制（max_history_days），只保留参考时间点前 N 天内的行为
    
    Args:
        action_history: 原始历史行为列表
        max_history_tokens: 最大 token 数限制，如果为 None 则使用 config.MAX_HISTORY_TOKENS
        max_history_days: 只保留近 N 天的历史行为，如果为 None 则不限制天数
        reference_timestamp: 参考时间戳（用于计算 N 天前），通常是测试行为的时间戳
                           格式：YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS
                           如果为 None 且设置了 max_history_days，则使用当前时间
    
    Returns:
        {
            "original_count": int,        # 原始行为数量
            "filtered_count": int,        # 过滤后的行为数量
            "actual_used_count": int,     # 实际使用的行为数量（截断后）
            "actual_used_tokens": int,    # 实际使用的 token 数量
            "actual_used_actions": List,  # 实际使用的行为列表
            "days_filtered_count": int,   # 因天数限制被过滤的行为数量
        }
    """
    from datetime import datetime, timedelta
    
    if not action_history:
        return {
            "original_count": 0,
            "filtered_count": 0,
            "actual_used_count": 0,
            "actual_used_tokens": 0,
            "actual_used_actions": [],
            "days_filtered_count": 0,
        }
    
    original_count = len(action_history)
    
    # 过滤掉 play_duration 为 0 的行为
    filtered_history = [action for action in action_history if not should_filter_action(action)]
    filtered_count = len(filtered_history)
    
    if not filtered_history:
        return {
            "original_count": original_count,
            "filtered_count": 0,
            "actual_used_count": 0,
            "actual_used_tokens": 0,
            "actual_used_actions": [],
            "days_filtered_count": 0,
        }
    
    # 如果设置了天数限制，先按天数过滤
    days_filtered_count = 0
    if max_history_days is not None and max_history_days > 0:
        # 解析参考时间戳（必须提供，通常是待预测行为的时间戳）
        if reference_timestamp:
            try:
                # 尝试解析多种格式
                if len(reference_timestamp) == 10:  # YYYY-MM-DD
                    ref_datetime = datetime.strptime(reference_timestamp, "%Y-%m-%d")
                else:  # YYYY-MM-DD HH:MM:SS
                    ref_datetime = datetime.strptime(reference_timestamp[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                # 解析失败，跳过天数过滤
                print(f"警告: 无法解析参考时间戳 '{reference_timestamp}'，跳过天数过滤")
                ref_datetime = None
        else:
            # 没有提供参考时间戳，跳过天数过滤
            ref_datetime = None
        
        # 只有成功解析参考时间戳时才进行天数过滤
        if ref_datetime is not None:
            # 计算截止时间（参考时间点前 N 天）
            cutoff_datetime = ref_datetime - timedelta(days=max_history_days)
            cutoff_str = cutoff_datetime.strftime("%Y-%m-%d %H:%M:%S")
            
            # 过滤掉超出天数范围的行为
            days_filtered_history = []
            for action in filtered_history:
                action_timestamp = action.get("timestamp", "")
                if action_timestamp and action_timestamp >= cutoff_str:
                    days_filtered_history.append(action)
            
            days_filtered_count = filtered_count - len(days_filtered_history)
            filtered_history = days_filtered_history
    
    if not filtered_history:
        return {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "actual_used_count": 0,
            "actual_used_tokens": 0,
            "actual_used_actions": [],
            "days_filtered_count": days_filtered_count,
        }
    
    # 使用 token 限制
    if max_history_tokens is None:
        max_history_tokens = MAX_HISTORY_TOKENS
    
    display_actions = []
    current_tokens = 0
    
    # 从最新的行为开始往前加
    for action in reversed(filtered_history):
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
        
        action_tokens = estimate_token_count(action_text)
        
        if current_tokens + action_tokens <= max_history_tokens:
            display_actions.insert(0, action)
            current_tokens += action_tokens
        else:
            break
    
    return {
        "original_count": original_count,
        "filtered_count": filtered_count,
        "actual_used_count": len(display_actions),
        "actual_used_tokens": current_tokens,
        "actual_used_actions": display_actions,
        "days_filtered_count": days_filtered_count,
    }


def build_history_summary(
    action_history: List[Dict], 
    max_history_tokens: int = None,
    max_history_days: int = None,
    reference_timestamp: str = None,
    history_process_mode: str = None,
    current_action: Dict = None
) -> str:
    """
    构建历史行为摘要（基于 Token 数限制和/或天数限制）
    
    支持两种截断方式（可同时使用）：
    1. 基于 token 数限制（max_history_tokens）
    2. 基于天数限制（max_history_days），只保留参考时间点前 N 天内的行为
    
    支持三种历史处理模式：
    1. none: 不做额外处理，直接使用截断后的历史
    2. summary: 使用 LangChain 对**全部历史**进行摘要，然后截断到限制长度
    3. rag: 使用 LangChain 对**全部历史**进行检索，返回最相关的行为（在限制长度内）
    
    过滤掉 play_duration 为 0 的行为记录
    
    Args:
        action_history: 历史行为列表
        max_history_tokens: 最大 token 数限制，如果为 None 则使用 config.MAX_HISTORY_TOKENS
        max_history_days: 只保留近 N 天的历史行为，如果为 None 则不限制天数
        reference_timestamp: 参考时间戳（用于计算 N 天前），通常是测试行为的时间戳
        history_process_mode: 历史处理模式，可选 "none", "summary", "rag"，默认使用全局配置
        current_action: 当前待预测的行为（RAG 模式需要用于检索）
    """
    # 检查是否需要使用 RAG/Summary 处理
    mode = history_process_mode or (_get_history_process_mode() if HISTORY_PROCESSOR_AVAILABLE else "none")
    
    if max_history_tokens is None:
        max_history_tokens = MAX_HISTORY_TOKENS
    
    # RAG/Summary 模式：使用全部历史（不预先截断）
    if mode != "none" and HISTORY_PROCESSOR_AVAILABLE:
        # 只过滤无效行为，不按 token 截断
        filtered_history = [action for action in action_history if not should_filter_action(action)]
        
        if not filtered_history:
            if len(action_history) == 0:
                return "这个用户目前还没有任何历史行为记录。"
            else:
                return "这个用户目前还没有任何有效的历史行为记录。"
        
        # 使用 LangChain 处理全部历史行为
        # RAG/Summary 处理器内部会处理长度限制
        processed_text, used = process_history(
            action_history=filtered_history,  # 传入全部过滤后的历史
            current_action=current_action,
            mode=mode,
            max_output_tokens=max_history_tokens  # 传入长度限制
        )
        if used and processed_text:
            return processed_text
        # 如果处理失败，降级到默认模式
        print(f"⚠️  {mode} 处理失败，降级到默认模式")
    
    # 默认模式（none）：使用截断后的历史
    history_info = get_actual_used_history(
        action_history, 
        max_history_tokens,
        max_history_days=max_history_days,
        reference_timestamp=reference_timestamp
    )
    display_actions = history_info["actual_used_actions"]
    current_tokens = history_info["actual_used_tokens"]
    
    if not display_actions:
        if history_info["original_count"] == 0:
            return "这个用户目前还没有任何历史行为记录。"
        else:
            return "这个用户目前还没有任何有效的历史行为记录。"
    
    # 原始格式化输出（默认模式）
    lines = []
    lines.append(f"以下是该用户最近的 {len(display_actions)} 条行为记录（约 {current_tokens} tokens，按时间从早到晚排列）：\n")
    
    for i, action in enumerate(display_actions, 1):
        timestamp = action.get("timestamp", "未知时间")
        action_type = action.get("type", "未知行为")
        context_str = format_action_context(action)
        result_str = format_action_result(action)
        
        lines.append(
            f"【行为 {i}】时间：{timestamp}\n"
            f"  场景：{action_type}\n"
            f"  详情：{context_str}\n"
            f"  反应：{result_str}\n"
        )
    
    return "\n".join(lines)


def build_test_action_description(action: Dict) -> str:
    """构建待预测行为的场景描述"""
    timestamp = action.get("timestamp", "未知时间")
    action_type = action.get("type", "未知行为")
    context_str = format_action_context(action)
    
    description = (
        f"现在时间是 {timestamp}，该用户遇到了一个【{action_type}】场景。\n"
        f"场景详细信息如下：\n{context_str}"
    )
    
    return description


def build_prediction_questions(action: Dict) -> List[Dict]:
    """
    根据行为类型构建预测问题
    
    Returns:
        List of {
            "question": str,  # 问题描述
            "type": "binary" or "continuous" or "text",  # 问题类型
            "field": str,  # 真实值在action中的字段路径
            "true_value": any,  # 真实值
        }
    """
    questions = []
    action_type = action.get("type", "")
    action_list = action.get("action", [])
    
    # 视频浏览 - 主要预测观看时长和完播
    if action_type == "视频浏览":
        context = action.get("context", {})
        
        # 提取视频时长（可能是数字格式或字符串格式）
        duration_raw = context.get("duration", 0)
        try:
            if isinstance(duration_raw, (int, float)):
                video_duration = float(duration_raw)
            elif isinstance(duration_raw, str):
                video_duration = float(duration_raw.replace("秒", "").strip()) if duration_raw else 0
            else:
                video_duration = 0
        except:
            video_duration = 0
        
        # 格式化 duration_str 用于显示
        if video_duration > 0:
            duration_str = f"{int(video_duration)}秒"
        else:
            duration_str = "未知时长"
        
        # 提取实际观看时长（可能是字符串格式 "194秒" 或数字格式 11.75）
        watch_seconds = 0
        play_duration_str = ""
        for act in action_list:
            if act.get("type") == "watch":
                play_duration = act.get("play_duration", 0)
                # 如果是字符串格式（如 "194秒"）
                if isinstance(play_duration, str):
                    play_duration_str = play_duration
                    try:
                        watch_seconds = float(play_duration.replace("秒", "").strip()) if play_duration else 0
                    except:
                        watch_seconds = 0
                # 如果是数字格式
                elif isinstance(play_duration, (int, float)):
                    watch_seconds = float(play_duration)
                    play_duration_str = f"{watch_seconds}秒"
                break
        
        # 判断是否完播：优先使用 is_complete_play 字段，其次根据观看时长判断
        completed = False
        for act in action_list:
            if act.get("type") == "watch":
                # 优先使用 is_complete_play 字段
                if "is_complete_play" in act:
                    completed = act.get("is_complete_play", False)
                # 如果没有 is_complete_play 字段，则根据观看时长判断
                elif video_duration > 0:
                    completed = watch_seconds >= video_duration
                break
        
        # 连续值：观看时长（始终预测）
        questions.append({
            "type": "continuous",
            "field": "video_watch_seconds",
            "true_value": watch_seconds,
            "video_duration": video_duration,
        })
        
        # 二分类：是否完播
        questions.append({
            "type": "binary",
            "field": "video_completed",
            "true_value": 1 if completed else 0,
        })
        
        # 提取互动行为
        liked = False
        commented = False
        shared = False
        collected = False
        followed = False
        
        for act in action_list:
            act_type = act.get("type", "")
            if act_type == "like":
                liked = True
            elif act_type == "comment":
                commented = True
            elif act_type == "share":
                shared = True
            elif act_type == "collect":
                collected = True
            elif act_type == "follow":
                followed = True
        
        # 二分类：是否点赞
        questions.append({
            "type": "binary",
            "field": "video_liked",
            "true_value": 1 if liked else 0,
        })
        
        # 二分类：是否评论
        questions.append({
            "type": "binary",
            "field": "video_commented",
            "true_value": 1 if commented else 0,
        })
        
        # 二分类：是否分享
        questions.append({
            "type": "binary",
            "field": "video_shared",
            "true_value": 1 if shared else 0,
        })
        
        # 二分类：是否收藏
        questions.append({
            "type": "binary",
            "field": "video_collected",
            "true_value": 1 if collected else 0,
        })
        
        # 二分类：是否关注作者
        questions.append({
            "type": "binary",
            "field": "video_followed",
            "true_value": 1 if followed else 0,
        })
    
    elif action_type == "电影评分":
        for act in action_list:
            questions.append({
                "type": "multi_class",
                "field": "movie_rating",
                "true_value": act['rating'],
            })

    # 商城购物 - 预测是否下单和加购
    elif action_type == "商城购物":
        # 二分类：是否加入购物车（需检查 is_add_to_cart 字段）
        added_to_cart = False
        for act in action_list:
            if act.get("type") == "cart" and act.get("is_add_to_cart"):
                added_to_cart = True
                break
        
        questions.append({
            "type": "binary",
            "field": "shop_added_to_cart",
            "true_value": 1 if added_to_cart else 0,
        })
        
        # 二分类：是否下单购买（支持 is_pay、paid、order_success 字段）
        order_success = False
        for act in action_list:
            if act.get("type") == "purchase":
                # 检查多个可能的字段
                if act.get("is_pay") or act.get("paid") or act.get("order_success"):
                    order_success = True
                    break
        
        questions.append({
            "type": "binary",
            "field": "shop_order_success",
            "true_value": 1 if order_success else 0,
        })
    
    # 广告推荐 - 预测观看、互动、转化和激活
    elif action_type == "广告推荐":
        # 连续值：观看时长（可能在 watch_seconds 或 play_duration 中）
        watch_seconds = 0
        for act in action_list:
            if act.get("type") == "watch":
                # 优先使用 watch_seconds
                if "watch_seconds" in act:
                    watch_seconds = act.get("watch_seconds", 0)
                # 如果没有 watch_seconds，尝试从 play_duration 提取
                elif "play_duration" in act:
                    play_duration = act.get("play_duration")
                    if isinstance(play_duration, str):
                        try:
                            watch_seconds = float(play_duration.replace("秒", "").strip())
                        except:
                            watch_seconds = 0
                    elif isinstance(play_duration, (int, float)):
                        watch_seconds = float(play_duration)
                break
        
        if watch_seconds > 0 or any(act.get("type") == "watch" for act in action_list):
            questions.append({
                "type": "continuous",
                "field": "ad_watch_seconds",
                "true_value": watch_seconds,
            })
        
        # 二分类：是否点赞
        liked = False
        for act in action_list:
            if act.get("type") == "like":
                liked = True
                break
        
        questions.append({
            "type": "binary",
            "field": "ad_liked",
            "true_value": 1 if liked else 0,
        })
        
        # 二分类：是否评论
        commented = False
        for act in action_list:
            if act.get("type") == "comment":
                commented = True
                break
        
        questions.append({
            "type": "binary",
            "field": "ad_commented",
            "true_value": 1 if commented else 0,
        })
        
        # 二分类：是否激活
        activated = False
        for act in action_list:
            if act.get("type") == "activation" and act.get("activation_cnt", 0) > 0:
                activated = True
                break
        
        questions.append({
            "type": "binary",
            "field": "ad_activated",
            "true_value": 1 if activated else 0,
        })
        
        # 二分类：是否提交表单
        form_submitted = False
        for act in action_list:
            if act.get("type") == "submit" and act.get("form_submit_total_cnt", 0) > 0:
                form_submitted = True
                break
        
        questions.append({
            "type": "binary",
            "field": "ad_form_submitted",
            "true_value": 1 if form_submitted else 0,
        })
    
    # 直播间 - 预测停留时长和互动行为
    elif action_type == "直播间":
        # 连续值：停留时长（从 play_duration 字段获取）
        watch_seconds = 0
        for act in action_list:
            if act.get("type") == "watch":
                play_duration = act.get("play_duration", 0)
                if isinstance(play_duration, str):
                    try:
                        watch_seconds = float(play_duration.replace("秒", "").strip())
                    except:
                        watch_seconds = 0
                elif isinstance(play_duration, (int, float)):
                    watch_seconds = float(play_duration)
                break
        
        questions.append({
            "type": "continuous",
            "field": "live_watch_seconds",
            "true_value": watch_seconds,
        })
        
        # 二分类：是否点赞（只要有 like action 就认为点赞了）
        liked = False
        for act in action_list:
            if act.get("type") == "like":
                liked = True
                break
        
        questions.append({
            "type": "binary",
            "field": "live_liked",
            "true_value": 1 if liked else 0,
        })
        
        # 二分类：是否评论（只要有 comment action 就认为评论了）
        commented = False
        for act in action_list:
            if act.get("type") == "comment":
                commented = True
                break
        
        questions.append({
            "type": "binary",
            "field": "live_commented",
            "true_value": 1 if commented else 0,
        })
        
        # 二分类：是否送礼物（只要有 send_gift action 就认为送礼物了）
        sent_gift = False
        for act in action_list:
            if act.get("type") == "send_gift":
                sent_gift = True
                break
        
        questions.append({
            "type": "binary",
            "field": "live_sent_gift",
            "true_value": 1 if sent_gift else 0,
        })
        
        # 二分类：是否关注主播（只要有 follow action 就认为关注了）
        followed = False
        for act in action_list:
            if act.get("type") == "follow":
                followed = True
                break
        
        questions.append({
            "type": "binary",
            "field": "live_followed",
            "true_value": 1 if followed else 0,
        })
        
        # 二分类：是否分享（只要有 share action 就认为分享了）
        shared = False
        for act in action_list:
            if act.get("type") == "share":
                shared = True
                break
        
        questions.append({
            "type": "binary",
            "field": "live_shared",
            "true_value": 1 if shared else 0,
        })
        
        # 如果是带货直播，预测是否加购/购买
        if action.get("context", {}).get("is_shop_live"):
            # 二分类：是否点击购物车（加购）
            clicked_cart = False
            for act in action_list:
                if act.get("type") == "click_cart" and act.get("is_click_cart_action"):
                    clicked_cart = True
                    break
            
            questions.append({
                "type": "binary",
                "field": "live_clicked_cart",
                "true_value": 1 if clicked_cart else 0,
            })
    
    # 搜索行为 - 预测搜索关键词（文本预测）
    elif action_type == "搜索行为":
        # 提取搜索关键词
        keyword = None
        query_category = None
        
        for act in action_list:
            if act.get("type") == "search":
                keyword = act.get("keyword")
                query_category = act.get("query_category")
                break
        
        # 如果有关键词，添加关键词预测问题
        if keyword:
            questions.append({
                "type": "text",
                "field": "search_keyword",
                "true_value": keyword,
                "query_category": query_category,
            })
    
    # 电商客服对话 - 支持：预测用户下一句话（BLEU/CharF1）
    elif action_type == "电商客服对话":
        # 定义需要跳过的特殊字符/内容模式
        # 包含"评价"（如"评价邀请已发出"、"感谢您的评价"）、"会话转移"等
        SKIP_PATTERNS = [
            "评价",  # 匹配所有包含"评价"的内容
            "会话转移",
        ]
        
        def should_skip_user_message(content: str) -> bool:
            """判断用户消息是否应该跳过"""
            if not content or not content.strip():
                return True
            # 检查是否包含需要跳过的特殊字符
            for pattern in SKIP_PATTERNS:
                if pattern in content:
                    return True
            return False
        
        def is_too_short(content: str) -> bool:
            """判断消息是否太短（字数<=2）"""
            if not content:
                return True
            # 去除空白字符后计算长度
            return len(content.strip()) <= 2
        
        # 提取对话历史
        dialogue_content = []
        for act in action_list:
            if act.get("type") == "dialogue":
                dialogue_content = act.get("content", [])
                break
        
        if dialogue_content and len(dialogue_content) > 0:
            # 从最后向前遍历，找到符合条件的用户发言
            # 策略：跳过包含特殊字符的，如果字数<=2则继续往前找
            target_user_message = None
            target_user_message_idx = -1  # 记录目标消息在列表中的索引
            
            for i in range(len(dialogue_content) - 1, -1, -1):
                msg = dialogue_content[i]
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    
                    # 检查是否包含特殊字符（跳过）
                    if should_skip_user_message(content):
                        continue
                    
                    # 检查字数是否太短（<=2，继续往前找）
                    if is_too_short(content):
                        continue
                    
                    # 找到符合条件的用户发言
                    target_user_message = content
                    target_user_message_idx = i
                    break
            
            # 如果没有找到符合条件的用户发言，跳过当前行为（不生成问题）
            if target_user_message is None:
                pass  # 返回空的 questions 列表
            else:
                # 检查目标用户发言是否是第一个用户发言（没有对话历史）
                # 判断方法：目标消息之前是否有任何对话（用户或客服的发言）
                has_dialogue_history = False
                has_prior_user_speech = False
                context_dialogue = []
                
                for j in range(target_user_message_idx):
                    msg = dialogue_content[j]
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    # 只要有用户或客服的发言，就认为有对话历史
                    if role in ["user", "assistant"] and content and content.strip():
                        has_dialogue_history = True
                        context_dialogue.append(msg)
                        # 检查是否有之前的用户发言
                        if role == "user":
                            has_prior_user_speech = True
                
                # 获取 context 信息（订单详情等）
                action_context = action.get("context", {})
                context_info = format_action_context(action)
                
                if not has_dialogue_history:
                    # 用户的第一个发言（没有对话历史），只提供context中包含的内容
                    question_text = (
                        f"这是一次电商客服对话场景。用户即将开始与客服对话。\n\n"
                        f"【订单/咨询背景信息】\n{context_info}\n\n"
                        f"请你站在这个用户的角度，结合订单信息和咨询背景，预测用户最有可能说的第一句话是什么？\n\n"
                        f"请直接输出用户会说的话（不要加引号，直接输出内容即可）："
                    )
                else:
                    # 有对话历史，提供历史对话（限制 token 数，避免超过模型上下文长度）
                    # 对话历史最多使用 4000 tokens，从最近的对话开始往前截取
                    MAX_DIALOGUE_TOKENS = 4000
                    
                    # 从最近的对话开始往前构建，直到达到 token 限制
                    truncated_dialogue = []
                    current_dialogue_tokens = 0
                    
                    for msg in reversed(context_dialogue):
                        msg_text = f"{'用户' if msg.get('role')=='user' else '客服'}: {msg.get('content', '')}\n"
                        msg_tokens = estimate_token_count(msg_text)
                        
                        if current_dialogue_tokens + msg_tokens <= MAX_DIALOGUE_TOKENS:
                            truncated_dialogue.insert(0, msg)
                            current_dialogue_tokens += msg_tokens
                        else:
                            break
                    
                    dialogue_text = "\n".join([
                        f"{'用户' if m.get('role')=='user' else '客服'}: {m.get('content', '')}" 
                        for m in truncated_dialogue
                    ])
                    
                    # 如果对话被截断，添加提示
                    if len(truncated_dialogue) < len(context_dialogue):
                        dialogue_text = f"（以下是最近 {len(truncated_dialogue)} 轮对话，更早的 {len(context_dialogue) - len(truncated_dialogue)} 轮已省略）\n" + dialogue_text
                    
                    question_text = (
                        f"这是一段电商客服对话记录。\n\n"
                        f"【订单/咨询背景信息】\n{context_info}\n\n"
                        f"【对话历史记录】\n{dialogue_text}\n\n"
                        f"请你站在这个用户的角度，结合TA的沟通风格、当前遇到的问题以及对话的上下文，预测TA接下来最有可能说的一句话是什么？\n\n"
                        f"请直接输出用户接下来会说的话（不要加引号，直接输出内容即可）："
                    )
                
                questions.append({
                    "question": question_text,
                    "type": "text",
                    "field": "next_user_message",
                    "true_value": target_user_message,
                    "context_dialogue": context_dialogue,
                    "has_dialogue_history": has_dialogue_history,
                    "has_prior_user_speech": has_prior_user_speech,
                })
    
    return questions


def build_movie_prompt(
    user_profile: str,
    action_history: List[Dict],
    test_action: Dict,
    question_info: Dict,
    max_history_tokens: int = None,
    max_history_days: int = None,
) -> Dict:
    """
    构建单个二分类问题的Prompt（只让模型输出 Yes 或 No）
    
    Args:
        user_profile: 用户画像
        action_history: 历史行为列表
        test_action: 待预测的测试行为
        question_info: 单个问题的信息，包含 question, type, field, true_value 等
        max_history_tokens: 最大 token 数限制，如果为 None 则使用 config.MAX_HISTORY_TOKENS
        max_history_days: 只保留近 N 天的历史行为，如果为 None 则不限制天数
    
    Returns:
        {
            "prompt": str,  # 完整的prompt
            "question_info": Dict,  # 原始问题信息
            "test_action": Dict,  # 原始的待预测行为
        }
        如果测试行为应该被过滤，返回 None
    """
    # 检查测试行为是否应该被过滤
    if should_filter_action(test_action):
        return None
    
    # 获取测试行为的时间戳作为参考时间点
    reference_timestamp = test_action.get("timestamp")
    
    # 构建历史行为摘要（内部会自动过滤）
    # 传递 current_action 以支持 RAG 模式的相关性检索
    history_summary = build_history_summary(
        action_history, 
        max_history_tokens=max_history_tokens,
        max_history_days=max_history_days,
        reference_timestamp=reference_timestamp,
        current_action=test_action
    )
    
    # 构建待预测场景描述
    scenario_desc = build_test_action_description(test_action)
    
    # 处理 user_profile，只保留第一个句号前的内容
    if user_profile:
        first_period_idx = user_profile.find('。')
        if first_period_idx != -1:
            user_profile_short = user_profile[:first_period_idx]
        else:
            user_profile_short = user_profile
    else:
        user_profile_short = ""
    
    # 组装完整prompt
    prompt_parts = [
        "你的核心任务是：基于给定的历史行为序列，推断该观众的电影偏好，并据此模拟TA对当前电影的评分。分值是1到5，五档，1最低，5最高",
        "## 输入一：用户画像",
        "这是该用户的基本信息，可以作为理解用户背景的参考：",
        user_profile_short,
        "## 输入二：历史行为轨迹信息",
        "这是该用户在过去一段时间内的对电影打分的情况，满分是5分。请仔细分析这些行为背后的动机和倾向，挖掘其隐含的长期偏好和短期意图：",
        history_summary,
        "## 输入三：当前测试场景",
        "用户现在遇到了以下场景：",
        scenario_desc,
        "## 预测任务",
        "请代入该用户视角，判断他会打几分，选项包括[1,2,3,4,5]，1是最低分，5是最高分",
        "## 输出要求",
        "**请只输出分数，也就是一个阿拉伯数字，不要输出任何其他内容、解释或分析。**",
        "你的回答：",
    ]
    
    prompt = "\n".join(prompt_parts)
    
    return {
        "prompt": prompt,
        "question_info": question_info,
        "test_action": test_action,
        **({"scenario_desc": scenario_desc} if "scenario_desc" in locals() else {}),
    }


def build_single_binary_prompt(
    user_profile: str,
    action_history: List[Dict],
    test_action: Dict,
    question_info: Dict,
    max_history_tokens: int = None,
    max_history_days: int = None,
) -> Dict:
    """
    构建单个二分类问题的Prompt（只让模型输出 Yes 或 No）
    
    Args:
        user_profile: 用户画像
        action_history: 历史行为列表
        test_action: 待预测的测试行为
        question_info: 单个问题的信息，包含 question, type, field, true_value 等
        max_history_tokens: 最大 token 数限制，如果为 None 则使用 config.MAX_HISTORY_TOKENS
        max_history_days: 只保留近 N 天的历史行为，如果为 None 则不限制天数
    
    Returns:
        {
            "prompt": str,  # 完整的prompt
            "question_info": Dict,  # 原始问题信息
            "test_action": Dict,  # 原始的待预测行为
        }
        如果测试行为应该被过滤，返回 None
    """
    # 检查测试行为是否应该被过滤
    if should_filter_action(test_action):
        return None
    
    # 获取测试行为的时间戳作为参考时间点
    reference_timestamp = test_action.get("timestamp")
    
    # 构建历史行为摘要（内部会自动过滤）
    # 传递 current_action 以支持 RAG 模式的相关性检索
    history_summary = build_history_summary(
        action_history, 
        max_history_tokens=max_history_tokens,
        max_history_days=max_history_days,
        reference_timestamp=reference_timestamp,
        current_action=test_action
    )
    
    # 构建待预测场景描述
    scenario_desc = build_test_action_description(test_action)
    
    # 根据 field 获取 Yes/No 问题格式
    yes_no_question = _get_yes_no_question_by_field(question_info)
    
    # 处理 user_profile，只保留第一个句号前的内容
    if user_profile:
        first_period_idx = user_profile.find('。')
        if first_period_idx != -1:
            user_profile_short = user_profile[:first_period_idx]
        else:
            user_profile_short = user_profile
    else:
        user_profile_short = ""
    
    # 组装完整prompt
    prompt_parts = [
        "你是快手平台的一位真实用户。你的核心任务是：基于给定的历史行为序列，推断该用户的兴趣偏好、消费水平和性格特征，并据此模拟TA在当前某场景下的真实决策。",
        "## 核心原则",
        "1. **数据驱动**：所有推断必须基于历史行为数据中的客观证据，避免无根据的臆测和假设。",
        "2. **行为连贯性**：新决策应与用户的历史行为模式保持内在逻辑一致，体现其稳定的偏好和习惯。",
        "3. **个体差异性**：充分尊重每个用户的独特性，不套用刻板印象或群体标签，从数据中发现真实的个性特征。",
        "4. **情境敏感性**：决策预测需考虑当前场景的特殊性，平衡长期偏好与短期情境因素的影响。",
        "5. **真实性优先**：模拟真实用户可能做出的选择，包括不感兴趣、犹豫、跳过等消极行为，而非总是给出积极响应。",
        "## 输入一：用户画像",
        "这是该用户的平台基本信息，可以作为理解用户背景的参考：",
        user_profile_short,
        "## 输入二：历史行为轨迹信息",
        "这是该用户在过去一段时间内的真实操作记录（包含直播、商城、视频、广告等跨场景行为）。请仔细分析这些行为背后的动机和倾向，挖掘其隐含的长期偏好和短期意图：",
        history_summary,
        "## 输入三：当前测试场景",
        "用户现在遇到了以下场景：",
        scenario_desc,
        "## 预测任务",
        "请代入该用户视角，回答以下问题：",
        yes_no_question,
        "## 输出要求",
        "**请只输出 Yes 或 No，不要输出任何其他内容、解释或分析。**",
        "你的回答：",
    ]
    
    prompt = "\n".join(prompt_parts)
    
    return {
        "prompt": prompt,
        "question_info": question_info,
        "test_action": test_action,
        **({"scenario_desc": scenario_desc} if "scenario_desc" in locals() else {}),
        **({"yes_no_question": yes_no_question} if "yes_no_question" in locals() else {}),

    }


def build_single_continuous_prompt(
    user_profile: str,
    action_history: List[Dict],
    test_action: Dict,
    question_info: Dict,
    max_history_tokens: int = None,
    max_history_days: int = None,
) -> Dict:
    """
    构建单个连续值问题的Prompt（让模型输出一个数字）
    
    Args:
        user_profile: 用户画像
        action_history: 历史行为列表
        test_action: 待预测的测试行为
        question_info: 单个问题的信息，包含 question, type, field, true_value 等
        max_history_tokens: 最大 token 数限制
        max_history_days: 只保留近 N 天的历史行为，如果为 None 则不限制天数
    
    Returns:
        {
            "prompt": str,  # 完整的prompt
            "question_info": Dict,  # 原始问题信息
            "test_action": Dict,  # 原始的待预测行为
        }
        如果测试行为应该被过滤，返回 None
    """
    # 检查测试行为是否应该被过滤
    if should_filter_action(test_action):
        return None
    
    # 获取测试行为的时间戳作为参考时间点
    reference_timestamp = test_action.get("timestamp")
    
    # 构建历史行为摘要
    # 传递 current_action 以支持 RAG 模式的相关性检索
    history_summary = build_history_summary(
        action_history, 
        max_history_tokens=max_history_tokens,
        max_history_days=max_history_days,
        reference_timestamp=reference_timestamp,
        current_action=test_action
    )
    
    # 构建待预测场景描述
    scenario_desc = build_test_action_description(test_action)
    
    # 获取问题相关信息
    field = question_info.get("field", "")
    
    # 根据字段类型定制连续值问题
    continuous_questions = {
        "video_watch_seconds": "你预计该用户会在这个视频上观看多少秒？",
        "live_watch_seconds": "你预计该用户会在这个直播间停留多少秒？",
        "ad_watch_seconds": "你预计该用户会在这条广告上停留多少秒？",
    }
    
    # 获取问题文本
    if field in continuous_questions:
        question_text = continuous_questions[field]
    else:
        # 未知字段返回通用问题
        question_text = f"你预计该用户会在这里停留多少秒？（{field}）"
    
    # 处理 user_profile
    if user_profile:
        first_period_idx = user_profile.find('。')
        if first_period_idx != -1:
            user_profile_short = user_profile[:first_period_idx]
        else:
            user_profile_short = user_profile
    else:
        user_profile_short = ""
    
    # 组装完整prompt
    prompt_parts = [
        "你是快手平台的一位真实用户。你的核心任务是：基于给定的历史行为序列，推断该用户的兴趣偏好、消费水平和性格特征，并据此模拟TA在当前某场景下的真实决策。",
        "## 核心原则",
        "1. **数据驱动**：所有推断必须基于历史行为数据中的客观证据，避免无根据的臆测和假设。",
        "2. **行为连贯性**：新决策应与用户的历史行为模式保持内在逻辑一致，体现其稳定的偏好和习惯。",
        "3. **个体差异性**：充分尊重每个用户的独特性，不套用刻板印象或群体标签，从数据中发现真实的个性特征。",
        "4. **情境敏感性**：决策预测需考虑当前场景的特殊性，平衡长期偏好与短期情境因素的影响。",
        "5. **真实性优先**：模拟真实用户可能做出的选择，包括不感兴趣、犹豫、跳过等消极行为，而非总是给出积极响应。",
        "## 输入一：用户画像",
        "这是该用户的平台基本信息，可以作为理解用户背景的参考：",
        user_profile_short,
        "## 输入二：历史行为轨迹信息",
        "这是该用户在过去一段时间内的真实操作记录（包含直播、商城、视频、广告等跨场景行为）。请仔细分析这些行为背后的动机和倾向，挖掘其隐含的长期偏好和短期意图：",
        history_summary,
        "## 输入三：当前测试场景",
        "用户现在遇到了以下场景：",
        scenario_desc,
        "## 预测任务",
        "请代入该用户视角，回答以下问题：",
        question_text,
        "## 输出要求",
        "**请只输出一个整数，不要输出任何其他内容、解释或单位。**",
        "你的回答：",
    ]
    
    prompt = "\n".join(prompt_parts)
    
    return {
        "prompt": prompt,
        "question_info": question_info,
        "test_action": test_action,
        **({"scenario_desc": scenario_desc} if "scenario_desc" in locals() else {}),
        **({"question_text": question_text} if "question_text" in locals() else {}),
    }


def build_single_text_prompt(
    user_profile: str,
    action_history: List[Dict],
    test_action: Dict,
    question_info: Dict,
    max_history_tokens: int = None,
    max_history_days: int = None,
) -> Dict:
    """
    构建单个文本预测问题的Prompt（如搜索关键词预测）
    
    Args:
        user_profile: 用户画像
        action_history: 历史行为列表
        test_action: 待预测的测试行为
        question_info: 单个问题的信息，包含 question, type, field, true_value 等
        max_history_tokens: 最大 token 数限制
        max_history_days: 只保留近 N 天的历史行为，如果为 None 则不限制天数
    
    Returns:
        {
            "prompt": str,  # 完整的prompt
            "question_info": Dict,  # 原始问题信息
            "test_action": Dict,  # 原始的待预测行为
        }
        如果测试行为应该被过滤，返回 None
    """
    # 检查测试行为是否应该被过滤
    if should_filter_action(test_action):
        return None
    
    # 获取测试行为的时间戳作为参考时间点
    reference_timestamp = test_action.get("timestamp")
    
    # 构建历史行为摘要
    # 传递 current_action 以支持 RAG 模式的相关性检索
    history_summary = build_history_summary(
        action_history, 
        max_history_tokens=max_history_tokens,
        max_history_days=max_history_days,
        reference_timestamp=reference_timestamp,
        current_action=test_action
    )
    
    # 获取问题文本
    question_text = question_info.get("question", "")
    field = question_info.get("field", "")
    
    
    # 处理 user_profile
    if user_profile:
        first_period_idx = user_profile.find('。')
        if first_period_idx != -1:
            user_profile_short = user_profile[:first_period_idx]
        else:
            user_profile_short = user_profile
    else:
        user_profile_short = ""
    
    # 根据 field 决定是否需要场景描述
    # 对于搜索关键词预测，不需要描述当前场景（因为就是要预测用户想搜索什么）
    if field == "search_keyword":
        # 搜索关键词预测：只基于历史行为，不暴露当前场景
        prompt_parts = [
            "你是快手平台的一位真实用户。你的核心任务是：基于给定的历史行为序列，推断该用户的兴趣偏好、消费习惯和当前可能的需求，并预测TA接下来想要搜索的内容。",
            "## 核心原则",
            "1. **数据驱动**：所有推断必须基于历史行为数据中的客观证据，避免无根据的臆测和假设。",
            "2. **行为连贯性**：预测的搜索内容应与用户的历史行为模式保持内在逻辑一致，体现其稳定的偏好和当前意图。",
            "3. **个体差异性**：充分尊重每个用户的独特性，不套用刻板印象或群体标签，从数据中发现真实的个性特征。",
            "4. **情境敏感性**：结合用户近期的行为趋势，推断其当前可能的需求或好奇心。",
            "5. **真实性优先**：预测真实用户可能搜索的内容，包括日常需求、兴趣探索、购物需求等。",
            "## 输入一：用户画像",
            "这是该用户的平台基本信息，可以作为理解用户背景的参考：",
            user_profile_short,
            "## 输入二：历史行为轨迹信息",
            "这是该用户在过去一段时间内的真实操作记录（包含直播、商城、视频、广告、搜索等跨场景行为）。请仔细分析这些行为背后的动机和倾向，挖掘其隐含的长期偏好和短期意图：",
            history_summary,
            "## 预测任务",
            "请代入该用户视角，预测TA现在打开搜索框后会输入什么关键词。",
            "## 输出要求",
            "**请只输出搜索关键词内容本身，不要输出任何其他内容、解释、引号或分析。**",
            "你的回答：",
        ]
    elif field == "next_user_message":
        # 电商客服对话预测：对话历史已在 question_text 中构建好
        # 判断是否有对话历史
        has_dialogue_history = question_info.get("has_dialogue_history", True)
        
        prompt_parts = [
            "你是一位真实的电商平台用户。你的核心任务是：基于给定的历史行为序列，推断该用户的沟通风格、性格特征和当前需求，并据此模拟TA在客服对话中的真实表达。",
            "## 核心原则",
            "1. **数据驱动**：所有推断必须基于历史行为数据中的客观证据，避免无根据的臆测和假设。",
            "2. **风格连贯性**：预测的表达应与用户在历史中展现的沟通风格、语气和用词习惯保持一致。",
            "3. **个体差异性**：充分尊重每个用户的独特性，不套用刻板印象或群体标签。",
            "4. **情境敏感性**：结合当前对话的上下文、用户遇到的问题和情绪状态进行预测。",
            "5. **真实性优先**：模拟真实用户可能说出的话，体现其独特的沟通风格和当前情绪。",
            "## 输入一：用户画像",
            "这是该用户的平台基本信息，可以作为理解用户背景的参考：",
            user_profile_short,
            "## 输入二：历史行为轨迹信息",
            "这是该用户在过去一段时间内的真实操作记录（包含直播、商城、视频、广告等跨场景行为）。请仔细分析这些行为背后的动机和倾向，挖掘其沟通风格和性格特征：",
            history_summary,
            "## 输入三：当前客服对话场景",
            question_text,
            "## 输出要求",
            "**请只输出用户会说的话，不要输出任何其他内容、解释、引号或分析。直接输出对话内容即可。**",
            "你的回答：",
        ]
    else:
        # 其他文本预测：需要场景描述
        scenario_desc = build_test_action_description(test_action)
        
        prompt_parts = [
            "你是快手平台的一位真实用户。你的核心任务是：基于给定的历史行为序列，推断该用户的兴趣偏好、沟通风格和性格特征，并据此模拟TA在当前场景下的真实表达。",
            "## 核心原则",
            "1. **数据驱动**：所有推断必须基于历史行为数据中的客观证据，避免无根据的臆测和假设。",
            "2. **行为连贯性**：预测的文本应与用户的历史行为模式和表达风格保持内在逻辑一致。",
            "3. **个体差异性**：充分尊重每个用户的独特性，不套用刻板印象或群体标签。",
            "4. **情境敏感性**：预测需考虑当前场景的特殊性，平衡长期风格与短期情境因素的影响。",
            "5. **真实性优先**：模拟真实用户可能说出的话，体现其独特的沟通风格。",
            "## 输入一：用户画像",
            "这是该用户的平台基本信息，可以作为理解用户背景的参考：",
            user_profile_short,
            "## 输入二：历史行为轨迹信息",
            "这是该用户在过去一段时间内的真实操作记录：",
            history_summary,
            "## 输入三：当前测试场景",
            "用户现在遇到了以下场景：",
            scenario_desc,
            "## 预测任务",
            "请代入该用户视角，回答以下问题：",
            question_text,
            "## 输出要求",
            "**请只输出预测的文本内容本身，不要输出任何其他内容、解释、引号或分析。**",
            "你的回答：",
        ]
    
    prompt = "\n".join(prompt_parts)
    
    return {
        "prompt": prompt,
        "question_info": question_info,
        "test_action": test_action,
        "field": field,
        **({"scenario_desc": scenario_desc} if "scenario_desc" in locals() else {}),
        **({"question_text": question_text} if "question_text" in locals() else {}),
    }


def _get_yes_no_question_by_field(question_info: Dict) -> str:
    """
    根据 field 获取 Yes/No 问题
    
    Args:
        question_info: 问题信息，包含 type, field 等
    
    Returns:
        Yes/No 问题文本
    """
    field = question_info.get("field", "")
    
    # 根据字段类型定制 Yes/No 问题
    yes_no_questions = {
        # 视频浏览场景
        "video_completed": "基于这个用户的观看习惯，该用户会把这个视频完整看完吗？",
        "video_liked": "结合这个用户的互动习惯和对视频内容的喜爱程度，该用户会为这个视频点赞吗？",
        "video_commented": "考虑到这个用户的表达欲望和参与度，该用户会在这个视频下发表评论吗？",
        "video_shared": "基于这个用户的分享习惯和社交行为，该用户会把这个视频分享给朋友吗？",
        "video_collected": "根据这个用户的收藏偏好，该用户会收藏这个视频吗？",
        "video_followed": "考虑到这个用户的关注习惯，该用户会关注这个视频的作者吗？",
        
        # 商城购物场景
        "shop_added_to_cart": "根据这个用户的购物习惯，该用户会把这件商品加入购物车吗？",
        "shop_order_success": "结合这个用户的购物偏好和消费能力，该用户会购买这件商品吗？",
        
        # 广告推荐场景
        "ad_liked": "考虑到这个用户对广告内容的喜爱程度，该用户会为这条广告点赞吗？",
        "ad_commented": "基于这个用户的表达欲望，该用户会在这条广告下发表评论吗？",
        "ad_activated": "假设用户点击了这条广告，基于TA的行为特征，该用户会激活或注册广告中的应用/服务吗？",
        "ad_form_submitted": "基于这个用户的行为特征和对广告的兴趣程度，该用户会填写并提交广告中的表单吗？",
        
        # 直播间场景
        "live_liked": "结合这个用户的互动习惯，该用户会为主播点赞吗？",
        "live_commented": "考虑到这个用户的表达欲望，该用户会在直播间发送弹幕或评论吗？",
        "live_sent_gift": "基于这个用户的消费能力和打赏习惯，该用户会在直播间给主播送礼物吗？",
        "live_followed": "根据这个用户的关注习惯，该用户会关注这个主播吗？",
        "live_shared": "基于这个用户的分享习惯，该用户会把这个直播间分享给朋友吗？",
        "live_clicked_cart": "根据该用户的购物习惯，该用户会把直播间的商品加入购物车吗？",
        
        # 搜索行为场景
        "search_clicked": "当用户搜索这个关键词后，该用户会点击搜索结果吗？",
    }
    
    # 返回预定义的问题
    if field in yes_no_questions:
        return yes_no_questions[field]
    
    # 未知字段返回通用问题
    return f"该用户会执行此操作吗？（{field}）"


def get_binary_questions_for_action(action: Dict) -> List[Dict]:
    """
    获取一个行为中所有的二分类问题
    
    Args:
        action: 行为记录
    
    Returns:
        List of 二分类问题信息
    """
    questions = build_prediction_questions(action)
    # 只返回二分类问题
    return [q for q in questions if q.get("type") == "binary"]


def get_all_questions_for_action(action: Dict) -> List[Dict]:
    """
    获取一个行为中所有的预测问题（包括二分类、连续值、文本）
    
    Args:
        action: 行为记录
    
    Returns:
        List of 所有问题信息（binary + continuous + text）
    """
    return build_prediction_questions(action)
