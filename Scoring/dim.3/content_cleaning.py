# -*- coding: utf-8 -*-

"""
混合语境文本处理模块 (XHS Mixed Language Processor)

- 专为处理小红书平台上的中英混合文本而设计的class, XHSMixedLanguageProcessor，其流程包括:               
    # 1. 基础清洗(去除HTML标签、URL、特殊符号等, 全角转半角, 空格标准化)
    text = self.basic_clean(text)

    # 2. 社交媒体特定清洗 (去除话题标签、@提及、Emoji的demojize, 处理小红书特定标签和表情包)
    text = self.social_media_clean(text)

    # 3. 中英语境优化（用langid做语言检测, 对批量处理超过100条且文本中文含量低于50%时用deep_translator API翻译）
    if enable_translation and is_batch_large:
        text = self.language_optimize(text, protected_terms=self.protected_terms)

    # 4. 语义清洗 (jieba中文分词, 根据TF-IDF去除噪声词, 并保护领域关键词）
    text = self.chinese_semantic_clean([text])[0]

- 使用方法:
    - 单条文本处理: Class.process_text(text)
    - 批量文本处理: Class.batch_process([text1, text2, ...]) # 文本数量>100提供翻译选项
    - 查看处理统计信息: Class.get_stats()

- 处理的亮点：
    - 预编译正则表达式以提高效率
    - 批量处理待翻译内容, 并添加缓存以减少API调用
    - 使用ThreadPoolExecutor并行执行I/O密集型任务来提高批量处理性能

XHS Mixed Language Processor

- Overview
A specialized class designed for processing mixed Chinese-English text from Xiaohongshu (XHS). 

- Workflows:
    - Basic Cleaning: Remove HTML tags, URLs, and special symbols; convert full-width to half-width; normalize spaces.
    - Social Media Cleaning: Remove hashtags, @mentions, demojize emojis, and handle XHS-specific tags.
    - Language Optimization (using langid for language detection and deep_translator API for sentences with batch size > 100 and Chinese content < 50%):
    - Semantic Cleaning: Use jieba for Chinese tokenization, remove noise words based on TF-IDF, and preserve domain-specific keywords.

- Usage
    - Single text processing: Class.process_text(text)
    - Batch processing: Class.batch_process([text1, text2, ...]) # translation option for >100 texts
    - View processing stats: Class.get_stats()

- Highlights
    - Precompiled regex for efficiency
    - Batch translation requests with caches to reduce API calls
    - Parallel I/O tasks with ThreadPoolExecutor for better batch performance
"""

import os
import logging
import time
import re
import emoji
from typing import Literal, List, Set, Dict, Union, Optional, Tuple
import json
from langid import classify
import jieba 
from deep_translator import GoogleTranslator
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MixedLanguageProcessor")
jieba_logger = logging.getLogger('jieba')
jieba_logger.setLevel(logging.INFO)  # Suppress DEBUG messages

# Precompile regex patterns for efficiency
ZH_CHAR_PATTERN = re.compile(r'[\u4e00-\u9fff]')  # 中文字符检测
NON_ZH_PATTERN = re.compile(r'[a-zA-Z]')  # 拉丁字母检测
SPLIT_PATTERN = re.compile(r'([。！？?!.])')  # 分句正则
URL_PATTERN = re.compile(
    r'(?:https?://)?(?:[a-zA-Z0-9\u4e00-\u9fff-]+\.)+[a-zA-Z]{2,}'
    r'(?:/\S*)?(?:\?\S*)?(?:#\S*)?',
    flags=re.IGNORECASE
)
TOPIC_PATTERN = re.compile(r'#([^#]+)#')
MENTION_PATTERN = re.compile(r'@[\w\u4e00-\u9fff-]+')  # 支持中英文用户名
XIAOHONGSHU_TAG_PATTERN = re.compile(r'\[(话题|表情|地点)\s*[^\]]*\]')
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'[\t\n\u3000]')
MULTI_SPACE_PATTERN = re.compile(r'\s+')

# Define the main class for mixed language processing
class XHSMixedLanguageProcessor:
    """混合语境文本处理器，专为处理中英混合的社交媒体内容设计
    单条文本处理不提供翻译功能，批量处理超过100条时才启用翻译"""
    
    def __init__(self, cache_size=1000, max_workers=12):
        """        
        Args:
            cache_size: 翻译缓存的最大条目数
            max_workers: 并行处理的最大工作线程数
        """
        self.max_workers = max_workers
        self._translator = GoogleTranslator(source='auto', target='zh-CN')
        self.translation_cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.translation_requests = 0

        # Translation management metrics
        self._success_streak = 0
        self._failure_streak = 0
        self._last_error_time = 0
        
        # protected_terms and domain_keywords
        self.domain_keywords = {
        '鲜芋仙', 'Meet Fresh', 'MeetFresh', '台湾美食', '甜品', 
        '芋圆', 'taro ball', '芋头', 'taro', '仙草', 'grass jelly', '奶茶', 'milk tea',
        '豆花', 'tofu pudding', '牛奶冰', 'milked shaved ice', '奶刨冰',
        '红豆汤', 'purple rice soup', '紫米粥', 'red bean soup',
        '2001 Coit Rd', 'Park Pavillion Center', '(972) 596-6088',
        '餐厅', '餐馆', '美食', '台湾小吃', '台湾甜品', '冰激凌',
        "VIP", "AI", "DFW", "Dallas", "达拉斯", '卫生', '整洁', '干净', '服务', '态度', 
        '好评', '推荐', '分享', '体验', '美味', '好吃', '好喝', '新鲜', '正宗',
        '特色', '口味', '菜品', '环境', '氛围', '价格', '实惠', '划算', '优惠', '折扣', 
        '活动', '套餐', '分量', '足量', '满意', '点赞', '刘一手', '小龙虾', '火锅',
        '烧烤', '串串香', '麻辣烫', '小吃', '炸鸡', '汉堡', '披萨', '寿司', '刺身',
        '奶昔', '海鲜', '鱼', '肉', '蔬菜', '锅底', '蘸料', '调料', '酱汁', '配菜',
        }
        self.protected_terms = {
        'Meet Fresh', 'MeetFresh', 'taro ball', 'taro', 'grass jelly', 'milk tea',
        'tofu pudding', 'milked shaved ice', 'purple rice soup', 'red bean soup',
        '2001 Coit Rd', 'Park Pavillion Center', "VIP", "AI", "DFW", "Dallas",
        }

    def register_domain_keywords_with_jieba(self, keywords):
        """Register domain keywords with jieba to improve segmentation"""
        for keyword in keywords:
            if isinstance(keyword, str) and len(keyword) >= 2:
                # The higher frequency (third parameter) ensures jieba treats this as a single word
                jieba.add_word(keyword, freq=10000)
        
    def fullwidth_to_halfwidth(self, text: str) -> str:
        """全角转半角（保留￥符号）"""
        translation_table = str.maketrans({
        '！': '!', '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
        '、': ',', '，': ',', '；': ';', '：': ':', '？': '?',
        '《': '<', '》': '>', '【': '[', '】': ']', '·': '.',
        '～': '~', '—': '-', '（': '(', '）': ')', '　': ' '
        })
        return text.translate(translation_table)

    def normalize_punctuation(self, text: str) -> str:
        """符号标准化（保留emoji和重要符号）"""
        # 定义保留符号集
        keep_symbols = {"'", '"', ',', '.', ':', ';', '!', '?', '-', 
                        '(', ')', '<', '>', '[', ']', '&', '#', '@',
                        '%', '$', '￥', '/', '=', '+', '~', '^'}
        
        # 字符级处理 
        cleaned_chars = []
        for char in text:
            # 保留条件：字母数字/汉字/keep_symbols/emoji
            if (char.isalnum() or
                '\u4e00' <= char <= '\u9fff' or
                char in keep_symbols or
                emoji.is_emoji(char)):
                cleaned_chars.append(char)
            else:
                cleaned_chars.append(' ')
        
        return ''.join(cleaned_chars)

    def remove_urls(self, text: str) -> str:
        """适配中文域名的URL移除"""
        return URL_PATTERN.sub('', text)

    def basic_clean(self, text: str) -> str:
        """
        对文本进行基础层清洗，包括去除HTML标签、URL、特殊符号处理, 空格标准化等。
        """
        if not text:
            return ""
            
        # 替换所有空白符（含小红书常见的全角空格\u3000、换行、制表符）
        text = WHITESPACE_PATTERN.sub(' ', text)
        # HTML标签移除
        text = HTML_TAG_PATTERN.sub('', text)
        # URL移除, 适配中文域名
        text = self.remove_urls(text)
        # 全角转半角（保留全角￥）
        text = self.fullwidth_to_halfwidth(text)
        # 符号标准化
        text = self.normalize_punctuation(text)
        # 标准化合并连续空格
        text = MULTI_SPACE_PATTERN.sub(' ', text).strip()
        return text

    def social_media_clean(self, text: str, strategy='demojize') -> str:
        """
        社交媒体文本清洗，主要针对小红书平台的特定格式和符号进行处理。
        """
        if not text:
            return ""
            
        # 移除话题标签但保留关键词（如 #达拉斯美食# → 达拉斯美食）
        text = TOPIC_PATTERN.sub(r'\1', text)
        
        # 移除@提及(包含其变体, 如@小红书用户)
        text = MENTION_PATTERN.sub('', text)
        
        # 转换Emoji（可选策略）
        if strategy == 'remove':
            text = emoji.replace_emoji(text, replace='')
        elif strategy == 'demojize':
            # 将emoji转换为文本描述（如:😀 → :grinning_face:）
            text = emoji.demojize(text, delimiters=('__EMOJI_', '__'))
        
        # 处理小红书特有的方括号标签（删除系统标签, 如[话题]→'')
        text = XIAOHONGSHU_TAG_PATTERN.sub('', text)

        # 移除小红书表情包标签，如[笑哭R]→''
        text = re.sub(r'\[[^\]]*R\]', '', text)
        
        return text.strip()

    def ratio_of_chinese(self, text: str) -> float:
        """返回 text 中（Unicode范围4E00-9FFF）汉字的占比。"""
        if not text:
            return 1.0
        zh_chars = ZH_CHAR_PATTERN.findall(text)
        return len(zh_chars) / len(text) if len(text) > 0 else 0.0

    def mask_protected_terms(self, text: str, protected_terms: set) -> (str, dict):
        """
        将 text 中出现的保护词用占位符替换，并返回替换后的文本以及占位符映射字典。
        使用更难被翻译API修改的占位符格式: __TERM_0__
        """
        if not protected_terms:
            return text, {}
        
        # 按术语长度降序排列（优先匹配长词）
        sorted_terms = sorted(protected_terms, key=lambda x: len(str(x)), reverse=True)
        
        # 不使用word boundary，而是直接匹配整个词，提高匹配精度
        pattern = re.compile(
            '(' + '|'.join(map(re.escape, sorted_terms)) + ')', 
            flags=re.IGNORECASE
        )
        
        placeholder_map = {}
        idx = 0
        
        def _repl(m):
            nonlocal idx
            # 使用更稳定的占位符格式
            placeholder = f"__TERM_{idx}__"
            placeholder_map[placeholder] = m.group(0)
            idx += 1
            return placeholder
        
        return pattern.sub(_repl, text), placeholder_map

    def unmask_protected_terms(self, text: str, placeholder_map: dict) -> str:
        """将翻译后文本中的占位符恢复成原始英文术语，支持不区分大小写"""
        if not placeholder_map or not text:
            return text
            
        result = text
        for placeholder, original in placeholder_map.items():
            # 创建不区分大小写的正则表达式模式
            pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
            # 替换所有匹配
            result = pattern.sub(original, result)
        return result
        
    def _update_cache(self, key: str, value: str) -> None:
        """将翻译结果添加到缓存，管理缓存大小"""
        # Skip caching for very long texts to save memory
        if len(key) > 1000:
            return
            
        # If cache is full, remove least recently used items (20% of capacity)
        if len(self.translation_cache) >= self.cache_size:
            # Remove oldest 20% of entries
            items_to_remove = max(1, int(self.cache_size * 0.2))
            for _ in range(items_to_remove):
                if self.translation_cache:
                    self.translation_cache.pop(next(iter(self.translation_cache)))
                    
        # Store in cache
        self.translation_cache[key] = value
        
    def batch_translate(self, texts: List[str]) -> List[str]:
        """批量翻译函数，利用缓存并使用令牌桶限流和智能重试策略, 仅在处理100条以上文本时使用"""
        if not texts:
            return []
        
        # Filter out empty texts or numeric-only texts
        to_translate = []
        indices = []
        results = [""] * len(texts)
        
        # First check cache for all texts
        for i, text in enumerate(texts):
            if not text or text.strip().isdigit():
                results[i] = text
            elif text in self.translation_cache:
                # Cache hit
                results[i] = self.translation_cache[text]
                self.cache_hits += 1
            else:
                # Cache miss, need to translate
                if len(text) > 5000:
                    text = text[:4900]
                to_translate.append(text)
                indices.append(i)
        
        if not to_translate:
            return results
        
        # Remove duplicates to minimize API calls (preserving original indices)
        unique_texts = []
        text_to_indices = {}
        for idx, text in zip(indices, to_translate):
            if text not in text_to_indices:
                text_to_indices[text] = [idx]
                unique_texts.append(text)
            else:
                text_to_indices[text].append(idx)
        
        # Translation parameters based on past performance
        success_streak = self._success_streak
        failure_streak = self._failure_streak
        last_error_time = self._last_error_time
        current_time = time.time()
        
        # Adaptive batch size - decrease after failures, increase after successes
        base_batch_size = 5
        if failure_streak > 0:
            # Reduce batch size after failures
            batch_size = max(1, min(3, base_batch_size - failure_streak))
            
            # Apply longer cooldown after frequent failures
            if current_time - last_error_time < 30 and failure_streak > 2:
                logger.info(f"Cooling down translation API for 5 seconds after {failure_streak} failures")
                time.sleep(5)
        else:
            # Gradually increase batch size after successful operations
            batch_size = min(8, base_batch_size + min(3, success_streak // 10))
        
        # Process unique texts in batches
        unique_results = {}
        total_unique = len(unique_texts)
        processed = 0
        
        for i in range(0, total_unique, batch_size):
            batch = unique_texts[i:i+batch_size]
            remaining = total_unique - processed
            logger.debug(f"Translating batch {i//batch_size + 1}, size={len(batch)}, remaining={remaining}")
            
            # Exponential backoff for retries
            max_retries = 3
            retry_count = 0
            success = False
            
            while not success and retry_count <= max_retries:
                try:
                    # Apply rate limiting with adaptive delay
                    delay = 0.1 * (2 ** retry_count) if retry_count > 0 else 0
                    if delay > 0:
                        time.sleep(delay)
                    
                    # Perform translation
                    self.translation_requests += len(batch)
                    translated = self._translator.translate_batch(batch)
                    
                    # Update results and cache
                    for txt, trans in zip(batch, translated):
                        unique_results[txt] = trans
                        self._update_cache(txt, trans)
                    
                    # Update success metrics
                    self._success_streak = success_streak + 1
                    self._failure_streak = 0
                    success = True
                    processed += len(batch)
                    
                except Exception as e:
                    retry_count += 1
                    self._last_error_time = time.time()
                    
                    if "too many requests" in str(e).lower():
                        # Rate limit hit - apply exponential backoff
                        self._failure_streak = failure_streak + 1
                        self._success_streak = 0
                        wait_time = min(30, 1 * (2 ** retry_count))  # Cap at 30 seconds
                        logger.warning(f"Hit rate limit, retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Translation error: {e}, retrying in {delay}s (attempt {retry_count}/{max_retries})")
                        if retry_count >= max_retries:
                            # Fall back to original texts after max retries
                            for txt in batch:
                                unique_results[txt] = txt
                            processed += len(batch)
            
            # Add small delay between batches to avoid rate limits
            if i + batch_size < total_unique:
                time.sleep(0.2)
        
        # Map unique results back to all requested texts
        for text, text_indices in text_to_indices.items():
            for idx in text_indices:
                results[idx] = unique_results.get(text, text)
        
        return results

    def language_optimize(
        self,
        text: str,
        threshold: float = 0.5,
        protected_terms: set = None,
        enable_translation: bool = True
    ) -> str:
        """
        负责处于双语环境的文本优化, 统一的语言优化方法，支持单文本和批处理模式
        修改为先检测整个note的中文含量，只有当整体中文含量低于阈值时才进行翻译处理
        1. 快速分句处理
        2. 语言检测过滤需要翻译的句子
        3. 合并重组处理结果
        """
        if not enable_translation or not text:
            return text

        if protected_terms is None:
            protected_terms = self.protected_terms

        # Skip if no non-Chinese characters
        if not NON_ZH_PATTERN.search(text):
            return text
        
        # Split into segments more efficiently
        segments = SPLIT_PATTERN.split(text)
        
        # Process sentences and separators in one pass
        sentences = segments[::2]  # Even indices are sentences
        separators = segments[1::2] if len(segments) > 1 else []  # Odd indices are separators
        
        # Single collection for all translation tasks
        to_translate = []
        indices = []
        ph_maps = []
        result_sentences = [""] * len(sentences)
        
        # Identify sentences that need translation
        for i, sentence in enumerate(sentences):
            if not sentence:
                result_sentences[i] = sentence
                continue
                
            if NON_ZH_PATTERN.search(sentence):
                masked, ph_map = self.mask_protected_terms(sentence, protected_terms)
                
                if self.ratio_of_chinese(masked) < threshold:
                    to_translate.append(masked)
                    indices.append(i)
                    ph_maps.append(ph_map)
                else:
                    result_sentences[i] = sentence
            else:
                result_sentences[i] = sentence
        
        # Only call batch_translate once with all sentences
        if to_translate:
            translated_batch = self.batch_translate(to_translate)
            
            # Process results back into original positions
            for j, (idx, translated, ph_map) in enumerate(zip(indices, translated_batch, ph_maps)):
                result_sentences[idx] = self.unmask_protected_terms(translated, ph_map)
        
        # Efficiently reassemble text
        result = []
        for i, sentence in enumerate(result_sentences):
            result.append(sentence)
            if i < len(separators):
                result.append(separators[i])
        
        return ''.join(result)

    def chinese_semantic_clean(
        self,
        texts: List[str],
        freq_threshold: float = 0.5,          # 词频阈值
        doc_freq_threshold: float = 0.8,       # 文档频率阈值
        min_word_length: int = 2,              # 最小词长度（过滤单字词）
        custom_stopwords: Optional[Set[str]] = None,
        domain_keywords: Optional[Set[str]] = None,
        return_noise_terms: bool = False,       # 是否返回识别出的噪声词
        verbose: bool = True
    ) -> Union[List[str], tuple]:
        """
        基于词频统计的中文语义清洗文本中的噪声词。
        通过并行处理加速清洗过程，支持自定义停用词和领域关键词。
        """
        if not texts:
            return [] if not return_noise_terms else ([], set())
            
        # Merge default and custom stopwords more efficiently
        stopwords = {
            # 情感强化词
            "真的", "真是", "太", "好", "很", "非常", "超级", "绝对", "简直",
            # 网络用语
            "哈哈", "哈哈哈", "啊啊", "啊啊啊", "呜呜", "呜呜呜", "omg", "OMG",
            "xswl", "awsl", "yyds", "绝绝子", "无语子",
            # 口头禅
            "真的是", "就是", "反正", "然后", "其实", "那个", "这个", "所以",
            "emmm", "emm", "啊这", "蹲一个", "冲鸭",
            # 标点符号组合
            "～～", "…"
        }
        
        if custom_stopwords:
            stopwords.update(custom_stopwords)
        
        # Initialize word statistics
        domain_keywords = domain_keywords or set()
        total_docs = len(texts)
        word_counts = Counter()
        doc_counts = defaultdict(int)
        
        if verbose and total_docs > 1:
            print(f"正在统计词频（共{total_docs}篇文本）...")
        
        # Process all texts in one loop
        for text in texts:
            # Cut words and filter by length in one operation
            words = [w for w in jieba.lcut(text) if len(w) >= min_word_length]
            
            # Update global word frequency
            word_counts.update(words)
            
            # Update document frequency with unique words
            unique_words = set(words)
            for word in unique_words:
                doc_counts[word] += 1
        
        # Calculate relative frequencies
        total_words = sum(word_counts.values()) or 1  # Avoid division by zero
        word_freq = {word: count/total_words for word, count in word_counts.items()}
        doc_freq = {word: count/total_docs for word, count in doc_counts.items()}
        
        # Adjust thresholds for small datasets more concisely
        if total_docs < 100:
            freq_threshold = min(0.8, freq_threshold * 2)
            doc_freq_threshold = min(0.9, doc_freq_threshold * 1.5)
        
        # Identify noise terms more efficiently
        noise_terms = {
            word for word, freq in word_freq.items()
            if ((freq > freq_threshold or doc_freq[word] > doc_freq_threshold) and 
                word not in domain_keywords and len(word) <= 2)
        }
        
        # Add longer stopwords
        noise_terms.update(w for w in stopwords if len(w) >= min_word_length)
        
        # Build pattern and clean texts
        if noise_terms:
            pattern = re.compile('|'.join(map(re.escape, noise_terms)))
            cleaned_texts = [pattern.sub('', text) for text in texts]
        else:
            cleaned_texts = texts[:]
        
        # Report processing stats if necessary
        if verbose and total_docs > 1:
            total_orig_len = sum(len(t) for t in texts)
            total_cleaned_len = sum(len(t) for t in cleaned_texts)
            
            if total_orig_len > 0:
                ratio = total_cleaned_len / total_orig_len
                print(f"已识别噪声词{len(noise_terms)}个，清洗完成!")
                print(f"噪声去除率: {(1-ratio):.2%}")
                print(f"原始/清洗后字符数: {total_orig_len}/{total_cleaned_len}")
        
        return (cleaned_texts, noise_terms) if return_noise_terms else cleaned_texts
        
    def process_text(self, text: str) -> str:
        """完整的中英双语单文本处理流程, 不提供翻译功能
        Args:
            text: 待处理文本
        Returns:
            str: 处理后的文本
        处理流程:
            1. 基础清洗(去除HTML标签、URL、特殊符号等, 全角转半角, 空格标准化)
            2. 社交媒体特定清洗 (去除话题标签、@提及、表情包的demojize, 处理小红书特定标签)
            3. 语义清洗 (jieba中文分词, 根据TF-IDF去除噪声词, 并保护领域关键词）             
        """
        # Early return for empty text
        if not text or text.strip() == '':
            return ""
        
        # Register domain keywords with jieba
        self.register_domain_keywords_with_jieba(self.domain_keywords)

        # 1. 基础清洗
        text = self.basic_clean(text)

        # 2. 社交媒体特定清洗
        text = self.social_media_clean(text)

        # 3. 语义清洗 (单条处理不提供翻译功能)
        text = self.chinese_semantic_clean([text], verbose=False)[0]

        return text
        
    def batch_process(self, texts, enable_translation=True, verbose=False) -> List[str]:
        """
        完整的中英双语多文本处理流程, 翻译默认开启, CPU和chunk_size自动调整
        同时, 如果文本数量小于100, 则不使用并行处理, 并关闭翻译
        如果文本数量大于100, 使用并行处理的同时, 批量翻译以优化deep_translator API调用

        Args:
            texts: 待处理文本列表
            enable_translation: 是否启用翻译
            verbose: 是否打印处理信息

        Returns:
            List[str]: 处理后的文本列表
        
        处理流程:
            1. 基础清洗(去除HTML标签、URL、特殊符号等, 全角转半角, 空格标准化)
            2. 社交媒体特定清洗 (去除话题标签、@提及、表情包的demojize, 处理小红书特定标签)
            3. 中英语境优化（用langid做语言检测, 并在句子中文含量低于60%时用deep_translator API翻译）
            4. 语义清洗 (jieba中文分词, 根据TF-IDF去除噪声词, 并保护领域关键词）
        """
        if not texts:
            return []
    
        # 确定是否执行翻译 - 只有当enable_translation=True且文本数量>100时才启用
        actual_enable_translation = enable_translation and len(texts) > 100
        
        # Register domain keywords once for all processing
        self.register_domain_keywords_with_jieba(self.domain_keywords)
        
        # For small datasets, process sequentially without translation
        if len(texts) <= 100:
            if verbose:
                print(f"Processing {len(texts)} texts sequentially (translation disabled)")
            return [self.process_text(text) for text in texts]
        
        # For larger datasets, use parallelization
        effective_cores = min(self.max_workers, os.cpu_count() or 4)
        chunk_size = max(10, len(texts) // effective_cores)
        
        if verbose:
            print(f"Processing {len(texts)} texts with {effective_cores} cores")
            print(f"Translation {'enabled' if actual_enable_translation else 'disabled'}")
        
        # Define a more efficient chunk processing function
        def process_chunk(chunk):
            # Create a pipeline for each text in the chunk
            processed_texts = []
            
            for text in chunk:
                if not text or not text.strip():
                    processed_texts.append("")
                    continue
                
                # Phase 1: Basic cleaning
                cleaned = self.basic_clean(text)
                
                # Phase 2: Social media cleaning
                cleaned = self.social_media_clean(cleaned)
                
                # Phase 3: Language optimization (only with translation enabled)
                if actual_enable_translation and NON_ZH_PATTERN.search(cleaned):
                    cleaned = self.language_optimize(
                        cleaned,
                        protected_terms=self.domain_keywords,
                        enable_translation=True
                    )
                
                processed_texts.append(cleaned)
            
            # Phase 4: Semantic cleaning (done as batch)
            return self.chinese_semantic_clean(processed_texts, verbose=False)
        
        # Process chunks in parallel
        chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=effective_cores) as executor:
            chunk_results = list(executor.map(process_chunk, chunks))
        
        # Flatten results
        return [text for chunk in chunk_results for text in chunk]


    def get_stats(self) -> dict:
        """返回处理统计信息"""
        return {
            "cache_size": len(self.translation_cache),
            "cache_capacity": self.cache_size,
            "cache_hits": self.cache_hits,
            "translation_requests": self.translation_requests,
            "cache_hit_rate": self.cache_hits / max(1, self.translation_requests)
        }

# # Define functions for text processing used in the class
# DOMAIN_KEYWORDS = {'鲜芋仙', 'Meet Fresh', 'MeetFresh', '台湾美食', '甜品', 
#         '芋圆', 'taro ball', '芋头', 'taro', '仙草', 'grass jelly', '奶茶', 'milk tea',
#         '豆花', 'tofu pudding', '奶刨冰', 'milked shaved ice',
#         '红豆汤', 'purple rice soup', '紫米粥', 'red bean soup',
#         '2001 Coit Rd', 'Park Pavillion Center', '(972) 596-6088',
#         '餐厅', '餐馆', '美食', '台湾小吃', '台湾甜品', '冰激凌',
#         "VIP", "AI", "DFW", "Dallas", "达拉斯"}