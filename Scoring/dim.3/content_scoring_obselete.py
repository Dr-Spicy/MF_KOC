import pandas as pd
import numpy as np
import re
import os
import time
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 强制设置环境变量，禁用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免tokenizer警告

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('content_scoring')

# 抑制特定警告
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")

# 定义核心关键词
CORE_KEYWORDS = {
    '鲜芋仙', 'Meet Fresh', 'MeetFresh', '台湾美食', '甜品',
    '芋圆', 'taro', '仙草', 'grass jelly', '奶茶', 'milk tea',
    '豆花', 'tofu pudding', '奶刨冰', 'milked shaved ice', 
    '红豆汤', 'purple rice soup', '紫米粥', 'red bean soup',
    '2001 Coit Rd', 'Park Pavillion Center', '(972) 596-6088',
    '刘一手', '锅底', '火锅', '牛奶冰'
}

VERTICAL_DOMAIN_TAGS = {
    '美食', '探店', '火锅', '甜品', '奶茶', 
    '外卖', '中餐', '小吃', 'food'
}

# 最大CPU核心数
MAX_WORKERS = 6  

# 创建一个全局共享的SentenceTransformer实例
global_simcse_model = None

# 实用工具函数
def get_device():
    """获取计算设备 - 强制使用CPU"""
    return torch.device("cpu")

def load_models():
    """加载所需的各种模型 - 强制使用CPU"""
    global global_simcse_model
    
    models = {}
    try:
        # 原创性检测模型
        models['tfidf'] = TfidfVectorizer(max_features=8000)
        
        # 使用全局变量以避免重复加载模型
        if global_simcse_model is None:
            global_simcse_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
        models['simcse'] = global_simcse_model
        
        # 情感分析模型 - 指定使用CPU
        device = get_device()
        models['sentiment_tokenizer'] = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")
        models['sentiment_model'] = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese").to_empty(device=device).to(device)
        models['sentiment_model'].eval()
        
        return models
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        raise

def preprocess_data(df):
    """数据预处理，创建用户级评分DataFrame"""
    try:
        df = df.copy()
        
        # 用户笔记映射
        user_notes = df.groupby('user_id')['semantic_proc_text'].apply(list)
        
        # 创建包含所有用户ID的DataFrame（dim3_score）
        unique_users = df['user_id'].unique()
        dim3_score = pd.DataFrame({
            'user_id': unique_users, 
            'score3a_originality': 0, 
            'score3b_vertical': 0, 
            'score3c_sentiment': 0,
            'score3d_keyword': 0, 
            'score3_content_quality': 0
        })
        
        return df, user_notes, dim3_score
    except Exception as e:
        logger.error(f"数据预处理出错: {str(e)}")
        raise

# 优化的并行处理函数 - 使用CPU
def encode_texts_batch(texts, batch_size=16):
    """批量编码文本 - CPU优化版本"""
    global global_simcse_model
    
    # 确保使用全局模型
    if global_simcse_model is None:
        global_simcse_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
    
    embeddings = []
    
    # 使用更小的批量以适应CPU
    for i in range(0, len(texts), batch_size):
        batch = texts[i:min(i + batch_size, len(texts))]
        try:
            # 明确指定使用CPU设备
            batch_embeddings = global_simcse_model.encode(batch, device='cpu', show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.warning(f"编码批次 {i//batch_size + 1} 时出错: {str(e)}")
            # 为失败的批次提供零向量
            embeddings.extend([np.zeros(global_simcse_model.get_sentence_embedding_dimension()) for _ in range(len(batch))])
    
    return embeddings

def get_bert_sentiment_batch(texts_batch):
    """批量处理文本进行情感分析 - CPU版本"""
    device = get_device()  # 强制使用CPU
    results = {}
    
   # 确保模型已加载
    if models is None or 'sentiment_model' not in models:
        tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")
        model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese").to_empty(device=device).to(device)
        model.eval()
    else:
        tokenizer = models['sentiment_tokenizer']
        model = models['sentiment_model']
    
    # 使用较小的批量以适应CPU内存
    batch_size = 8  # CPU版本使用较小的批量
    
    for i in range(0, len(texts_batch), batch_size):
        batch = texts_batch[i:min(i+batch_size, len(texts_batch))]
        batch_ids = [idx for idx, _ in batch]
        batch_texts = [text for _, text in batch]
        
        # 过滤掉空文本或过短文本
        filtered_data = []
        for idx, text in zip(batch_ids, batch_texts):
            if text is None or not isinstance(text, str) or len(text.strip()) < 3:
                # 对于无效文本，设置中性情感
                results[idx] = 0.5
            else:
                filtered_data.append((idx, text))
        
        if not filtered_data:
            continue
            
        filtered_ids = [idx for idx, _ in filtered_data]
        filtered_texts = [text for _, text in filtered_data]
        
        try:
            # 使用try/except包裹所有可能出错的操作
            with torch.no_grad():
                # 添加最大长度限制，并处理截断
                inputs = tokenizer(filtered_texts, 
                                  return_tensors="pt", 
                                  padding=True, 
                                  truncation=True,
                                  max_length=512)
                
                # 确保所有tensor都移动到正确的设备上
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 执行推理
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # 获取积极情感的概率（假定积极情感的索引为1）
                positive_probs = probs[:, 1].cpu().numpy()
                
                # 保存结果
                for idx, prob in zip(filtered_ids, positive_probs):
                    results[idx] = float(prob)
        
        except Exception as e:
            logger.warning(f"情感批处理出错: {str(e)}")
            # 为当前批次中的所有ID设置默认值
            for idx in filtered_ids:
                if idx not in results:
                    results[idx] = 0.5
    
    return results

def process_keyword_batch(texts_batch, pattern):
    """批量处理文本进行关键词匹配"""
    results = {}
    
    for idx, text in texts_batch:
        if pd.isna(text) or text == "":
            results[idx] = 0
        else:
            results[idx] = len(pattern.findall(str(text)))
    
    return results

# 评分函数
def calculate_creator_originality(df, dim3_score, models, n_workers=MAX_WORKERS, batch_size=24):
    """
    创作者原创性评分计算, 分为内部多样性和外部原创性两个维度
    使用完全不抽样的方式：使用创作者的全部笔记进行计算
    
    参数:
        df (pd.DataFrame): 输入数据框，包含用户笔记和其他信息
        dim3_score (pd.DataFrame): 用户级评分数据框，将被更新
        models (dict): 包含所需模型的字典
        n_workers (int): 工作进程数量
        batch_size (int): 批处理大小
    
    返回:
        pd.DataFrame: 更新后的用户级评分数据框
    """
    logger.info("  - 计算创作者原创性...")
    
    # 确保工作进程不超过最大限制
    n_workers = min(n_workers, MAX_WORKERS)
    logger.info(f"  - 使用 {n_workers} 个CPU进程进行计算")
    
    # 按创作者分组准备数据
    user_texts = df.groupby('user_id')['semantic_proc_text'].apply(list).to_dict()
    user_to_indices = {name: group.index.tolist() for name, group in df.groupby('user_id')}
    
    # 存储各项指标
    internal_diversity = {}
    external_originality = {}
    
    # 1. 计算内部多样性（同时使用TF-IDF和SimCSE）
    logger.info("  - 计算创作者内部多样性...")

    # 每个用户使用独立的TF-IDF向量化器
    def process_user_texts(texts, batch_size):
        # 为每个用户创建新的TF-IDF向量化器
        user_tfidf = TfidfVectorizer(max_features=8000)
        
        try:
            # TF-IDF计算 - 使用用户特定的向量化器
            tfidf = user_tfidf.fit_transform(texts)
            tfidf_sim = cosine_similarity(tfidf)
            np.fill_diagonal(tfidf_sim, 0)
            tfidf_avg_sim = np.mean(tfidf_sim)
            
            # SimCSE计算 - 使用固定的CPU设备
            embeddings = encode_texts_batch(texts, batch_size=batch_size)
            emb_sim = cosine_similarity(embeddings)
            np.fill_diagonal(emb_sim, 0)
            emb_avg_sim = np.mean(emb_sim)
            
            return {"tfidf": tfidf_avg_sim, "simcse": emb_avg_sim}
        except Exception as e:
            # 捕获并记录错误
            logger.error(f"处理用户文本时出错: {str(e)}")
            # 返回默认值，防止整个过程失败
            return {"tfidf": 0.5, "simcse": 0.5}
    
    # 使用线程池而非进程池来避免内存问题
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for user_id, texts in user_texts.items():
            indices = user_to_indices[user_id]
            
            # 处理笔记数量少的情况
            if len(texts) < 20:  # 少于20篇
                internal_diversity[user_id] = 0.8  # 默认中上多样性分数
                continue
            
            # 提交任务
            futures[user_id] = executor.submit(
                process_user_texts, 
                texts,
                batch_size
            )
        
    # 收集结果
    for user_id, future in tqdm(futures.items(), desc="内部多样性计算"):
        try:
            result = future.result()
            if result:
                tfidf_avg_sim = result["tfidf"]
                emb_avg_sim = result["simcse"]
                
                # 结合两种方法（取平均相似度）
                avg_sim = 0.5 * tfidf_avg_sim + 0.5 * emb_avg_sim
                
                # 转换为内部多样性评分
                internal_diversity[user_id] = 1 - avg_sim
        except Exception as e:
            logger.warning(f"计算用户 {user_id} 的内部多样性时出错: {str(e)}")
            internal_diversity[user_id] = 0.5  # 出错时使用中等分数
    
    # 2. 计算外部原创性
    logger.info("  - 计算创作者之间的原创性差异...")
    
    # 为每个创作者构建代表性文本集
    representative_texts = {}
    for user_id in user_texts.keys():
        # 获取该创作者的所有笔记索引（已经按时间排序，最新的在前面）
        indices = user_to_indices[user_id]
        
        # 提取头部笔记（最新的10篇）
        top_indices = indices[:min(10, len(indices))]
        
        # 获取热门笔记（所有热门笔记，不再限制数量）
        hot_indices = df.loc[indices].query("hot_note == 1").index.tolist()
        
        # 合并已选择的索引并去重
        selected_indices = list(set(top_indices + hot_indices))
        
        # 如果最新加热门笔记不够100篇，按时间顺序补充
        if len(selected_indices) < 100 and len(indices) > len(selected_indices):
            # 找出尚未选择的笔记
            remaining_indices = [i for i in indices if i not in selected_indices]
            # 按顺序取足够的笔记补足到100篇
            additional_count = min(100 - len(selected_indices), len(remaining_indices))
            additional_indices = remaining_indices[:additional_count]
            # 合并所有选定的索引
            selected_indices = list(set(selected_indices + additional_indices))
        
        # 获取对应的文本
        representative_texts[user_id] = df.loc[selected_indices, 'semantic_proc_text'].tolist() if selected_indices else []
    
    # 对代表性文本进行编码
    logger.info("  - 对代表性文本进行语义编码...")
    user_embeddings = {}
    all_rep_texts = []
    user_mapping = []
    
    for user_id, texts in representative_texts.items():
        if not texts:  # 防止空文本列表
            continue
        all_rep_texts.extend(texts)
        user_mapping.extend([user_id] * len(texts))
    
    # 只对代表性文本使用SimCSE - 使用更小的批量
    if all_rep_texts:
        embeddings = []
        
        # 将文本分成更小的块以适应CPU内存
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            chunk_size = 100  # CPU处理使用更小的块大小
            
            for i in range(0, len(all_rep_texts), chunk_size):
                chunk_texts = all_rep_texts[i:min(i+chunk_size, len(all_rep_texts))]
                futures.append(executor.submit(encode_texts_batch, chunk_texts, batch_size=batch_size))
                
            # 收集结果
            for future in tqdm(futures, desc="编码代表性文本"):
                chunk_embeddings = future.result()
                embeddings.extend(chunk_embeddings)
        
        # 按创作者聚合
        for i, user_id in enumerate(user_mapping):
            if user_id not in user_embeddings:
                user_embeddings[user_id] = []
            if i < len(embeddings):
                user_embeddings[user_id].append(embeddings[i])
    
    # 计算创作者级平均表示
    for user_id, embs in user_embeddings.items():
        if embs:  # 确保有嵌入向量
            user_embeddings[user_id] = np.mean(embs, axis=0)
    
    # 计算创作者间相似度 - 使用简化版本避免并行计算问题
    user_ids = list(user_embeddings.keys())
    
    for i, user_id in enumerate(user_ids):
        user_emb = user_embeddings.get(user_id)
        if user_emb is None:
            external_originality[user_id] = 0.9  # 默认高的原创性
            continue
            
        similarities = []
        for j, other_id in enumerate(user_ids):
            if i != j:
                other_emb = user_embeddings.get(other_id)
                if other_emb is not None:
                    sim = cosine_similarity([user_emb], [other_emb])[0][0]
                    similarities.append(sim)
        
        # 外部原创性
        if similarities:
            # 不仅考虑平均相似度，也考虑最大相似度（最大相似度权重更高）
            avg_sim = np.mean(similarities)
            max_sim = np.max(similarities)
            weighted_sim = 0.6 * avg_sim + 0.4 * max_sim
            external_originality[user_id] = 1 - weighted_sim
        else:
            external_originality[user_id] = 0.9  # 默认高的原创性
    
    # 3. 结合两个维度并应用分数映射
    logger.info("  - 组合评分并应用分布校正...")
    user_originality = {}
    
    for user_id in user_texts.keys():
        int_div = internal_diversity.get(user_id, 0.5)
        ext_orig = external_originality.get(user_id, 0.5)
        
        # 组合分数
        combined_raw = 0.5 * int_div + 0.5 * ext_orig
        
        # Sigmoid映射到0-25区间
        user_originality[user_id] = 25 / (1 + np.exp(-8 * (combined_raw - 0.4)))
    
    # 分布校正
    scores = np.array(list(user_originality.values()))
    mean = np.mean(scores)
    std = np.std(scores)
    
    if std < 4:  # 分布过于集中
        # 双边拉伸但保持相对关系
        stretch_factor = 5 / std if std > 0 else 1.5
        for user_id in user_originality:
            diff = user_originality[user_id] - mean
            user_originality[user_id] = np.clip(mean + diff * stretch_factor, 0, 25)
            
            # 软化极端值
            if user_originality[user_id] > 23:
                user_originality[user_id] = 23 + (user_originality[user_id] - 23) * 0.5
            elif user_originality[user_id] < 2:
                user_originality[user_id] = 2 * (user_originality[user_id] / 2)
    
    # 直接更新dim3_score的score3a_originality列
    for user_id, score in user_originality.items():
        idx = dim3_score[dim3_score['user_id'] == user_id].index
        if len(idx) > 0:
            dim3_score.loc[idx, 'score3a_originality'] = score
    
    return dim3_score

def calculate_vertical_score(df, dim3_score, target_tags=VERTICAL_DOMAIN_TAGS, n_workers=MAX_WORKERS, batch_size=200):
    """
    计算垂直领域得分
    基于创作者笔记中包含目标标签子字符串的比例，使用Sigmoid映射
    
    参数:
        df (pd.DataFrame): 输入数据框，包含笔记信息
        dim3_score (pd.DataFrame): 用户级评分数据框，将被更新
        target_tags (set): 目标标签集合
        n_workers (int): 工作进程数量
        batch_size (int): 批处理大小
    
    返回:
        pd.DataFrame: 更新后的用户级评分数据框
    """
    logger.info(f"  - 计算垂直领域分数... (目标标签: {target_tags})")
    
    # 确保工作进程不超过最大限制
    n_workers = min(n_workers, MAX_WORKERS)
    logger.info(f"  - 使用 {n_workers} 个CPU进程进行计算")
    
    # 检查每个笔记的标签列表是否包含任何目标标签作为子字符串 - 使用线程池
    def has_target_tag_batch(tags_batch):
        results = {}
        for idx, tags in tags_batch:
            if not tags or pd.isna(tags):
                results[idx] = 0
                continue
            
            # 将用户标签拆分为列表
            user_tags = str(tags).split(',')
            
            # 检查每个用户标签是否包含任何目标标签作为子字符串
            has_tag = 0
            for user_tag in user_tags:
                user_tag = user_tag.strip().lower()
                for target_tag in target_tags:
                    target_tag = str(target_tag).lower()
                    if target_tag in user_tag:
                        has_tag = 1
                        break
                if has_tag:
                    break
            
            results[idx] = has_tag
        
        return results
    
    # 准备批处理 - 使用较大批次以减少开销
    tag_batches = []
    for i in range(0, len(df), batch_size):
        batch_indices = df.index[i:min(i+batch_size, len(df))]
        batch_data = [(idx, df.loc[idx, 'tag_list']) for idx in batch_indices]
        tag_batches.append(batch_data)
    
    # 使用线程池而非进程池 - 标签处理是轻量级的IO密集型任务
    has_target_tag_results = {}
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(has_target_tag_batch, batch) for batch in tag_batches]
        
        # 收集结果
        for future in tqdm(futures, desc="处理标签"):
            try:
                batch_results = future.result()
                has_target_tag_results.update(batch_results)
            except Exception as e:
                logger.error(f"处理标签批次出错: {str(e)}")
    
    # 添加标记列
    df['has_target_tag'] = pd.Series(has_target_tag_results)
    
    # 按创作者分组计算包含目标标签的笔记比例
    user_vertical_scores = {}
    raw_ratios = {}  # 存储原始比例，用于诊断
    
    for user_id, group in df.groupby('user_id'):
        # 计算包含目标标签的笔记比例
        target_tag_ratio = group['has_target_tag'].mean()
        raw_ratios[user_id] = target_tag_ratio  # 保存原始比例
        
        # 使用幂函数提升低比例的得分
        adjusted_ratio = target_tag_ratio ** 0.7  # 使用0.7次幂拉伸低分区间
        
        # Sigmoid映射参数
        slope = 5.0  # 控制曲线陡峭程度
        midpoint = 0.4  # 中心点，调整为0.4使整体分数偏高些
        
        # Sigmoid映射到0-25区间
        sigmoid_score = 25 / (1 + np.exp(-slope * (adjusted_ratio - midpoint)))
        
        # 存储sigmoid转换后的分数
        user_vertical_scores[user_id] = sigmoid_score
    
    # 输出原始分数统计
    scores = np.array(list(user_vertical_scores.values()))
    mean = np.mean(scores)
    std = np.std(scores)
    
    logger.info(f"  - Sigmoid转换后分数: 均值={mean:.2f}, 标准差={std:.2f}")
    
    # 特殊处理: 将0覆盖率用户设为0分
    for user_id, ratio in raw_ratios.items():
        if ratio == 0:
            user_vertical_scores[user_id] = 0
    
    # 处理极端情况: 防止极高或极低分过多
    high_scores = sum(1 for s in scores if s > 23)
    low_scores = sum(1 for s in scores if s > 0 and s < 2)
    
    # 如果高分或低分过多，微调分布
    if high_scores > len(scores) * 0.15 or low_scores > len(scores) * 0.15:
        logger.info("  - 检测到极端分数过多，应用轻微分布校正...")
        
        for user_id in user_vertical_scores:
            score = user_vertical_scores[user_id]
            # 仅对非零分进行处理
            if score > 0:
                # 软化高分
                if score > 23:
                    user_vertical_scores[user_id] = 23 + (score - 23) * 0.5
                # 提升低分
                elif score < 2:
                    user_vertical_scores[user_id] = 2 * (score / 2)
    
    # 最终分数统计
    corrected_scores = np.array(list(user_vertical_scores.values()))
    corrected_mean = np.mean(corrected_scores)
    corrected_std = np.std(corrected_scores)
    logger.info(f"  - 最终垂直领域分数: 均值={corrected_mean:.2f}, 标准差={corrected_std:.2f}")
    
    # 输出分布情况
    bins = [0, 5, 10, 15, 20, 25]
    hist, _ = np.histogram(corrected_scores, bins=bins)
    logger.info("  - 分数分布:")
    for i in range(len(bins)-1):
        logger.info(f"    {bins[i]}-{bins[i+1]}: {hist[i]} 人")
    
    # 更新dim3_score的score3b_vertical列
    for user_id, score in user_vertical_scores.items():
        idx = dim3_score[dim3_score['user_id'] == user_id].index
        if len(idx) > 0:
            dim3_score.loc[idx, 'score3b_vertical'] = score
    
    # 移除临时列
    if 'has_target_tag' in df.columns:
        df.drop(columns=['has_target_tag'], inplace=True)
    
    return dim3_score

def calculate_sentiment_score(df, dim3_score, models, max_notes_per_user=60, n_workers=MAX_WORKERS, batch_size=8):
    """
    计算情感强度得分，整合BERT情感分析、时间衰减和分布校正
    - 每个创作者最多只处理指定数量的笔记（默认60篇）
    - 优先选择最新的10篇笔记和所有热门笔记
    - 如果不足指定上限，添加更多最新笔记
    - 当max_notes_per_user=1000时，使用全部笔记
    
    参数:
        df (pd.DataFrame): 输入数据框，包含用户笔记
        dim3_score (pd.DataFrame): 用户级评分数据框，将被更新
        models (dict): 包含所需模型的字典
        max_notes_per_user (int): 每个用户最多处理的笔记数量，默认60，设为1000时使用全部笔记
        n_workers (int): 工作进程数量
        batch_size (int): 批处理大小
        
    返回:
        pd.DataFrame: 更新后的用户级评分数据框
    """
    logger.info("  - 计算情感强度得分...")
    logger.info(f"  - 每个用户最多处理 {max_notes_per_user} 篇笔记...")
    
    # 确保工作进程不超过最大限制
    n_workers = min(n_workers, MAX_WORKERS)
    logger.info(f"  - 使用 {n_workers} 个CPU进程进行计算")
    
    # 获取计算设备 - 强制CPU
    device = get_device()
    
    # 1. 为每个创作者选择要计算的笔记
    logger.info("  - 为每位创作者选择笔记...")
    user_selected_indices = {}
    
    for user_id, group in df.groupby('user_id'):
        # 按时间排序 - 确保索引是从新到旧排列的
        group = group.sort_values('elapsed_time', ascending=True)
        indices = group.index.tolist()
        
        # 检查是否使用全部笔记
        use_all_notes = (max_notes_per_user >= 1000)
        
        if use_all_notes:
            # 使用全部笔记
            selected_indices = indices
        else:
            # 选择最新的10篇笔记
            top_indices = indices[:min(10, len(indices))]
            
            # 选择所有热门笔记
            hot_indices = group[group['hot_note'] == 1].index.tolist()
            
            # 合并并去重
            selected_indices = list(set(top_indices + hot_indices))
            
            # 如果不足指定上限，添加更多最新笔记
            if len(selected_indices) < max_notes_per_user and len(indices) > len(selected_indices):
                # 找出尚未选择的最新笔记
                remaining_indices = [i for i in indices if i not in selected_indices]
                # 取足够的笔记以达到上限或用完所有笔记
                additional_count = min(max_notes_per_user - len(selected_indices), len(remaining_indices))
                additional_indices = remaining_indices[:additional_count]
                # 合并所有选定的索引
                selected_indices = list(set(selected_indices + additional_indices))
        
        # 存储选定的索引
        user_selected_indices[user_id] = selected_indices
    
    # 2. 只对选定的笔记计算情感值 - 使用更小的批处理来优化CPU处理
    logger.info("  - 计算选定笔记的情感值...")
    
    # 准备所有选定的笔记
    all_selected_texts = []
    for user_id, indices in user_selected_indices.items():
        for idx in indices:
            text = df.loc[idx, 'semantic_proc_text']
            if text and isinstance(text, str):
                all_selected_texts.append((idx, text))
    
    # 分批处理，使用较小的批次
    sentiment_batches = []
    for i in range(0, len(all_selected_texts), batch_size):
        sentiment_batches.append(all_selected_texts[i:min(i+batch_size, len(all_selected_texts))])
    
    # 使用较少的工作进程来处理情感分析批次
    sentiment_workers = min(n_workers, 4)  # 情感分析使用较少的工作进程以避免内存问题
    logger.info(f"  - 情感分析使用 {sentiment_workers} 个CPU工作进程处理")
    
    # 初始化情感值字典
    sentiment_values = {}
    
    # 使用线程池来协调批量情感分析任务
    with ThreadPoolExecutor(max_workers=sentiment_workers) as executor:
        futures = [executor.submit(get_bert_sentiment_batch, batch) for batch in sentiment_batches]
        
        # 收集结果
        for future in tqdm(futures, desc="处理情感批次"):
            try:
                batch_results = future.result()
                sentiment_values.update(batch_results)
            except Exception as e:
                logger.error(f"处理情感批次出错: {str(e)}")
    
    # 将情感值添加到DataFrame，未计算的设为0
    df['sentiment_raw'] = pd.Series(sentiment_values).reindex(df.index).fillna(0.0)
    
    # 3. 应用时间衰减系数
    logger.info("  - 应用时间衰减...")
    decay = np.exp(-0.05 * df['elapsed_time'] / 7)  # 一周衰减约30%
    df['sentiment'] = df['sentiment_raw'] * decay
    
    # 4. 按用户聚合情感分数
    logger.info("  - 聚合用户级情感分数...")
    user_sentiment_scores = {}
    
    # 使用绝对值处理情感分数 - 同时保留一部分极性
    df['sentiment_abs'] = df['sentiment'].abs()
    
    for user_id, indices in user_selected_indices.items():
        # 只使用选定的笔记计算情感得分
        selected_df = df.loc[indices]
        
        if not selected_df.empty:
            # 计算情感均值(极性)
            sentiment_mean = selected_df['sentiment'].mean()
            # 计算情感强度(绝对值)
            sentiment_abs_mean = selected_df['sentiment_abs'].mean()
            # 情感极值(最大正向和最小负向)
            sentiment_max = selected_df['sentiment'].max()
            sentiment_min = selected_df['sentiment'].min()
            
            # 组合多种情感特征 - 同时考虑极性和强度
            combined_sentiment = (0.3 * sentiment_mean + 
                                0.3 * sentiment_abs_mean + 
                                0.2 * sentiment_max + 
                                0.2 * abs(sentiment_min))
        else:
            combined_sentiment = 0.0
            
        # 使用Sigmoid函数映射到0-25区间
        slope = 5.0  # 控制曲线陡峭程度
        midpoint = 0.4  # 中心点位置
        
        # Sigmoid映射
        sigmoid_score = 25 / (1 + np.exp(-slope * (sentiment_mean - midpoint)))
        
        # 存储Sigmoid转换后的分数
        user_sentiment_scores[user_id] = sigmoid_score
    
    # 5. 分布校正
    logger.info("  - 校正情感分数分布...")
    scores = np.array(list(user_sentiment_scores.values()))
    mean = np.mean(scores)
    std = np.std(scores)
    
    logger.info(f"  - Sigmoid转换后情感分数: 均值={mean:.2f}, 标准差={std:.2f}")
    
    # 处理极端情况: 防止极高或极低分过多
    high_scores = sum(1 for s in scores if s > 23)
    low_scores = sum(1 for s in scores if s < 2)
    
    # 如果高分或低分过多，微调分布
    if high_scores > len(scores) * 0.15 or low_scores > len(scores) * 0.15:
        logger.info("  - 检测到极端分数过多，应用轻微分布校正...")
        
        for user_id in user_sentiment_scores:
            score = user_sentiment_scores[user_id]
            # 软化高分
            if score > 23:
                user_sentiment_scores[user_id] = 23 + (score - 23) * 0.5
            # 提升低分
            elif score < 2:
                user_sentiment_scores[user_id] = 2 * (score / 2)
    
    # 最终分数统计
    corrected_scores = np.array(list(user_sentiment_scores.values()))
    corrected_mean = np.mean(corrected_scores)
    corrected_std = np.std(corrected_scores)
    logger.info(f"  - 最终情感分数: 均值={corrected_mean:.2f}, 标准差={corrected_std:.2f}")
    
    # 输出分布情况
    bins = [0, 5, 10, 15, 20, 25]
    hist, _ = np.histogram(corrected_scores, bins=bins)
    logger.info("  - 分数分布:")
    for i in range(len(bins)-1):
        logger.info(f"    {bins[i]}-{bins[i+1]}: {hist[i]} 人")
    
    # 6. 更新dim3_score的score3c_sentiment列
    for user_id, score in user_sentiment_scores.items():
        idx = dim3_score[dim3_score['user_id'] == user_id].index
        if len(idx) > 0:
            dim3_score.loc[idx, 'score3c_sentiment'] = score
    
    # 7. 清理中间数据
    for col in ['sentiment_raw', 'sentiment', 'sentiment_abs']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    return dim3_score

def calculate_keyword_score(df, dim3_score, core_keywords, n_workers=MAX_WORKERS, batch_size=500):
    """
    计算关键词覆盖得分 - Sigmoid增强版
    1. 支持子字符串匹配
    2. 使用增强型Sigmoid函数映射，确保分数充分利用0-25范围
    3. 应用更激进的参数调整
    
    参数:
        df (pd.DataFrame): 输入数据框，包含笔记信息
        dim3_score (pd.DataFrame): 用户级评分数据框，将被更新
        core_keywords (set): 核心关键词集合
        n_workers (int): 工作进程数量
        batch_size (int): 批处理大小
    
    返回:
        pd.DataFrame: 更新后的用户级评分数据框
    """
    logger.info("  - 计算关键词覆盖得分...")

    # 确保工作进程不超过最大限制
    n_workers = min(n_workers, MAX_WORKERS)
    logger.info(f"  - 使用 {n_workers} 个CPU进程进行计算")

    # 1. 创建高效的关键词匹配模式 (预编译一次)
    logger.info("  - 创建关键词匹配模式...")
    pattern = re.compile('|'.join(re.escape(kw) for kw in core_keywords), 
                        flags=re.IGNORECASE)

    # 2. 计算关键词数量 - 批量处理使用线程池
    logger.info("  - 计算笔记级关键词覆盖...")
    
    # 准备批处理 - 使用较大的批量以减少开销
    text_batches = []
    for i in range(0, len(df), batch_size):
        batch_indices = df.index[i:min(i+batch_size, len(df))]
        batch_data = [(idx, df.loc[idx, 'semantic_proc_text']) for idx in batch_indices]
        text_batches.append(batch_data)
    
    # 使用线程池进行关键词匹配
    kw_count_results = {}
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_keyword_batch, batch, pattern) for batch in text_batches]
        
        # 收集结果
        for future in tqdm(futures, desc="关键词匹配"):
            try:
                batch_results = future.result()
                kw_count_results.update(batch_results)
            except Exception as e:
                logger.error(f"处理关键词批次出错: {str(e)}")
    
    # 添加关键词计数到DataFrame
    df['kw_count'] = pd.Series(kw_count_results)

    # 3. 应用时间衰减系数
    logger.info("  - 应用时间衰减系数...")
    decay = np.exp(-0.05 * df['elapsed_time'] / 7)

    # 4. 计算单篇得分
    df['single_kw_score'] = np.maximum(1, df['kw_count'] / 5) * decay

    # 5. 按创作者聚合关键词得分
    logger.info("  - 聚合用户级关键词得分...")
    user_keyword_scores = {}
    raw_ratios = {}  # 存储原始比例，用于诊断

    for user_id, group in df.groupby('user_id'):
        # 计算覆盖比例
        cover_rate = (group['kw_count'] > 0).mean()
        raw_ratios[user_id] = cover_rate  # 保存原始覆盖率
        
        # 计算平均得分并标准化
        avg_score = group['single_kw_score'].mean()
        
        # 根据文档中的比例组合指标，覆盖率基准值默认为0.05
        combined_raw = 0.7 * avg_score + 0.3 * (cover_rate / 0.05)
        
        # 使用sigmoid函数映射到0-25区间
        normalized_score = combined_raw / 2 - 0.2  # 标准化到合理范围
        
        # 应用sigmoid函数 - 调整参数使分布更分散
        slope = 3
        midpoint = 0.4  # 调整中点位置使分数分布更均匀
        
        # Sigmoid映射到0-25区间
        sigmoid_score = 25 / (1 + np.exp(-slope * (normalized_score - midpoint)))
        
        # 存储sigmoid转换后的分数
        user_keyword_scores[user_id] = sigmoid_score

    # 输出原始分数统计
    scores = np.array(list(user_keyword_scores.values()))
    mean = np.mean(scores)
    std = np.std(scores)
    
    logger.info(f"  - Sigmoid转换后分数: 均值={mean:.2f}, 标准差={std:.2f}")
    
    # 特殊处理: 将0覆盖率用户设为0分
    for user_id, ratio in raw_ratios.items():
        if ratio == 0:
            user_keyword_scores[user_id] = 0
    
    # 处理极端情况: 防止极高或极低分过多
    high_scores = sum(1 for s in scores if s > 23)
    low_scores = sum(1 for s in scores if s > 0 and s < 2)
    
    # 如果高分或低分过多，微调分布
    if high_scores > len(scores) * 0.15 or low_scores > len(scores) * 0.15:
        logger.info("  - 检测到极端分数过多，应用轻微分布校正...")
        
        for user_id in user_keyword_scores:
            score = user_keyword_scores[user_id]
            # 仅对非零分进行处理
            if score > 0:
                # 软化高分
                if score > 23:
                    user_keyword_scores[user_id] = 23 + (score - 23) * 0.5
                # 提升低分
                elif score < 2:
                    user_keyword_scores[user_id] = 2 * (score / 2)
    
    # 最终分数统计
    corrected_scores = np.array(list(user_keyword_scores.values()))
    corrected_mean = np.mean(corrected_scores)
    corrected_std = np.std(corrected_scores)
    logger.info(f"  - 最终关键词分数: 均值={corrected_mean:.2f}, 标准差={corrected_std:.2f}")
    
    # 输出分布情况
    bins = [0, 5, 10, 15, 20, 25]
    hist, _ = np.histogram(corrected_scores, bins=bins)
    logger.info("  - 分数分布:")
    for i in range(len(bins)-1):
        logger.info(f"    {bins[i]}-{bins[i+1]}: {hist[i]} 人")

    # 7. 更新dim3_score的keyword_score列
    for user_id, score in user_keyword_scores.items():
        idx = dim3_score[dim3_score['user_id'] == user_id].index
        if len(idx) > 0:
            dim3_score.loc[idx, 'score3d_keyword'] = score

    # 8. 清理临时列
    if 'kw_count' in df.columns:
        df.drop(columns=['kw_count'], inplace=True)
    if 'single_kw_score' in df.columns:
        df.drop(columns=['single_kw_score'], inplace=True)

    return dim3_score

def plot_score_distributions(dim3_score, output_dir):
    """绘制分数分布图并保存"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 原创性分数分布
        plt.figure(figsize=(10, 6))
        sns.histplot(dim3_score['score3a_originality'], bins=20, kde=True, color='blue', stat='count')
        plt.title('Distribution of Creator Originality Scores')
        plt.xlabel('Originality Score')
        plt.ylabel('Count')
        plt.xlim(0, 25)
        plt.xticks(np.arange(0, 26, 5))
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dim3_score_distribution.png'))
        plt.close()
        
        # 垂直领域分数分布
        plt.figure(figsize=(10, 6))
        sns.histplot(dim3_score['score3b_vertical'], bins=30, kde=True, color='green', stat='count')
        plt.title('Distribution of Vertical Scores')
        plt.xlabel('Vertical Score')
        plt.ylabel('Count')
        plt.xlim(0, 25)
        plt.xticks(np.arange(0, 26, 5))
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dim3_vertical_score_distribution.png'))
        plt.close()
        
        # 情感分数分布
        plt.figure(figsize=(10, 6))
        sns.histplot(dim3_score['score3c_sentiment'], bins=30, kde=True, color='orange', stat='count')
        plt.title('Distribution of Sentiment Scores')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Count')
        plt.xlim(0, 25)
        plt.xticks(np.arange(0, 26, 5))
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dim3_score3c_sentiment_distribution.png'))
        plt.close()
        
        # 关键词分数分布
        plt.figure(figsize=(10, 6))
        sns.histplot(dim3_score['score3d_keyword'], bins=30, kde=True, color='red', stat='count')
        plt.title('Distribution of Keyword Scores')
        plt.xlabel('Keyword Coverage Score')
        plt.ylabel('Count')
        plt.xlim(0, 25)
        plt.xticks(np.arange(0, 26, 5))
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dim3_keyword_score_distribution.png'))
        plt.close()
        
        # 内容质量总分分布
        plt.figure(figsize=(10, 6))
        sns.histplot(dim3_score['score3_content_quality'], bins=25, kde=True, color='purple', stat='count')
        plt.title('Distribution of Content Quality Scores')
        plt.xlabel('Content Quality Score')
        plt.ylabel('Count')
        plt.xlim(0, 100)
        plt.xticks(np.arange(0, 101, 10))
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dim3_content_quality_score_distribution.png'))
        plt.close()
        
        logger.info(f"分数分布图保存到 {output_dir}")
        
    except Exception as e:
        logger.error(f"绘制分数分布图时出错: {str(e)}")

def main(input_path, output_path, workers=MAX_WORKERS, max_notes=60, plots_dir=None):
    """主流程：计算所有内容分数"""
    start_time = time.time()
    
    # 确保工作进程不超过最大限制
    workers = min(workers, MAX_WORKERS)
    logger.info(f"开始内容评分流程，使用 {workers} 个工作进程 (限制最多使用 {MAX_WORKERS} 个)")
    
    try:
        # 1. 加载数据
        logger.info(f"从 {input_path} 加载数据...")
        df = pd.read_json(input_path, lines=True)
        logger.info(f"加载了 {len(df)} 条记录")
        
        # 2. 加载模型
        logger.info("加载模型...")
        models = load_models()
        
        # 3. 预处理数据
        logger.info("预处理数据...")
        df, user_notes, dim3_score = preprocess_data(df)
        
        # 4. 计算原创性评分
        logger.info("计算创作者原创性评分...")
        dim3_score = calculate_creator_originality(df, dim3_score, models, n_workers=workers, batch_size=24)
        
        # 5. 计算垂直领域评分
        logger.info("计算垂直领域评分...")
        dim3_score = calculate_vertical_score(df, dim3_score, VERTICAL_DOMAIN_TAGS, n_workers=workers, batch_size=200)
        
        # 6. 计算情感评分
        logger.info("计算情感评分...")
        dim3_score = calculate_sentiment_score(df, dim3_score, models, max_notes_per_user=max_notes, n_workers=workers, batch_size=8)
        
        # 7. 计算关键词覆盖评分
        logger.info("计算关键词覆盖评分...")
        dim3_score = calculate_keyword_score(df, dim3_score, CORE_KEYWORDS, n_workers=workers, batch_size=500)
        
        # 8. 计算内容质量总分
        logger.info("计算内容质量总分...")
        dim3_score['score3_content_quality'] = dim3_score[['score3a_originality', 'score3b_vertical', 
                                                          'score3c_sentiment', 'score3d_keyword']].sum(axis=1)
        
        # 9. 保存结果
        logger.info(f"保存结果到 {output_path}...")
        dim3_score.to_json(output_path, orient='records', lines=True)
        
        # 10. 生成图表
        if plots_dir:
            logger.info("生成分数分布图...")
            plot_score_distributions(dim3_score, plots_dir)
        
        elapsed_time = time.time() - start_time
        logger.info(f"内容评分完成，耗时 {elapsed_time:.2f} 秒")
        
        return dim3_score
        
    except Exception as e:
        logger.error(f"内容评分过程出错: {str(e)}")
        raise

# 支持直接接受DataFrame作为输入的版本
def score_dataframe(df, output_path=None, workers=MAX_WORKERS, max_notes=60, plots_dir=None):
    """与main函数功能相同，但直接接受DataFrame作为输入"""
    start_time = time.time()
    
    # 确保工作进程不超过最大限制
    workers = min(workers, MAX_WORKERS)
    logger.info(f"开始内容评分流程，使用 {workers} 个工作进程 (限制最多使用 {MAX_WORKERS} 个)")
    
    try:
        # 1. 加载模型
        logger.info("加载模型...")
        models = load_models()
        
        # 2. 预处理数据
        logger.info("预处理数据...")
        df_copy, user_notes, dim3_score = preprocess_data(df)
        
        # 3. 计算原创性评分
        logger.info("计算创作者原创性评分...")
        dim3_score = calculate_creator_originality(df_copy, dim3_score, models, n_workers=workers, batch_size=24)
        
        # 4. 计算垂直领域评分
        logger.info("计算垂直领域评分...")
        dim3_score = calculate_vertical_score(df_copy, dim3_score, VERTICAL_DOMAIN_TAGS, n_workers=workers, batch_size=200)
        
        # 5. 计算情感评分
        logger.info("计算情感评分...")
        dim3_score = calculate_sentiment_score(df_copy, dim3_score, models, max_notes_per_user=max_notes, n_workers=workers, batch_size=8)
        
        # 6. 计算关键词覆盖评分
        logger.info("计算关键词覆盖评分...")
        dim3_score = calculate_keyword_score(df_copy, dim3_score, CORE_KEYWORDS, n_workers=workers, batch_size=500)
        
        # 7. 计算内容质量总分
        logger.info("计算内容质量总分...")
        dim3_score['score3_content_quality'] = dim3_score[['score3a_originality', 'score3b_vertical', 
                                                          'score3c_sentiment', 'score3d_keyword']].sum(axis=1)
        
        # 8. 保存结果 (如果提供了输出路径)
        if output_path:
            logger.info(f"保存结果到 {output_path}...")
            dim3_score.to_json(output_path, orient='records', lines=True)
        
        # 9. 生成图表
        if plots_dir:
            logger.info("生成分数分布图...")
            plot_score_distributions(dim3_score, plots_dir)
        
        elapsed_time = time.time() - start_time
        logger.info(f"内容评分完成，耗时 {elapsed_time:.2f} 秒")
        
        return dim3_score
        
    except Exception as e:
        logger.error(f"内容评分过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用并行处理计算内容质量评分 (仅CPU版)')
    parser.add_argument('--input', type=str, default='../../Data/processed/contents_cooked_semantic.json', 
                        help='输入JSON文件路径')
    parser.add_argument('--output', type=str, default='../../Data/processed/dim3_scores.json',
                        help='输出JSON文件路径')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, 
                        help=f'工作进程数量（默认：{MAX_WORKERS}，不超过{MAX_WORKERS}）')
    parser.add_argument('--max-notes', type=int, default=60,
                        help='每个用户进行情感分析的最大笔记数量（默认: 60，使用1000+表示所有笔记）')
    parser.add_argument('--plots', type=str, default='../../Data/processed', 
                        help='保存分布图的目录（默认：无）')
    
    args = parser.parse_args()
    
    # 确保工作进程不超过最大限制
    workers = min(args.workers, MAX_WORKERS)
    
    main(args.input, args.output, workers, args.max_notes, args.plots)