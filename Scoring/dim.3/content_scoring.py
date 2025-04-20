import pandas as pd
import numpy as np
import jieba
import re
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import warnings
import time
import os
import argparse

warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")

# Define core keywords and vertical domain tags
CORE_KEYWORDS = {
    '鲜芋仙', 'Meet Fresh', 'MeetFresh', '台湾美食', '甜品',
    '芋圆', 'taro', '仙草', 'grass jelly', '奶茶', 'milk tea',
    '豆花', 'tofu pudding', '奶刨冰', 'milked shaved ice', 
    '红豆汤', 'purple rice soup', '紫米粥', 'red bean soup',
    '2001 Coit Rd', 'Park Pavillion Center', '(972) 596-6088',
    '刘一手', '锅底', '火锅', '牛奶冰'
}

VERTICAL_DOMAIN_TAGS = {
    '美食', '探店', '火锅', '甜品',
    '奶茶', '外卖', '中餐', '小吃',
    'food'
}

def load_models():
    """Load all required models"""
    models = {}
    
    # Originality detection models
    models['tfidf'] = TfidfVectorizer(max_features=8000)
    models['simcse'] = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Sentiment analysis models
    models['sentiment_tokenizer'] = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")
    models['sentiment_model'] = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")
    models['sentiment_model'].eval()
    
    return models

def preprocess_data(df):
    """Preprocess data and create user-level scoring DataFrame"""
    df = df.copy()
    
    # User notes mapping
    user_notes = df.groupby('user_id')['semantic_proc_text'].apply(list)
    
    # Create DataFrame with all user IDs (dim3_score)
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

def calculate_creator_originality(df, dim3_score, models):
    """
    Calculate creator originality scores based on internal diversity and external originality
    
    Args:
        df (pd.DataFrame): Input data frame with user notes
        dim3_score (pd.DataFrame): User-level scoring DataFrame to be updated
        models (dict): Dictionary containing required models
    
    Returns:
        pd.DataFrame: Updated user-level scoring DataFrame
    """
    print("  - Calculating creator originality...")
    
    # Prepare data by creator
    user_texts = df.groupby('user_id')['semantic_proc_text'].apply(list).to_dict()
    user_to_indices = {name: group.index.tolist() for name, group in df.groupby('user_id')}
    
    # Store metrics
    internal_diversity = {}
    external_originality = {}
    
    # 1. Calculate internal diversity (using both TF-IDF and SimCSE)
    print("  - Calculating creator internal diversity...")
    for user_id, texts in user_texts.items():
        indices = user_to_indices[user_id]
        
        # Handle cases with few notes
        if len(texts) < 20:  # Less than 20 notes
            internal_diversity[user_id] = 0.8  # Default above-average diversity score
            continue
        
        # Use all texts - no sampling
        all_texts = texts
            
        # Calculate diversity using two methods
        # 1. TF-IDF calculation
        tfidf = models['tfidf'].fit_transform(all_texts)
        tfidf_sim = cosine_similarity(tfidf)
        np.fill_diagonal(tfidf_sim, 0)
        tfidf_avg_sim = np.mean(tfidf_sim)
        
        # 2. SimCSE calculation
        embeddings = models['simcse'].encode(all_texts, show_progress_bar=False)
        emb_sim = cosine_similarity(embeddings)
        np.fill_diagonal(emb_sim, 0)
        emb_avg_sim = np.mean(emb_sim)
        
        # Combine both methods (average similarity)
        avg_sim = 0.5 * tfidf_avg_sim + 0.5 * emb_avg_sim
        
        # Convert to internal diversity score
        internal_diversity[user_id] = 1 - avg_sim
    
    # 2. Calculate external originality
    print("  - Calculating originality differences between creators...")
    
    # Build representative text set for each creator
    representative_texts = {}
    for user_id in user_texts.keys():
        # Get all note indices for this creator (already sorted by time, newest first)
        indices = user_to_indices[user_id]
        
        # Extract top notes (latest 10)
        top_indices = indices[:min(10, len(indices))]
        
        # Get hot notes (all hot notes, no limit)
        hot_indices = df.loc[indices].query("hot_note == 1").index.tolist()
        
        # Combine and deduplicate selected indices
        selected_indices = list(set(top_indices + hot_indices))
        
        # If latest plus hot notes are less than 100, add more by time order
        if len(selected_indices) < 100 and len(indices) > len(selected_indices):
            # Find notes not yet selected
            remaining_indices = [i for i in indices if i not in selected_indices]
            # Take enough notes to reach 100 or use all remaining
            additional_count = min(100 - len(selected_indices), len(remaining_indices))
            additional_indices = remaining_indices[:additional_count]
            # Combine all selected indices
            selected_indices = list(set(selected_indices + additional_indices))
        
        # Get corresponding texts
        representative_texts[user_id] = df.loc[selected_indices, 'semantic_proc_text'].tolist() if selected_indices else []
    
    # Encode representative texts
    print("  - Encoding representative texts semantically...")
    user_embeddings = {}
    all_rep_texts = []
    user_mapping = []
    
    for user_id, texts in representative_texts.items():
        if not texts:  # Prevent empty text lists
            continue
        all_rep_texts.extend(texts)
        user_mapping.extend([user_id] * len(texts))
    
    # Use SimCSE only for representative texts
    if all_rep_texts:
        embeddings = models['simcse'].encode(all_rep_texts, show_progress_bar=True)
        
        # Aggregate by creator
        for i, user_id in enumerate(user_mapping):
            if user_id not in user_embeddings:
                user_embeddings[user_id] = []
            user_embeddings[user_id].append(embeddings[i])
    
    # Calculate creator-level average representations
    for user_id, embs in user_embeddings.items():
        if embs:  # Ensure there are embedding vectors
            user_embeddings[user_id] = np.mean(embs, axis=0)
    
    # Calculate similarity between creators
    user_ids = list(user_embeddings.keys())
    
    for i, user_id in enumerate(user_ids):
        user_emb = user_embeddings[user_id]
        similarities = []
        
        for j, other_id in enumerate(user_ids):
            if i != j:
                other_emb = user_embeddings[other_id]
                sim = cosine_similarity([user_emb], [other_emb])[0][0]
                similarities.append(sim)
        
        # External originality
        if similarities:
            # Consider both average similarity and maximum similarity (higher weight for max)
            avg_sim = np.mean(similarities)
            max_sim = np.max(similarities)
            weighted_sim = 0.6 * avg_sim + 0.4 * max_sim
            external_originality[user_id] = 1 - weighted_sim
        else:
            external_originality[user_id] = 0.9  # Default high originality
    
    # 3. Combine dimensions and apply score mapping
    print("  - Combining scores and applying distribution correction...")
    user_originality = {}
    
    for user_id in user_texts.keys():
        int_div = internal_diversity.get(user_id, 0.5)
        ext_orig = external_originality.get(user_id, 0.5)
        
        # Combined score
        combined_raw = 0.5 * int_div + 0.5 * ext_orig
        
        # Sigmoid mapping to 0-25 range
        user_originality[user_id] = 25 / (1 + np.exp(-8 * (combined_raw - 0.4)))
    
    # Distribution correction
    scores = np.array(list(user_originality.values()))
    mean = np.mean(scores)
    std = np.std(scores)
    
    if std < 4:  # Distribution too concentrated
        # Bilateral stretch but maintain relative relationships
        stretch_factor = 5 / std if std > 0 else 1.5
        for user_id in user_originality:
            diff = user_originality[user_id] - mean
            user_originality[user_id] = np.clip(mean + diff * stretch_factor, 0, 25)
            
            # Soften extreme values
            if user_originality[user_id] > 23:
                user_originality[user_id] = 23 + (user_originality[user_id] - 23) * 0.5
            elif user_originality[user_id] < 2:
                user_originality[user_id] = 2 * (user_originality[user_id] / 2)
    
    # Directly update dim3_score's score3a_originality column
    for user_id, score in user_originality.items():
        idx = dim3_score[dim3_score['user_id'] == user_id].index
        if len(idx) > 0:
            dim3_score.loc[idx, 'score3a_originality'] = score
    
    return dim3_score

def calculate_vertical_score(df, dim3_score, target_tags=VERTICAL_DOMAIN_TAGS):
    """
    Calculate vertical domain scores
    Based on the proportion of notes containing target tags, using Sigmoid mapping
    
    Args:
        df (pd.DataFrame): Input data frame with note information
        dim3_score (pd.DataFrame): User-level scoring DataFrame to be updated
        target_tags (set): Set of target tags
    
    Returns:
        pd.DataFrame: Updated user-level scoring DataFrame
    """
    print(f"  - Calculating vertical domain scores... (Target tags: {target_tags})")
    
    # Check if any tag in the notes' tag list contains any target tag as a substring
    def has_target_tag(tags):
        if not tags or pd.isna(tags):
            return 0
        
        # Split user tags into list
        user_tags = str(tags).split(',')
        
        # Check if any user tag contains any target tag as substring
        for user_tag in user_tags:
            user_tag = user_tag.strip().lower()  # Remove whitespace and convert to lowercase
            for target_tag in target_tags:
                target_tag = str(target_tag).lower()  # Convert to lowercase for case-insensitive comparison
                if target_tag in user_tag:  # If target tag is a substring of user tag
                    return 1
                    
        return 0
    
    # Add flag column
    df['has_target_tag'] = df['tag_list'].apply(has_target_tag)
    
    # Group by creator and calculate proportion of notes with target tags
    user_vertical_scores = {}
    raw_ratios = {}  # Store raw ratios for diagnostics
    
    for user_id, group in df.groupby('user_id'):
        # Calculate proportion of notes containing target tags
        target_tag_ratio = group['has_target_tag'].mean()
        raw_ratios[user_id] = target_tag_ratio  # Save raw ratio
        
        # Use power function to boost low ratio scores - keep this step
        adjusted_ratio = target_tag_ratio ** 0.7  # Use 0.7 power to stretch low score range
        
        # Directly use Sigmoid function for mapping - adjust parameters for more even distribution
        # Adjust slope from 8 to 5 for a steeper curve and more spread distribution
        slope = 5.0  # Controls curve steepness
        midpoint = 0.4  # Center point, adjusted to 0.4 for somewhat higher overall scores
        
        # Sigmoid mapping to 0-25 range
        sigmoid_score = 25 / (1 + np.exp(-slope * (adjusted_ratio - midpoint)))
        
        # Store sigmoid transformed score
        user_vertical_scores[user_id] = sigmoid_score
    
    # Output raw score statistics
    scores = np.array(list(user_vertical_scores.values()))
    mean = np.mean(scores)
    std = np.std(scores)
    
    print(f"  - Sigmoid-transformed scores: mean={mean:.2f}, std={std:.2f}")
    
    # Special handling: Set users with 0 coverage to 0 points
    for user_id, ratio in raw_ratios.items():
        if ratio == 0:
            user_vertical_scores[user_id] = 0
    
    # Handle extreme cases: Prevent too many extremely high or low scores
    high_scores = sum(1 for s in scores if s > 23)
    low_scores = sum(1 for s in scores if s > 0 and s < 2)
    
    # If too many high or low scores, adjust distribution
    if high_scores > len(scores) * 0.15 or low_scores > len(scores) * 0.15:
        print("  - Detected too many extreme scores, applying mild distribution correction...")
        
        for user_id in user_vertical_scores:
            score = user_vertical_scores[user_id]
            # Only process non-zero scores
            if score > 0:
                # Soften high scores
                if score > 23:
                    user_vertical_scores[user_id] = 23 + (score - 23) * 0.5
                # Boost low scores
                elif score < 2:
                    user_vertical_scores[user_id] = 2 * (score / 2)
    
    # Final score statistics
    corrected_scores = np.array(list(user_vertical_scores.values()))
    corrected_mean = np.mean(corrected_scores)
    corrected_std = np.std(corrected_scores)
    print(f"  - Final vertical domain scores: mean={corrected_mean:.2f}, std={corrected_std:.2f}")
    
    # Output distribution
    bins = [0, 5, 10, 15, 20, 25]
    hist, _ = np.histogram(corrected_scores, bins=bins)
    print("  - Score distribution:")
    for i in range(len(bins)-1):
        print(f"    {bins[i]}-{bins[i+1]}: {hist[i]} people")
    
    # Update dim3_score's score3b_vertical column
    for user_id, score in user_vertical_scores.items():
        idx = dim3_score[dim3_score['user_id'] == user_id].index
        if len(idx) > 0:
            dim3_score.loc[idx, 'score3b_vertical'] = score
    
    # Remove temporary column
    if 'has_target_tag' in df.columns:
        df.drop(columns=['has_target_tag'], inplace=True)
    
    return dim3_score

def calculate_sentiment_score(df, dim3_score, models, max_notes_per_user=60):
    """
    Calculate sentiment intensity scores, integrating BERT sentiment analysis,
    time decay, and distribution correction
    
    - For each creator, process at most the specified number of notes (default 60)
    - Prioritize the latest 10 notes and all hot notes
    - If below the limit, add more recent notes
    - When max_notes_per_user=1000, use all notes
    
    Args:
        df (pd.DataFrame): Input data frame with user notes
        dim3_score (pd.DataFrame): User-level scoring DataFrame to be updated
        models (dict): Dictionary containing required models
        max_notes_per_user (int): Maximum number of notes to process per user
            default 60, when set to 1000 use all notes
        
    Returns:
        pd.DataFrame: Updated user-level scoring DataFrame
    """
    print("  - Calculating sentiment intensity scores...")
    print(f"  - Processing at most {max_notes_per_user} notes per user...")
    
    # Inner function: Use BERT to calculate sentiment value for a single note
    def get_bert_sentiment(text, tokenizer, model):
        """Calculate sentiment value for a single note using BERT"""
        if not text or not isinstance(text, str):
            return 0.0
            
        try:
            # Truncate long text for performance
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # For classification models, use logits directly
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            # Use probability of positive sentiment minus negative sentiment, range -1 to 1
            sentiment_score = (probabilities[0, 1] - probabilities[0, 0]).item()
            # Enhance sentiment extremes for more dispersed distribution
            if sentiment_score > 0:
                sentiment_score = sentiment_score ** 0.85  # Lower exponent for stronger positive sentiment
            else:
                sentiment_score = -(abs(sentiment_score) ** 0.85)  # Enhance negative sentiment
            return sentiment_score
        except Exception as e:
            print(f"  - Warning: Error in sentiment calculation: {str(e)[:100]}...")
            return 0.0
    
    # 1. Select notes to process for each creator
    print("  - Selecting notes for each creator...")
    user_selected_indices = {}
    
    for user_id, group in df.groupby('user_id'):
        # Sort by time - ensure indices are sorted from newest to oldest
        group = group.sort_values('elapsed_time', ascending=True)
        indices = group.index.tolist()
        
        # Check if using all notes
        use_all_notes = (max_notes_per_user >= 1000)
        
        if use_all_notes:
            # Use all notes
            selected_indices = indices
            print(f"User {user_id}: Using all {len(selected_indices)} notes for sentiment analysis")
        else:
            # Select the newest 10 notes
            top_indices = indices[:min(10, len(indices))]
            
            # Select all hot notes
            hot_indices = group[group['hot_note'] == 1].index.tolist()
            
            # Combine and deduplicate
            selected_indices = list(set(top_indices + hot_indices))
            
            # If below specified limit, add more recent notes
            if len(selected_indices) < max_notes_per_user and len(indices) > len(selected_indices):
                # Find not yet selected newest notes
                remaining_indices = [i for i in indices if i not in selected_indices]
                # Take enough notes to reach limit or use all notes
                additional_count = min(max_notes_per_user - len(selected_indices), len(remaining_indices))
                additional_indices = remaining_indices[:additional_count]
                # Combine all selected indices
                selected_indices = list(set(selected_indices + additional_indices))
        
        # Store selected indices
        user_selected_indices[user_id] = selected_indices
    
    # 2. Calculate sentiment values only for selected notes
    print("  - Calculating sentiment values for selected notes...")
    # Create a mapping to store calculated sentiment values
    sentiment_values = {}
    
    # Use a small sample to estimate average processing time
    sample_size = min(50, len(df))
    sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
    
    # Estimate processing time
    start = time.time()
    sample_sentiments = [get_bert_sentiment(text, models['sentiment_tokenizer'], models['sentiment_model']) 
                       for text in sample_df['semantic_proc_text'].values[:5]]
    elapsed = time.time() - start
    
    # Output some sample values for diagnostics
    print(f"  - Sample sentiment values: {[round(s, 4) for s in sample_sentiments[:5]]}")
    
    # Calculate total number of selected notes for estimating total processing time
    total_selected_notes = sum(len(indices) for indices in user_selected_indices.values())
    est_total_time = elapsed * total_selected_notes / 5 / 60  # minutes
    print(f"  - Estimated total time for sentiment analysis: {est_total_time:.1f} minutes (processing {total_selected_notes} notes)")
    
    # Process selected notes for each user
    for user_id, indices in user_selected_indices.items():
        for idx in indices:
            text = df.loc[idx, 'semantic_proc_text']
            sentiment_values[idx] = get_bert_sentiment(text, models['sentiment_tokenizer'], models['sentiment_model'])
    
    # Add sentiment values to DataFrame, set uncalculated ones to 0
    df['sentiment_raw'] = pd.Series(sentiment_values).reindex(df.index).fillna(0.0)
    
    # Print sentiment value statistics for diagnostics
    print(f"  - Raw sentiment value statistics: {df['sentiment_raw'].describe()}")
    
    # 3. Apply time decay coefficient
    print("  - Applying time decay...")
    # Weaken time decay influence, from 0.05 to 0.03
    decay = np.exp(-0.03 * df['elapsed_time'] / 7)  # ~3% decay per week
    df['sentiment'] = df['sentiment_raw'] * decay
    
    # 4. Aggregate sentiment scores by user
    print("  - Aggregating user-level sentiment scores...")
    user_sentiment_scores = {}
    
    # Use absolute values to process sentiment scores - also retain some polarity
    df['sentiment_abs'] = df['sentiment'].abs()
    
    for user_id, indices in user_selected_indices.items():
        # Only use selected notes to calculate sentiment scores
        selected_df = df.loc[indices]
        
        if not selected_df.empty:
            # Calculate sentiment mean (polarity)
            sentiment_mean = selected_df['sentiment'].mean()
            # Calculate sentiment intensity (absolute value)
            sentiment_abs_mean = selected_df['sentiment_abs'].mean()
            # Sentiment extremes (max positive and min negative)
            sentiment_max = selected_df['sentiment'].max()
            sentiment_min = selected_df['sentiment'].min()
            
            # Adjust combination method - increase extreme value weights
            combined_sentiment = (0.25 * sentiment_mean + 
                              0.25 * sentiment_abs_mean * 1.2 + # Enhance absolute value
                              0.3 * sentiment_max + # Increase max value weight
                              0.2 * abs(sentiment_min))
            
            # Enhance sentiment signal for more dispersed scores
            # Add non-linear amplification, making high scores higher and low scores lower
            amplified_sentiment = np.sign(combined_sentiment) * (abs(combined_sentiment) ** 0.9) * 1.3
        else:
            amplified_sentiment = 0.0
            
        # Use modified Sigmoid function mapping to 0-25 range
        slope = 4.0  # Lower slope for more dispersed distribution
        midpoint = 0.3  # Lower midpoint to increase overall scores
        
        # Sigmoid mapping
        sigmoid_score = 25 / (1 + np.exp(-slope * (amplified_sentiment - midpoint)))
        
        # Store Sigmoid transformed score
        user_sentiment_scores[user_id] = sigmoid_score
    
    # 5. Distribution correction - modify correction method
    print("  - Correcting sentiment score distribution...")
    scores = np.array(list(user_sentiment_scores.values()))
    mean = np.mean(scores)
    std = np.std(scores)
    
    print(f"  - Sigmoid-transformed sentiment scores: mean={mean:.2f}, std={std:.2f}")
    
    # Use more aggressive distribution stretching
    print("  - Applying enhanced distribution stretching...")
    
    # Force stretch distribution
    target_std = 5.5  # Target standard deviation
    if std < target_std:
        stretch_factor = target_std / std if std > 0 else 2.0
        mean_value = np.mean(scores)
        
        for user_id in user_sentiment_scores:
            score = user_sentiment_scores[user_id]
            # Stretch distance from center point
            diff = score - mean_value
            stretched_score = mean_value + diff * stretch_factor
            # Apply stretching but keep within 0-25 range
            user_sentiment_scores[user_id] = np.clip(stretched_score, 0, 25)
    
    # Boost overall scores
    if mean < 10:  # If mean is below 10
        boost_factor = min(1.4, 10 / mean if mean > 0 else 1.2)  # Boost factor, but not exceeding 1.4
        print(f"  - Boosting overall scores, boost factor: {boost_factor:.2f}")
        
        for user_id in user_sentiment_scores:
            user_sentiment_scores[user_id] = min(25, user_sentiment_scores[user_id] * boost_factor)
    
    # Final score statistics
    corrected_scores = np.array(list(user_sentiment_scores.values()))
    corrected_mean = np.mean(corrected_scores)
    corrected_std = np.std(corrected_scores)
    print(f"  - Final sentiment scores: mean={corrected_mean:.2f}, std={corrected_std:.2f}")
    
    # Output distribution
    bins = [0, 5, 10, 15, 20, 25]
    hist, _ = np.histogram(corrected_scores, bins=bins)
    print("  - Score distribution:")
    for i in range(len(bins)-1):
        print(f"    {bins[i]}-{bins[i+1]}: {hist[i]} people")
    
    # 6. Update dim3_score's score3c_sentiment column
    for user_id, score in user_sentiment_scores.items():
        idx = dim3_score[dim3_score['user_id'] == user_id].index
        if len(idx) > 0:
            dim3_score.loc[idx, 'score3c_sentiment'] = score
    
    return dim3_score

def calculate_keyword_score(df, dim3_score, core_keywords):
    """
    Calculate keyword coverage scores - Sigmoid enhanced version
    1. Support substring matching
    2. Use enhanced Sigmoid function mapping to ensure scores fully utilize 0-25 range
    3. Apply more aggressive parameter adjustments
    
    Args:
        df (pd.DataFrame): Input data frame with note information
        dim3_score (pd.DataFrame): User-level scoring DataFrame to be updated
        core_keywords (set): Set of core keywords
    
    Returns:
        pd.DataFrame: Updated user-level scoring DataFrame
    """
    print("  - Calculating keyword coverage scores...")

    # 1. Create efficient keyword matching pattern (pre-compile once)
    print("  - Creating keyword matching pattern...")
    pattern = re.compile('|'.join(re.escape(kw) for kw in core_keywords), 
                        flags=re.IGNORECASE)

    # 2. Calculate keyword count
    print("  - Calculating note-level keyword coverage...")
    df['kw_count'] = df['semantic_proc_text'].fillna("").apply(
        lambda x: len(pattern.findall(str(x)))
    )

    # 3. Apply time decay coefficient
    print("  - Applying time decay coefficient...")
    decay = np.exp(-0.05 * df['elapsed_time'] / 7)

    # 4. Calculate single note scores
    df['single_kw_score'] = np.maximum(1, df['kw_count'] / 5) * decay

    # 5. Aggregate keyword scores by creator - don't use 90th percentile
    print("  - Aggregating user-level keyword scores...")
    user_keyword_scores = {}
    raw_ratios = {}  # Store raw ratios for diagnostics

    for user_id, group in df.groupby('user_id'):
        # Calculate coverage ratio
        cover_rate = (group['kw_count'] > 0).mean()
        raw_ratios[user_id] = cover_rate  # Save raw coverage rate
        
        # Calculate average score and normalize
        avg_score = group['single_kw_score'].mean()
        
        # Combine metrics based on document proportions, coverage rate baseline default 0.05
        combined_raw = 0.7 * avg_score + 0.3 * (cover_rate / 0.05)
        
        # Use sigmoid function mapping to 0-25 range, consistent with originality calculation
        # Here we adjust combined_raw to put it in a reasonable range
        normalized_score = combined_raw / 2 - 0.2  # Normalize to reasonable range
        
        # Apply sigmoid function - adjust parameters for more dispersed distribution
        slope = 5
        midpoint = 0.3  # Adjust midpoint for more even score distribution
        
        # Sigmoid mapping to 0-25 range
        sigmoid_score = 25 / (1 + np.exp(-slope * (normalized_score - midpoint)))
        
        # Store sigmoid transformed score
        user_keyword_scores[user_id] = sigmoid_score

    # Output raw score statistics
    scores = np.array(list(user_keyword_scores.values()))
    mean = np.mean(scores)
    std = np.std(scores)
    
    print(f"  - Sigmoid-transformed scores: mean={mean:.2f}, std={std:.2f}")
    
    # Special handling: Set users with 0 coverage to 0 points
    for user_id, ratio in raw_ratios.items():
        if ratio == 0:
            user_keyword_scores[user_id] = 0
    
    # Handle extreme cases: Prevent too many extremely high or low scores
    high_scores = sum(1 for s in scores if s > 23)
    low_scores = sum(1 for s in scores if s > 0 and s < 2)
    
    # If too many high or low scores, adjust distribution
    if high_scores > len(scores) * 0.15 or low_scores > len(scores) * 0.15:
        print("  - Detected too many extreme scores, applying mild distribution correction...")
        
        for user_id in user_keyword_scores:
            score = user_keyword_scores[user_id]
            # Only process non-zero scores
            if score > 0:
                # Soften high scores
                if score > 23:
                    user_keyword_scores[user_id] = 23 + (score - 23) * 0.5
                # Boost low scores
                elif score < 2:
                    user_keyword_scores[user_id] = 2 * (score / 2)
    
    # Final score statistics
    corrected_scores = np.array(list(user_keyword_scores.values()))
    corrected_mean = np.mean(corrected_scores)
    corrected_std = np.std(corrected_scores)
    print(f"  - Final keyword scores: mean={corrected_mean:.2f}, std={corrected_std:.2f}")
    
    # Output distribution
    bins = [0, 5, 10, 15, 20, 25]
    hist, _ = np.histogram(corrected_scores, bins=bins)
    print("  - Score distribution:")
    for i in range(len(bins)-1):
        print(f"    {bins[i]}-{bins[i+1]}: {hist[i]} people")

    # 7. Update dim3_score's keyword_score column
    for user_id, score in user_keyword_scores.items():
        idx = dim3_score[dim3_score['user_id'] == user_id].index
        if len(idx) > 0:
            dim3_score.loc[idx, 'score3d_keyword'] = score

    # 8. Clean up temporary columns
    if 'kw_count' in df.columns:
        df.drop(columns=['kw_count'], inplace=True)
    if 'single_kw_score' in df.columns:
        df.drop(columns=['single_kw_score'], inplace=True)

    return dim3_score

def create_plots(dim3_score, output_dir='.'):
    """Create and save distribution plots for all scores"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot originality score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(dim3_score['score3a_originality'], bins=20, kde=True, color='blue', stat='count')
    plt.title('Distribution of Creator Originality Scores')
    plt.xlabel('Originality Score')
    plt.ylabel('Count')
    plt.xlim(0, 25)
    plt.xticks(np.arange(0, 26, 5))
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dim3_originality_score_distribution.png'))
    plt.close()
    
    # Plot vertical domain score distribution
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
    
    # Plot sentiment score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(dim3_score['score3c_sentiment'], bins=30, kde=True, color='orange', stat='count')
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.xlim(0, 25)
    plt.xticks(np.arange(0, 26, 5))
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dim3_sentiment_score_distribution.png'))
    plt.close()
    
    # Plot keyword score distribution
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
    
    # Plot total content quality score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(dim3_score['score3_content_quality'], bins=25, kde=True, color='purple', stat='count')
    plt.title('Distribution of Content Quality Scores')
    plt.xlabel('Content Quality Score')
    plt.ylabel('Count')
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 5))
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dim3_content_quality_score_distribution.png'))
    plt.close()
    
    print(f"All plots saved to {output_dir}")

def save_results(dim3_score, output_file='dim3_score_nested.json'):
    """
    Save results to a nested JSON file
    """
    # Create the array structure with user_id as a field inside each object
    nested_data = []
    for _, row in dim3_score.iterrows():
        user_id = row['user_id']
        user_data = {
            "user_id": user_id,  # Include user_id as a field inside the object
            "score3_content_quality": float(row['score3_content_quality']),
            "component_scores": {
                "score3a_originality": float(row['score3a_originality']),
                "score3b_vertical": float(row['score3b_vertical']),
                "score3c_sentiment": float(row['score3c_sentiment']),
                "score3d_keyword": float(row['score3d_keyword'])
            }
        }
        nested_data.append(user_data)

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(nested_data, f, ensure_ascii=False, indent=4)

    print(f"Nested JSON file saved to {output_file}")
    
    # Show the first and last item of the nested json array
    print("First item:", nested_data[0])
    print("Last item:", nested_data[-1])
    
    return nested_data

def main(input_file, output_dir='.'):
    """
    Main function to run the content scoring process
    
    Args:
        input_file (str): Path to the input JSON file
        output_dir (str): Directory to save output files
    """
    print("Starting content scoring process...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    print(f"Loading data from {input_file}...")
    conts = pd.read_json(input_file, lines=True)
    
    # 2. Load models
    print("Loading models...")
    models = load_models()
    
    # 3. Preprocess data
    print("Preprocessing data...")
    df, user_notes, dim3_score = preprocess_data(conts)
    
    # 4. Calculate creator originality scores
    print("Calculating creator originality scores...")
    dim3_score = calculate_creator_originality(df, dim3_score, models)
    
    # 5. Calculate vertical domain scores
    print("Calculating vertical domain scores...")
    dim3_score = calculate_vertical_score(df, dim3_score)
    
    # 6. Calculate sentiment scores
    print("Calculating sentiment scores...")
    dim3_score = calculate_sentiment_score(df, dim3_score, models, max_notes_per_user=1000)
    
    # 7. Calculate keyword scores
    print("Calculating keyword scores...")
    dim3_score = calculate_keyword_score(df, dim3_score, CORE_KEYWORDS)
    
    # 8. Calculate total content quality score
    print("Calculating total content quality scores...")
    dim3_score['score3_content_quality'] = dim3_score[[
        'score3a_originality', 'score3b_vertical', 
        'score3c_sentiment', 'score3d_keyword'
    ]].sum(axis=1)
    
    # 9. Create and save plots
    print("Creating score distribution plots...")
    create_plots(dim3_score, output_dir)
    
    # 10. Save results
    output_file = os.path.join(output_dir, 'dim3_score_nested.json')
    print(f"Saving results to {output_file}...")
    nested_data = save_results(dim3_score, output_file)
    
    # 11. Print final statistics
    print("\nFinal score statistics:")
    print(dim3_score['score3_content_quality'].describe())
    
    print("\nContent scoring process completed successfully!")
    
    return dim3_score, nested_data

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Content Quality Scoring')
    parser.add_argument('--input', type=str, default='../../Data/processed/contents_cooked_semantic.json',
                        help='Path to the input JSON file')
    parser.add_argument('--output', type=str, default='./output',
                        help='Directory to save output files')
    args = parser.parse_args()
    
    # Run the main function
    main(args.input, args.output)