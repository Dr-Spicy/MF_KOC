import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.ndimage import gaussian_filter1d

# Set Chinese font manually (if needed)
plt.rcParams["font.family"] = "SimHei"  # Use "Microsoft YaHei" if SimHei doesn't work
plt.rcParams["axes.unicode_minus"] = False  # Avoids issues with negative signs

# File paths
content_file = "content.json"
creator_file = "creator.json"
output_folder = "daily_interaction_plots"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Read JSON data
with open(content_file, "r", encoding="utf-8") as f:
    content_data = json.load(f)

with open(creator_file, "r", encoding="utf-8") as f:
    creator_data = json.load(f)

# Convert to DataFrame
content_df = pd.DataFrame(content_data)
creator_df = pd.DataFrame(creator_data)

# Convert time column to datetime
content_df["time"] = pd.to_datetime(content_df["time"], unit="ms")

# Ensure numeric interaction counts
for col in ["liked_count", "collected_count", "comment_count", "share_count"]:
    content_df[col] = content_df[col].astype(int)

# Automatically set a Chinese-compatible font
def set_chinese_font():
    font_candidates = ["SimHei", "SimSun", "STSong", "Noto Sans CJK SC"]
    for font in font_candidates:
        if font in fm.findSystemFonts():
            plt.rcParams["font.family"] = font
            print(f"Using font: {font}")
            return
    print("Warning: No Chinese font found! Some characters may not display correctly.")

set_chinese_font()

# Create mapping of user_id to nickname
user_nickname_map = dict(zip(creator_df["user_id"], creator_df["nickname"]))

# Get all unique user IDs
all_users = content_df["user_id"].unique()

# Set time window
time_window_days = 180

# Iterate over **ALL users**
for user_id in all_users:
    user_data = content_df[content_df["user_id"] == user_id]

    # Get nickname or fallback to "Unknown"
    nickname = user_nickname_map.get(user_id, "Unknown")

    # Filter data within last X days
    latest_time = user_data["time"].max()
    filtered_data = user_data[user_data["time"] >= latest_time - pd.Timedelta(days=time_window_days)]

    # **🔹 Aggregate by Weekly Bins**
    filtered_data["time_bin"] = filtered_data["time"].dt.to_period("D").dt.to_timestamp()

    weekly_counts = filtered_data.groupby("time_bin")[["liked_count", "collected_count", "comment_count", "share_count"]].sum()

    # Compute total interaction per time bin
    weekly_counts["total_interactions"] = weekly_counts.sum(axis=1)

    # If no interactions, skip the user
    if weekly_counts.empty:
        continue

    # Apply Gaussian smoothing to the interaction trend
    smoothed_interactions = gaussian_filter1d(weekly_counts["total_interactions"], sigma=3)

    # Plot Weekly Interaction Trends (Stacked Bar Chart)
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Stacked bar chart components
    ax1.bar(weekly_counts.index, weekly_counts["liked_count"], color="red", label="LikedCount", width=5)
    ax1.bar(weekly_counts.index, weekly_counts["collected_count"], bottom=weekly_counts["liked_count"], color="yellow", label="CollectedCount", width=5)
    ax1.bar(weekly_counts.index, weekly_counts["comment_count"], 
           bottom=weekly_counts["liked_count"] + weekly_counts["collected_count"], color="green", label="CommentCount", width=5)
    ax1.bar(weekly_counts.index, weekly_counts["share_count"], 
           bottom=weekly_counts["liked_count"] + weekly_counts["collected_count"] + weekly_counts["comment_count"], color="purple", label="ShareCount", width=5)

    # **Add smooth line on top**
    ax2 = ax1.twinx()
    ax2.plot(weekly_counts.index, smoothed_interactions, linestyle='-', color="black", linewidth=0.8, label="Smoothed Total Interactions")

    # Set Labels
    ax1.set_xlabel("Time ")
    ax1.set_ylabel("Total Interactions")
    ax1.set_title(f"Daily Interaction Breakdown\nUser: {nickname} (Last {time_window_days} Days)")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Save the Plot with a Safe Filename
    safe_nickname = nickname.encode("utf-8", "ignore").decode("utf-8")  # Handle Chinese characters
    plt.savefig(f"{output_folder}/weekly_interactions_{safe_nickname}.png", bbox_inches="tight")
    plt.close()

    print(f"Weekly interaction plot saved in {output_folder} for user {nickname}")

print("All weekly interaction plots have been generated!")
