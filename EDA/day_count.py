import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set Chinese font manually (if needed)
plt.rcParams["font.family"] = "SimHei"  # Use "Microsoft YaHei" if SimHei doesn't work
plt.rcParams["axes.unicode_minus"] = False  # Avoids issues with negative signs

# File paths
content_file = "content.json"
creator_file = "creator.json"
output_folder = "post_interaction_plots"

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

    # Sort posts by time
    filtered_data = filtered_data.sort_values(by="time")

    # If no interactions, skip the user
    if filtered_data.empty:
        continue

    # Plot Interaction Trends Over Time (Stacked Bar Chart)
    fig, ax = plt.subplots(figsize=(12, 5))

    # Stacked bar chart components
    ax.bar(filtered_data["time"], filtered_data["liked_count"], color="red", label="LikedCount")
    ax.bar(filtered_data["time"], filtered_data["collected_count"], bottom=filtered_data["liked_count"], color="blue", label="CollectedCount")
    ax.bar(filtered_data["time"], filtered_data["comment_count"], 
           bottom=filtered_data["liked_count"] + filtered_data["collected_count"], color="green", label="CommentCount")
    ax.bar(filtered_data["time"], filtered_data["share_count"], 
           bottom=filtered_data["liked_count"] + filtered_data["collected_count"] + filtered_data["comment_count"], color="purple", label="ShareCount")

    # Set Labels
    ax.set_xlabel("Time")
    ax.set_ylabel("Total Interactions Per Post")
    ax.set_title(f"Post Interaction Breakdown Over Time\nUser: {nickname} (Last {time_window_days} Days)")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Legend
    ax.legend()

    # Save the Plot with a Safe Filename
    safe_nickname = nickname.encode("utf-8", "ignore").decode("utf-8")  # Handle Chinese characters
    plt.savefig(f"{output_folder}/post_interactions_{safe_nickname}.png", bbox_inches="tight")
    plt.close()

    print(f"Post interaction plot saved in {output_folder} for user {nickname}")

print("All post interaction plots have been generated!")
