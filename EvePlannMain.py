import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and clean data
df = pd.read_excel("Database_Edited.xlsx")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Drop rows missing key values
df.dropna(subset=["place_available", "place_location", "price"], inplace=True)

# Combine all tags into one string
df["tags_combined"] = df[[f"tag_{i}" for i in range(1, 9)]].fillna("").agg(" ".join, axis=1)

# User input
username = input("Enter your username/User_id: ")
start_time_str = input("Enter your start time (e.g., 14:00): ")
end_time_str = input("Enter your end time (e.g., 18:00): ")

# Time conversion
start_time = datetime.strptime(start_time_str, "%H:%M")
end_time = datetime.strptime(end_time_str, "%H:%M")

# Search history
user_history = df[df["user_id"] == username]["search_history"]
if user_history.empty:
    print("⚠️ No search history found. Using default keywords.")
    user_profile = "dog cafe fun gaming event pet animals music"
else:
    user_profile = " ".join(user_history.astype(str).tolist())

# TF-IDF Similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["tags_combined"].tolist() + [user_profile])
similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
df["similarity"] = similarities

# Sort by similarity
sorted_df = df.sort_values(by="similarity", ascending=False).copy()
selected_tags = set()
recommendations = []
current_time = start_time

# Plan based on available time
for _, row in sorted_df.iterrows():
    place_tags = set(row["tags_combined"].split())
    if place_tags & selected_tags:
        continue  # skip if overlapping tags

    travel_time = random.randint(10, 25)
    activity_time = random.choice([60, 90])  # in minutes
    total_time = travel_time + activity_time

    if current_time + timedelta(minutes=total_time) > end_time:
        continue

    # Add to recommendations
    activity_end = current_time + timedelta(minutes=activity_time)
    recommendations.append({
        "place": row["place_available"],
        "location": row["place_location"],
        "tags": row["tags_combined"],
        "price": row["price"],
        "travel_time": travel_time,
        "time_slot": f"{current_time.strftime('%H:%M')} - {activity_end.strftime('%H:%M')}"
    })

    # Move time forward
    current_time = activity_end + timedelta(minutes=travel_time)
    selected_tags.update(place_tags)

# Show results
if not recommendations:
    print("⚠️ Only found 0 recommendation(s).")
else:
    print(f"\n✅ Showing {len(recommendations)} recommendation(s):")
    for rec in recommendations:
        print(f"\n📍 {rec['place']} — {rec['location']}")
        print(f"🏷️ Tags: {rec['tags']}")
        print(f"💸 Price: ₹{rec['price']}")
        print(f"🚕 Travel: {rec['travel_time']} mins")
        print(f"🕒 Time Slot: {rec['time_slot']}")
