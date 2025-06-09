import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load Excel data
df = pd.read_excel("Database_Edited.xlsx")

# Clean and filter data
user_df = df.dropna(subset=['User_id', 'Search_History', 'User_Location'])
place_df = df[['Place_Available']].drop_duplicates()
place_full_df = df[['Place_Available', 'Place_Location', 'Price']].drop_duplicates()

# Combine textual info for vectorization
place_full_df["combined_info"] = (
    place_full_df["Place_Available"] + " " +
    place_full_df["Place_Location"] + " " +
    place_full_df["Price"].astype(str)
)

# Build user search profiles and locations
user_profiles = user_df.groupby("User_id")["Search_History"].apply(lambda x: ' '.join(x)).to_dict()
user_locations = user_df.groupby("User_id")["User_Location"].first().to_dict()

# Cache dictionary to remember last query per user
user_last_query_cache = {}

# Recommendation function
def recommend_places_by_username(username, top_n=12):
    if username not in user_profiles or username not in user_locations:
        return place_df.sample(n=top_n)["Place_Available"].str.strip().tolist()

    user_query = user_profiles[username]
    user_location = user_locations[username].strip().lower()

    # Check if the same user and search history
    if username in user_last_query_cache:
        cached_query, cached_results = user_last_query_cache[username]
        if cached_query == user_query:
            shuffled = cached_results[:]  # copy
            random.shuffle(shuffled)
            return shuffled

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    place_vectors = vectorizer.fit_transform(place_full_df["combined_info"])
    user_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vector, place_vectors).flatten()
    place_full_df['similarity'] = similarities

    sorted_df = place_full_df.sort_values(by='similarity', ascending=False)

    best_matches = {}
    for _, row in sorted_df.iterrows():
        name = row['Place_Available'].strip()
        location = row['Place_Location'].strip().lower()

        if name not in best_matches:
            if location == user_location:
                best_matches[name] = row
            elif name not in best_matches:
                best_matches[name] = row

        if len(best_matches) >= top_n:
            break

    results = [entry['Place_Available'].strip() for entry in best_matches.values()]

    # Save to cache
    user_last_query_cache[username] = (user_query, results)

    return results

# Input and output
username_input = input("Enter your username: ")
recommendations = recommend_places_by_username(username_input, top_n=12)
random.shuffle(recommendations)

print("\nRecommended Places:")
for place in recommendations:
    print("-", place)
