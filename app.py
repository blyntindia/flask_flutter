from flask import Flask, request, jsonify
import pandas as pd
import random

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "Flask Recommendation API is running!"

try:
    df = pd.read_excel("Database_Edited.xlsx")
except Exception as e:
    print("❌ Error reading Excel file:", e)
    df = pd.DataFrame()  # fallback to avoid crashing

if not df.empty:
    try:
        user_df = df.dropna(subset=['User_id', 'Search_History', 'User_Location'])
        place_df = df[['Place_Available']].drop_duplicates()
        place_full_df = df[['Place_Available', 'Place_Location', 'Price']].drop_duplicates()

        place_full_df["combined_info"] = (
            place_full_df["Place_Available"] + " " +
            place_full_df["Place_Location"] + " " +
            place_full_df["Price"].astype(str)
        )

        user_profiles = user_df.groupby("User_id")["Search_History"].apply(lambda x: ' '.join(x)).to_dict()
        user_locations = user_df.groupby("User_id")["User_Location"].first().to_dict()
    except Exception as e:
        print("❌ Error processing data:", e)
        user_profiles = {}
        user_locations = {}
        place_df = pd.DataFrame()
        place_full_df = pd.DataFrame()
else:
    user_profiles = {}
    user_locations = {}
    place_df = pd.DataFrame()
    place_full_df = pd.DataFrame()

user_last_query_cache = {}

@app.route("/recommend", methods=["POST"])
def recommend():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    data = request.get_json()
    username = data.get("username")

    if not username:
        return jsonify({"error": "Username is required"}), 400

    if username not in user_profiles or username not in user_locations:
        if not place_df.empty:
            random_places = place_df.sample(n=12)["Place_Available"].str.strip().tolist()
        else:
            random_places = []
        return jsonify({"recommendations": random_places})

    user_query = user_profiles[username]
    user_location = user_locations[username].strip().lower()

    if username in user_last_query_cache:
        cached_query, cached_results = user_last_query_cache[username]
        if cached_query == user_query:
            shuffled = cached_results[:]
            random.shuffle(shuffled)
            return jsonify({"recommendations": shuffled})

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

        if len(best_matches) >= 12:
            break

    results = [entry['Place_Available'].strip() for entry in best_matches.values()]
    user_last_query_cache[username] = (user_query, results)

    return jsonify({"recommendations": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
