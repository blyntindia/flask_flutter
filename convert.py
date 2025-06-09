from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)  # ✅ Standard practice, avoid using same name as file

# ✅ Load and prepare data once
df = pd.read_excel("Book1.xlsx")

user_df = df.dropna(subset=['User_id', 'Search_History', 'User_Location'])
place_df = df[['Place_Available']].drop_duplicates()
place_full_df = df[['Place_Available', 'Place_Location', 'Price']].drop_duplicates()

place_full_df["combined_info"] = (
    place_full_df["Place_Available"].astype(str) + " " +
    place_full_df["Place_Location"].astype(str) + " " +
    place_full_df["Price"].astype(str)
)

vectorizer = TfidfVectorizer()
place_vectors = vectorizer.fit_transform(place_full_df["combined_info"])

user_profiles = user_df.groupby("User_id")["Search_History"].apply(lambda x: ' '.join(x)).to_dict()
user_locations = user_df.groupby("User_id")["User_Location"].first().to_dict()

# 🧠 Recommendation function
def recommend_places_by_username(username, top_n=20):
    if username not in user_profiles or username not in user_locations:
        return place_df.sample(n=top_n)["Place_Available"].str.strip().tolist()

    user_query = user_profiles[username]
    user_location = user_locations[username].strip().lower()
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

    return [entry['Place_Available'].strip() for entry in best_matches.values()]

# 🔗 API Endpoint
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True)
    username = data.get("username", "")
    recommendations = recommend_places_by_username(username)
    return jsonify({"recommendations": recommendations})

# 🚀 Start the server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
