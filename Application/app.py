import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
import joblib

app = Flask(__name__)

# Load the TF-IDF vectorizer, recipe data, and the TF-IDF matrix
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")  # Load the saved TF-IDF matrix
df = joblib.load("recipe_data.pkl")


def recommend_recipes(input_ingredients, num_recommendations=5):
    input_ingredients_str = " ".join(input_ingredients)
    input_tfidf = tfidf_vectorizer.transform([input_ingredients_str])
    cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix)

    similarity_scores = list(enumerate(cosine_similarities[0]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_indices = [i[0] for i in similarity_scores[1 : num_recommendations + 1]]
    return df.iloc[recommended_indices][["recipe_name", "recipe_urls"]]


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None  # Initialize as None
    if request.method == "POST":
        input_ingredients = request.form["ingredients"].split(",")
        input_ingredients = [ingredient.strip() for ingredient in input_ingredients]
        recommendations = recommend_recipes(input_ingredients)

    return render_template("index.html", recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)
