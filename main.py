from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load dataset (Ensure your CSV file has a 'recipe' column)
df = pd.read_csv("indian_food.csv")

# Convert ingredients to lowercase text
df["ingredients"] = df["ingredients"].astype(str).str.lower()

# Ensure 'recipe' column exists and fill NaN values with a placeholder
if "recipe" not in df.columns:
    df["recipe"] = "Recipe not available."
else:
    df["recipe"] = df["recipe"].fillna("Recipe not available.")

# Vectorization
vectorizer = TfidfVectorizer()
ingredient_matrix = vectorizer.fit_transform(df["ingredients"])

# General chat responses
general_responses = {
    "hello": "Hey there! What dish are you craving today?",
    "hi": "Hello! Want some delicious food suggestions?",
    "hii": "Hello! Want some delicious food suggestions?",  # Added "hii"
    "hey": "Hey! How can I assist you today?",  # Added "hey"
    "how are you": "I'm just a bot, but I'm ready to suggest some tasty recipes!",
    "what's up": "Not much, just thinking about food. How about you?",
    "who are you": "I'm your personal food assistant! Ask me for recipes or ingredient-based dish suggestions."
}

@app.get("/recommend/")
def recommend(ingredients: str):
    user_input = ingredients.lower().strip()

    # General chat detection
    if user_input in general_responses:
        return {"message": general_responses[user_input]}

    # Recipe suggestion
    input_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(input_vector, ingredient_matrix)
    top_indices = similarity.argsort()[0][-5:][::-1]

    result = df.iloc[top_indices][["name", "ingredients", "diet", "course", "state", "recipe"]].to_dict(orient="records")
    return {"recommendations": result}

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
