from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load dataset
df = pd.read_csv("indian_food_with_recipes.csv")

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
    "hii": "Hello! Want some delicious food suggestions?",
    "hey": "Hey! How can I assist you today?",
    "how are you": "I'm just a bot, but I'm ready to suggest some tasty recipes!",
    "what's up": "Not much, just thinking about food. How about you?",
    "who are you": "I'm your personal food assistant! Ask me for recipes or ingredient-based dish suggestions."
}

@app.get("/recommend/")
@app.get("/recommend/")
def recommend(ingredients: str):
    user_input = ingredients.lower().strip()

    # Handle general chat responses
    if user_input in general_responses:
        return {"message": general_responses[user_input]}  # Return general response

    # Ignore very short inputs (e.g., "yes", "ok")
    if len(user_input) <= 3:
        return {"message": "Can you provide some ingredients or a dish name?"}

    # Check if the user input matches a dish name
    matching_recipe = df[df["name"].str.lower() == user_input]
    if not matching_recipe.empty:
        recipe_info = matching_recipe.iloc[0][["name", "ingredients", "diet", "course", "state", "recipe"]].to_dict()
        return {"recommendations": [recipe_info]}  # Wrap in an array to match Android format

    # Use TF-IDF for best match if input is a valid ingredient
    input_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(input_vector, ingredient_matrix)

    # Prevent random selections for irrelevant input
    top_index = similarity.argsort()[0][-1]
    if similarity[0][top_index] < 0.1:  # If similarity is too low, avoid giving wrong results
        return {"message": "I couldn't find a matching recipe. Please try with different ingredients!"}

    result = df.iloc[top_index][["name", "ingredients", "diet", "course", "state", "recipe"]].to_dict()
    return {"recommendations": [result]}  # Wrap in an array
@app.get("/suggested")
def get_suggested_recipes():
    # Return first 5 recipes as suggestions
    suggestions = df.head(5)[["name", "ingredients", "diet", "course", "state", "recipe"]].to_dict(orient="records")
    return suggestions


# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
