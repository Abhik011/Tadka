from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load dataset
df = pd.read_csv("indian_food.csv")  # Ensure this CSV has a 'recipe' column

# Convert ingredients to lowercase
df["ingredients"] = df["ingredients"].apply(lambda x: x.lower())

# Vectorization
vectorizer = TfidfVectorizer()
ingredient_matrix = vectorizer.fit_transform(df["ingredients"])

# List of general chat inputs
general_responses = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hey there! Looking for a delicious recipe?",
    "how are you": "I'm just a food AI, but I'm always ready to help!",
    "what's up": "Not much! Just cooking up some great food ideas for you!"
}

# Recommendation function
@app.get("/recommend/")
def recommend(ingredients: str):
    ingredients = ingredients.lower().strip()
    
    # Check if input is a general chat
    if ingredients in general_responses:
        return {"message": general_responses[ingredients]}
    
    # Otherwise, treat it as an ingredient list
    input_vector = vectorizer.transform([ingredients])
    similarity = cosine_similarity(input_vector, ingredient_matrix)
    top_indices = similarity.argsort()[0][-5:][::-1]

    result = df.iloc[top_indices][["name", "ingredients", "diet", "course", "state", "recipe"]].to_dict(orient="records")
    
    return {"recommendations": result}

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
