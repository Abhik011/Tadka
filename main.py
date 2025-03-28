from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load dataset
df = pd.read_csv("indian_food.csv")  # Using the provided Indian food dataset

# Convert ingredients to lowercase text
df["ingredients"] = df["ingredients"].apply(lambda x: x.lower())

# Vectorization
vectorizer = TfidfVectorizer()
ingredient_matrix = vectorizer.fit_transform(df["ingredients"])

# Recommendation function
@app.get("/recommend/")
def recommend(ingredients: str):
    input_vector = vectorizer.transform([ingredients.lower()])
    similarity = cosine_similarity(input_vector, ingredient_matrix)
    top_indices = similarity.argsort()[0][-5:][::-1]
    result = df.iloc[top_indices][["name", "ingredients", "diet", "course", "state"]].to_dict(orient="records")
    return {"recommendations": result}

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
