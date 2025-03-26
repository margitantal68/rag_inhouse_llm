import numpy as np
import ollama
import requests
from scipy.spatial.distance import cosine

def get_embedding_inhouse_model(text: str, model: str = "nomic-embed-text"):
    host = "http://192.168.11.102:11500"
    response = requests.post(f"{host}/api/embeddings", json={"model": model, "prompt": text})
    response_data = response.json()
    return response_data["embedding"]

def cosine_distance(embedding1, embedding2):
    return cosine(embedding1, embedding2)

if __name__ == "__main__":
    # text1 = "Artificial intelligence is transforming the world."
    # text2 = "Machine learning is a subset of artificial intelligence."
    # text1 = "I like to eat apples."
    # text2 = "I like to eat pears." 
    # text3 = "There is a book on the table." 
    # text4 = "There is a book on the chair."

    text1 = "Szeretek almát enni."
    text2 = "Szeretek körtét enni."

    text3 = "Van egy könyv az asztalon." 
    text4 = "van egy könyv a széken."
    
    embedding1 = get_embedding_inhouse_model(text1)
    embedding2 = get_embedding_inhouse_model(text2)
    embedding3 = get_embedding_inhouse_model(text3)
    embedding4 = get_embedding_inhouse_model(text4)
    
    distance12 = cosine_distance(embedding1, embedding2)
    distance13 = cosine_distance(embedding1, embedding3)
    distance34 = cosine_distance(embedding3, embedding4)

    print(f"LOCAL MODEL - Cosine distance between <{text1}> and <{text2}> embeddings: {distance12:.4f}")
    print(f"LOCAL MODEL - Cosine distance between <{text1}> and <{text3}> embeddings: {distance13:.4f}")
    print(f"LOCAL MODEL - Cosine distance between <{text3}> and <{text4}> embeddings: {distance34:.4f}")

