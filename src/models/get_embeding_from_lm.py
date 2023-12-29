from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model_name = "cointegrated/LaBSE-en-ru"
    model = SentenceTransformer(model_name)
    sentences = ["привет мир", "hello world", "здравствуй вселенная"]
    embeddings = model.encode(sentences)
    for embedding in embeddings:
        print(type(embedding), len(embedding))
