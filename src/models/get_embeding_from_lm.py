from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model = SentenceTransformer("cointegrated/rubert-tiny2")
    sentences = ["привет мир", "hello world", "здравствуй вселенная"]
    embeddings = model.encode(sentences)
    for embedding in embeddings:
        print(type(embedding), len(embedding))
