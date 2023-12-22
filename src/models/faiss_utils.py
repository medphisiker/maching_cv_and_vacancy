import pandas as pd
import numpy as np


def get_text_from_df(dataset_path, text_columns):
    df_vacancy = pd.read_csv(dataset_path)[text_columns]
    df_vacancy = df_vacancy[text_columns].astype(str).agg(" ".join, axis=1)
    vacancies = df_vacancy.to_list()
    return vacancies


def encode_text_by_batches(model, sentences, batch_size):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)

    embeddings = np.array(embeddings)

    return embeddings


def get_embedding_by_id(index, vec_id):
    # https://github.com/facebookresearch/faiss/issues/1068#issuecomment-1740979811
    # Create an empty numpy array to hold the vector
    vec = np.empty((1, index.d), dtype="float32")

    # Call the reconstruct method
    index.reconstruct(vec_id, vec[0])

    return vec
