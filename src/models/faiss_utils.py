import functools
import os
import time

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def timer_decorator(func):
    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function {func.__name__!r} executed in {elapsed_time:.4f}s")
        return result

    return wrapper_func


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


def search_elems(
    id_index_1,
    topn,
    faiss_index_1,
    df1,
    faiss_index_2,
    df2,
):
    """Ищет в faiss_index_1 по номеру id_index_1, topn наиболее похожих векторов из
    faiss_index_2.

    Parameters
    ----------
    id_index_1 : int
        номер id по которому мы будем доставать вектор из faiss_index_1
    topn : int
        количество наиболее похожих векторов, которые мы будем искать в faiss_index_2
    faiss_index_1 : faiss index
        faiss index из которого возьмем вектор по id_index_1 и для которого
        будем искать похожие вектора
    df1 : pandas.DataFrame
        хранилище данных в виде pandas.DataFrame, для которых был построен
        faiss_index_1
    faiss_index_2 : faiss index
        faiss index в котором будем искать topn наиболее похожих векторов
    df2 : pandas.DataFrame
        хранилище данных в виде pandas.DataFrame, для которых был построен
        faiss_index_2

    Returns
    -------
    df1_elem_id, df2_elems
        df1_elem_id : pandas.DataFrame
            элемент из хранилища df1, для которого мы искали наиболее похожие
            элементы из хранилища df2
        df2_elems : pandas.DataFrame
            наиболее похожие элементы из хранилища df2
    """
    vacancy_embed = get_embedding_by_id(faiss_index_1, id_index_1)
    _, resume_ids = faiss_index_2.search(vacancy_embed, topn)
    df1_elem_id = df1.iloc[id_index_1]
    df2_elems = df2.iloc[resume_ids[0]]

    return df1_elem_id, df2_elems


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


@timer_decorator
def create_faiss_index(df_path, info_columns, model_name, out_index_path):
    model = SentenceTransformer(model_name)

    dir_path = os.path.split(out_index_path)[0]
    create_dir(dir_path)

    texts = get_text_from_df(df_path, info_columns)
    print("Получили тексты")

    text_vectors = encode_text_by_batches(model, texts, batch_size=256)
    print(f"Получили векторы текстов от модели {model_name}")

    dim = text_vectors.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(text_vectors)
    print("Добавили векторы в индекс faiss")

    faiss.write_index(faiss_index, out_index_path)
    print(f"Сохранили faiss индекс:")
    print(f"{out_index_path}")
