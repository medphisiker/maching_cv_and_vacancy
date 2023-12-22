import os

import faiss
from faiss_utils import encode_text_by_batches, get_text_from_df
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    # конфигурируем хранилище
    data_storage_path = "src/data"
    vacancy_db_name = "vacancy_db.index"

    # считываем описания вакансий из интересующих нас столбцов
    vacancy_csv_path = "./data/processed/hhparser_vacancy_prep.csv"
    info_columns = ["name", "description"]

    # указываем модель, с которой будем получать эмбединги
    model_name = "cointegrated/rubert-tiny2"

    model = SentenceTransformer(model_name)

    # получаем базу для вакансий
    vacancies = get_text_from_df(vacancy_csv_path, info_columns)
    vacancy_vectors = encode_text_by_batches(model, vacancies, batch_size=256)
    dim = vacancy_vectors.shape[1]

    vacancy_index = faiss.IndexFlatL2(dim)
    vacancy_index.add(vacancy_vectors)

    vacancy_db_path = os.path.join(data_storage_path, vacancy_db_name)
    faiss.write_index(vacancy_index, vacancy_db_path)