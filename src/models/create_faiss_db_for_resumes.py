import os

import faiss
from faiss_utils import encode_text_by_batches, get_text_from_df
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    # конфигурируем хранилище
    data_storage_path = "src/data"
    resume_db_name = "resume_db.index"

    # считываем описания вакансий из интересующих нас столбцов
    resume_csv_path = "./data/processed/dst-3.0_16_1_hh_database_prep.csv"
    info_columns = ["Ищет работу на должность:", "Опыт работы"]

    # указываем модель, с которой будем получать эмбединги
    model_name = "cointegrated/rubert-tiny2"

    model = SentenceTransformer(model_name)

    # получаем базу для вакансий
    resumes = get_text_from_df(resume_csv_path, info_columns)
    resumes_vectors = encode_text_by_batches(model, resumes, batch_size=256)
    dim = resumes_vectors.shape[1]

    resume_index = faiss.IndexFlatL2(dim)
    resume_index.add(resumes_vectors)

    resume_db_path = os.path.join(data_storage_path, resume_db_name)
    faiss.write_index(resume_index, resume_db_path)
