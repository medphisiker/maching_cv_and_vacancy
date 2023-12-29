import os

from faiss_utils import create_faiss_index

if __name__ == "__main__":
    # конфигурируем хранилище
    data_storage_path = "data/faiss_index"
    resume_db_name = "resume_db.index"

    # считываем описания вакансий из интересующих нас столбцов
    resume_csv_path = "./data/processed/dst-3.0_16_1_hh_database_prep.csv"
    info_columns = ["Ищет работу на должность:", "Опыт работы"]

    # указываем модель, с которой будем получать эмбединги
    model_name = "cointegrated/LaBSE-en-ru"
    model_title = "LaBSE-en-ru"

    data_storage_path = os.path.join(data_storage_path, model_title, resume_db_name)

    # получаем faiss индекс
    create_faiss_index(resume_csv_path, info_columns, model_name, data_storage_path)
