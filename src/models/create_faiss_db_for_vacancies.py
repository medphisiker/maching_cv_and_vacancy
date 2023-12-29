import os

from faiss_utils import create_faiss_index

if __name__ == "__main__":
    # конфигурируем хранилище
    data_storage_path = "data/faiss_index"
    vacancy_db_name = "vacancy_db.index"

    # считываем описания вакансий из интересующих нас столбцов
    vacancy_csv_path = "./data/processed/hhparser_vacancy_prep.csv"
    info_columns = ["name", "description"]

    # указываем модель, с которой будем получать эмбединги
    model_name = "cointegrated/LaBSE-en-ru"
    model_title = "LaBSE-en-ru"

    data_storage_path = os.path.join(data_storage_path, model_title, vacancy_db_name)

    # получаем faiss индекс
    create_faiss_index(vacancy_csv_path, info_columns, model_name, data_storage_path)
