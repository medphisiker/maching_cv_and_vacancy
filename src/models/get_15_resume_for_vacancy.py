import faiss
import pandas as pd
from faiss_utils import get_embedding_by_id

if __name__ == "__main__":
    resume_csv_path = "./data/processed/dst-3.0_16_1_hh_database_prep.csv"
    vacancy_csv_path = "./data/processed/hhparser_vacancy_prep.csv"
    resume_index_path = "src/data/resume_db.index"
    vacancy_index_path = "src/data/vacancy_db.index"
    vacancy_id = 5
    topn = 10

    # считываем faiss-индексы
    resume_index = faiss.read_index(resume_index_path)
    vacancy_index = faiss.read_index(vacancy_index_path)

    # считываем базы резюме
    df_resume = pd.read_csv(resume_csv_path)
    df_vacancy = pd.read_csv(vacancy_csv_path)

    vacancy_embed = get_embedding_by_id(vacancy_index, vacancy_id)
    _, resume_ids = resume_index.search(vacancy_embed, topn)

    print("Для вакансии")
    print(df_vacancy.iloc[vacancy_id])

    print(f"Мы подобрали {topn} резюме:")
    print(df_vacancy.iloc[resume_ids[0]])
