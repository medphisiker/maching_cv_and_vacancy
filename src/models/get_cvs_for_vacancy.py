import os
import random

import faiss
import pandas as pd
from faiss_utils import search_elems, create_dir


if __name__ == "__main__":
    resume_csv_path = "data/processed/dst-3.0_16_1_hh_database_prep.csv"
    vacancy_csv_path = "data/processed/hhparser_vacancy_prep.csv"
    model_title = "LaBSE-en-ru"
    resume_index_path = f"data/faiss_index/{model_title}/resume_db.index"
    vacancy_index_path = f"data/faiss_index/{model_title}/vacancy_db.index"
    topn = 10
    vacancies_number = 10
    output_dir = f"data/analysis/{model_title}/cvs_for_vacancy"

    random.seed(42)

    # считываем faiss-индексы
    resume_index = faiss.read_index(resume_index_path)
    vacancy_index = faiss.read_index(vacancy_index_path)

    # считываем базы резюме
    df_resume = pd.read_csv(resume_csv_path)
    df_vacancy = pd.read_csv(vacancy_csv_path)

    create_dir(output_dir)

    for i in range(vacancies_number):
        vacancy_id = random.randint(0, len(df_vacancy))

        df_vacancy_id, cvs_for_vacancy_id = search_elems(
            vacancy_id,
            topn,
            vacancy_index,
            df_vacancy,
            resume_index,
            df_resume,
        )

        # для вакансии
        vacancy_path = f"vacancy_{vacancy_id}.csv"
        vacancy_path = os.path.join(output_dir, vacancy_path)
        df_vacancy_id.to_csv(vacancy_path)

        # мы подобрали topn резюме
        cvs_for_vacancy_path = f"vacancy_{vacancy_id}_{topn}_resumes.csv"
        cvs_for_vacancy_path = os.path.join(output_dir, cvs_for_vacancy_path)
        cvs_for_vacancy_id.to_csv(cvs_for_vacancy_path)
