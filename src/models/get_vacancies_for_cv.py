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
    cvs_number = 10
    output_dir = f"data/analysis/{model_title}/vacancies_for_cv"

    random.seed(42)

    # считываем faiss-индексы
    resume_index = faiss.read_index(resume_index_path)
    vacancy_index = faiss.read_index(vacancy_index_path)

    # считываем базы резюме
    df_resume = pd.read_csv(resume_csv_path)
    df_vacancy = pd.read_csv(vacancy_csv_path)

    create_dir(output_dir)

    for i in range(cvs_number):
        resume_id = random.randint(0, len(df_resume))

        df_resume_id, cvs_for_vacancy_id = search_elems(
            resume_id, topn, resume_index, df_resume, vacancy_index, df_vacancy
        )

        # для резюме
        resume_path = f"resume_{resume_id}.csv"
        resume_path = os.path.join(output_dir, resume_path)
        df_resume_id.to_csv(resume_path)

        # мы подобрали topn резюме
        vacancies_for_cvs_path = f"resume_{resume_id}_{topn}_vacancies.csv"
        vacancies_for_cvs_path = os.path.join(output_dir, vacancies_for_cvs_path)
        cvs_for_vacancy_id.to_csv(vacancies_for_cvs_path)
