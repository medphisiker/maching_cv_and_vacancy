import os

import faiss
import numpy as np
import pandas as pd
import streamlit as st

# выполняем один раз при инициализации веб страницы
resume_csv_path = "data/dst-3.0_16_1_hh_database_prep.csv"
vacancy_csv_path = "data/hhparser_vacancy_prep.csv"
resume_index_path = "data/resume_db.index"
vacancy_index_path = "data/vacancy_db.index"
vacancy_id = 5
topn = 10

# считываем faiss-индексы
resume_index = faiss.read_index(resume_index_path)
vacancy_index = faiss.read_index(vacancy_index_path)

# считываем базы резюме
df_resume = pd.read_csv(resume_csv_path)
df_vacancy = pd.read_csv(vacancy_csv_path)
vacancy_num = len(df_vacancy) - 1


def get_embedding_by_id(index, vec_id):
    # https://github.com/facebookresearch/faiss/issues/1068#issuecomment-1740979811
    # Create an empty numpy array to hold the vector
    vec = np.empty((1, index.d), dtype="float32")

    # Call the reconstruct method
    index.reconstruct(vec_id, vec[0])

    return vec


def main():
    # только веб приложение
    st.title("Подберем лучших кандидатов для вашей вакансии по их резюме")

    vacancy_id = st.number_input(
        "Введите id вашей вакансии",
        min_value=0,
        max_value=vacancy_num,
        value=5,
    )

    get_vacancy_btn = st.button("Показать вакансию")
    if get_vacancy_btn:
        st.write("Выбранная вами вакансия")
        st.dataframe(df_vacancy.iloc[vacancy_id])

    topn = st.number_input(
        "Введите число резюме которые мы подберем",
        min_value=1,
        max_value=20,
        value=10,
    )

    get_resumes_btn = st.button("Подобрать вакансии")
    if get_resumes_btn:
        vacancy_embed = get_embedding_by_id(vacancy_index, vacancy_id)
        _, resume_ids = resume_index.search(vacancy_embed, topn)

        st.write("Для выбранной вами вакансии")
        st.dataframe(df_vacancy.iloc[vacancy_id])

        st.write(f"Мы подобрали {topn} резюме:")
        st.dataframe(df_resume.iloc[resume_ids[0]])


if __name__ == "__main__":
    main()
