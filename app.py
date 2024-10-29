
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import os
import torch

# Загрузка данных
data_path = 'book_data.csv'  # Убедитесь, что путь к файлу указан правильно
books_df = pd.read_csv(data_path)

# Фильтруем слишком короткие аннотации для улучшения качества поиска
books_df = books_df[books_df['annotation'].apply(lambda x: len(str(x)) > 50)]

# Загрузка модели
model_name = 'cointegrated/rubert-tiny2'  # Модель, которая будет использоваться для создания экземпляра
model = SentenceTransformer(model_name)  # Создание экземпляра модели
model.eval()  # Перевод модели в режим оценки

# Загрузка сохраненного индекса FAISS
index_path = 'faiss_index.index'  # Путь к файлу индекса
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    st.write("Ошибка: файл индекса не найден.")


def search_books(query, author_query=None, top_k=5):
    results = []

    # Если введен только автор
    if author_query and not query:
        # Фильтруем книги по автору
        filtered_books = books_df[books_df['author'].str.contains(author_query.strip(), case=False, na=False)]

        # Ограничиваем результаты до top_k
        for _, row in filtered_books.head(top_k).iterrows():  # Берем только первые top_k записей
            results.append({
                'cover_image': row['image_url'],
                'author': row['author'],
                'title': row['title'],
                'annotation': row['annotation'],
                'similarity_score': None  # Устанавливаем None, поскольку поиск по аннотации не выполняется
            })
    else:
        # Поиск по аннотациям, если введено описание
        query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)

        # Обработка результатов
        for idx, score in zip(indices[0], distances[0]):
            author = books_df.iloc[idx]['author']

            # Проверяем, есть ли фильтр по автору
            if author_query is None or (author_query is not None and author_query.strip().lower() in author.strip().lower()):
                title = books_df.iloc[idx]['title']
                annotation = books_df.iloc[idx]['annotation']
                cover_image = books_df.iloc[idx]['image_url']

                if pd.notna(author) and pd.notna(title):  # Проверка на наличие автора и названия
                    results.append({
                        'cover_image': cover_image,
                        'author': author,
                        'title': title,
                        'annotation': annotation,
                        'similarity_score': score  # Устанавливаем значение совпадения, так как запрос не пуст
                    })

    return pd.DataFrame(results)


# Streamlit app setup
st.title("📚 find my book")
st.subheader("умный поиск книг")

# Ввод пользователем
user_query = st.text_input("Введите описание книги для поиска")
author_query = st.text_input("Введите имя автора (необязательно)")
top_k = st.number_input("Количество книг для поиска", min_value=1, value=3, step=1)

# Кнопка для запуска поиска
if st.button("Найти"):
    if user_query or author_query:
        results_df = search_books(user_query, author_query if author_query else None, top_k)

        if not results_df.empty:
            st.subheader("Результаты, которые лучше всего соответствуют запросу:")
            for index, row in results_df.iterrows():
                col1, col2 = st.columns([1, 2])  # Создаем 2 колонки
                with col1:
                    if row['cover_image']:  # Проверка наличия ссылки на обложку
                        st.image(row['cover_image'], use_column_width=True)
                with col2:
                    st.subheader(row['title'])
                    st.write(f"<strong>Автор:</strong> {row['author']}", unsafe_allow_html=True)
                    st.write(f"<strong>Аннотация:</strong> {row['annotation']}", unsafe_allow_html=True)
                    if row['similarity_score'] is not None:  # Отображаем только если значение совпадения не None
                        st.write(f"<strong>Совпадение по описанию:</strong> {row['similarity_score']:.2f}%", unsafe_allow_html=True)
        else:
            st.write("Нет подходящих книг для данного запроса")
    else:
        st.write("Введите описание книги или автора для поиска")
