import os
import re
import psycopg2
import requests
from dotenv import load_dotenv
import json
import uuid
import pandas as pd
from pgvector.psycopg2 import register_vector
import numpy as np
from sentence_transformers import SentenceTransformer

# Загружаем переменные окружения из .env файла
load_dotenv()

# --- Конфигурация из .env файла ---
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = "db"
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# Настройки Ollama для генерации ответов
OLLAMA_HOST = "ollama"
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_GENERATION_MODEL = os.getenv("OLLAMA_GENERATION_MODEL", "llama3")

# Загрузка русскоязычной модели для эмбеддингов
EMBEDDING_MODEL = SentenceTransformer("ai-forever/sbert_large_nlu_ru")

# --- Вспомогательные функции ---

def split_into_sentences(text: str) -> list:
    """Улучшенное разбиение текста на предложения с учетом специфики технических документов"""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def preprocess_text(text: str) -> str:
    """Очистка текста от лишних пробелов и форматирования"""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Функции для работы с Excel и RAG ---

def extract_data_from_excel(excel_path):
    """Извлекает данные из Excel файла с метаданными."""
    try:
        # Читаем метаданные документа
        metadata_df = pd.read_excel(excel_path, sheet_name="Метаданные Документа")
        metadata = metadata_df.iloc[0].to_dict()
        
        # Читаем чанки регламента
        chunks_df = pd.read_excel(excel_path, sheet_name="Чанки Регламента")
        
        # Формируем список чанков с метаданными
        chunks = []
        for _, row in chunks_df.iterrows():
            chunk_data = {
                "chunk_order": row["Chunk_Order"],
                "page_number": row["Page_Number"],
                "section_title": row["Section_Title"],
                "chunk_text": preprocess_text(row["Chunk_Text_Content"]),
                "metadata": {
                    "document_id": metadata["Document_ID"],
                    "document_title": metadata["Document_Title"],
                    "document_type": metadata["Document_Type"],
                    "document_date": str(metadata["Document_Date"]),
                    "source_file": metadata["Source_File_Name"]
                }
            }
            chunks.append(chunk_data)
            
        return chunks
    except Exception as e:
        print(f"Ошибка при чтении Excel файла '{excel_path}': {e}")
        return None

def get_embedding(text):
    """Получает векторное представление текста с помощью SentenceTransformer."""
    try:
        if not text or not isinstance(text, str):
            print(f"Ошибка: Получен неправильный текст для эмбеддинга: {text}")
            return None
            
        text = str(text).strip()
        if not text:
            return None
            
        embedding = EMBEDDING_MODEL.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"Ошибка при получении эмбеддинга для текста '{text[:50]}...': {e}")
        return None

def create_documents_table(conn):
    """Создает таблицу для хранения документов и их эмбеддингов."""
    try:
        with conn.cursor() as cur:
            # Удаляем таблицу, если она уже существует
            cur.execute("DROP TABLE IF EXISTS documents_meta;")
            
            # Создаем новую таблицу с расширением pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
            CREATE TABLE documents_meta (
                id UUID PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(1024),
                metadata JSONB,
                chunk_order TEXT,
                page_number INTEGER,
                section_title TEXT,
                document_id TEXT,
                document_title TEXT,
                document_date DATE,
                is_sentence BOOLEAN DEFAULT FALSE,
                parent_section TEXT,
                sentence_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # Создаем индексы для ускорения поиска
            cur.execute("""
            CREATE INDEX ON documents_meta USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """)
            cur.execute("CREATE INDEX ON documents_meta (parent_section);")
            cur.execute("CREATE INDEX ON documents_meta (document_id);")
            
            print("Таблица 'documents_meta' успешно создана.")
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при создании таблицы: {e}")

def store_sentences_in_db(conn, chunk_data):
    """Разбивает текст на предложения и сохраняет каждое с метаданными."""
    sentences = split_into_sentences(chunk_data["chunk_text"])
    
    for i, sentence in enumerate(sentences, 1):
        sent_metadata = chunk_data["metadata"].copy()
        sent_metadata.update({
            "is_sentence": True,
            "sentence_number": i,
            "total_sentences": len(sentences),
            "parent_section": chunk_data["section_title"]
        })
        
        embedding = get_embedding(sentence)
        if not embedding:
            continue
            
        try:
            with conn.cursor() as cur:
                sql = """
                INSERT INTO documents_meta (
                    id, content, embedding, metadata, 
                    chunk_order, page_number, section_title,
                    document_id, document_title, document_date,
                    is_sentence, parent_section, sentence_number
                )
                VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
                doc_id = uuid.uuid4()
                cur.execute(sql, (
                    str(doc_id),
                    sentence,
                    embedding,
                    json.dumps(sent_metadata),
                    f"{chunk_data['chunk_order']}.{i}",
                    chunk_data["page_number"],
                    chunk_data["section_title"],
                    sent_metadata["document_id"],
                    sent_metadata["document_title"],
                    sent_metadata["document_date"],
                    True,
                    chunk_data["section_title"],
                    i
                ))
            conn.commit()
            print(f"Сохранено предложение {i}/{len(sentences)} из раздела '{chunk_data['section_title']}'")
        except Exception as e:
            conn.rollback()
            print(f"Ошибка при сохранении предложения: {e}")

def check_documents_exist(conn, document_id):
    """Проверяет, есть ли уже чанки для данного документа в базе данных."""
    try:
        with conn.cursor() as cur:
            sql = """
            SELECT COUNT(*) FROM documents_meta
            WHERE document_id = %s;
            """
            cur.execute(sql, (document_id,))
            count = cur.fetchone()[0]
            return count > 0
    except Exception as e:
        print(f"Ошибка при проверке наличия документов в БД: {e}")
        return False

def retrieve_sections(conn, query_embedding, top_k=3):
    """Ищет наиболее релевантные разделы по запросу."""
    try:
        with conn.cursor() as cur:
            # 1. Находим наиболее релевантные предложения
            sql = """
            SELECT parent_section, document_id,
                   AVG(1 - (embedding <=> %s::vector)) AS avg_similarity,
                   COUNT(*) AS sentence_count
            FROM documents_meta
            WHERE is_sentence = TRUE
            AND (embedding <=> %s::vector) < 0.8
            GROUP BY parent_section, document_id
            ORDER BY avg_similarity DESC
            LIMIT %s;
            """
            cur.execute(sql, (query_embedding, query_embedding, top_k))
            section_results = cur.fetchall()
            
            if not section_results:
                return []
            
            # 2. Получаем все предложения для найденных разделов
            final_results = []
            for section_title, doc_id, avg_sim, sent_count in section_results:
                cur.execute("""
                SELECT content, sentence_number
                FROM documents_meta
                WHERE parent_section = %s AND document_id = %s
                ORDER BY sentence_number;
                """, (section_title, doc_id))
                
                sentences = cur.fetchall()
                full_text = " ".join([s[0] for s in sentences])
                
                final_results.append({
                    "section_title": section_title,
                    "document_id": doc_id,
                    "content": full_text,
                    "sentences": [s[0] for s in sentences],
                    "avg_similarity": avg_sim,
                    "sentence_count": sent_count
                })
            
            return final_results
            
    except Exception as e:
        print(f"Ошибка при поиске разделов: {e}")
        return []

def generate_ollama_response(prompt: str, model: str = OLLAMA_GENERATION_MODEL) -> str:
    """Генерирует ответ от LLM модели через Ollama API."""
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при генерации ответа от Ollama: {e}")
        return None


def generate_rag_response(query: str, retrieved_sections: list) -> str:
    """Генерирует ответ с использованием контекста из извлеченных документов."""
    if not retrieved_sections:
        return "Не найдено релевантной информации в базе данных."
    
    # Формируем контекст для LLM
    context = "\n\n".join([
        f"=== Раздел: {section['section_title']} ===\n"
        f"{section['content']}\n"
        f"(Релевантность: {section['avg_similarity']:.3f})"
        for section in retrieved_sections
    ])
    
    prompt = f"""Ты - технический ассистент, который помогает с информацией из регламентов. Используй предоставленный контекст для ответа на вопрос. 

Контекст:
{context}

Вопрос: {query}

Сформулируй точный и лаконичный ответ на русском языке, используя только предоставленный контекст. Если в контексте нет информации для ответа, скажи "В предоставленных материалах нет информации по этому вопросу".

Ответ:"""
    
    print("\n--- Отправляемый промпт в Ollama ---")
    print(prompt[:500] + "...")  # Логируем часть промпта для отладки
    print("------------------------------------\n")
    
    # Получаем ответ от LLM
    llm_response = generate_ollama_response(prompt)
    
    if llm_response:
        # Формируем финальный ответ с дополнительной информацией
        sources = "\n".join(
            f"- {section['section_title']} (релевантность: {section['avg_similarity']:.3f})"
            for section in retrieved_sections
        )
        response = f"Ответ:\n{llm_response}\n\nИсточники:\n{sources}"
        return response
    else:
        sections_info = "\n\n".join(
            f"{section['section_title']}:\n{section['content'][:200]}..."
            for section in retrieved_sections
        )
        return f"Не удалось получить ответ от модели. Вот наиболее релевантные разделы:\n\n{sections_info}"


# --- Основная логика скрипта ---
if __name__ == "__main__":
    # excel_file = "Регламент_Технического_Обслуживания_Промышленной_Машины_А-3000_2025-07-01.xlsx"
    excel_file = "Регламент_ЕКС_векторизация.xlsx"
    if not os.path.exists(excel_file):
        print(f"Ошибка: Файл '{excel_file}' не найден. Пожалуйста, поместите его в текущую директорию.")
        exit(1)

    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        register_vector(conn)
        print("Успешное подключение к PostgreSQL.")

        # Создаем таблицу (если нужно)
        create_documents_table(conn)

        # Извлекаем данные из Excel
        print(f"Извлечение данных из '{excel_file}'...")
        chunks_data = extract_data_from_excel(excel_file)
        if not chunks_data:
            exit(1)
        
        # Проверяем, есть ли уже этот документ в базе
        document_id = chunks_data[0]["metadata"]["document_id"]
        if check_documents_exist(conn, document_id):
            print(f"Чанки для документа '{document_id}' уже существуют в базе данных. Пропускаем загрузку.")
        else:
            print(f"Найдено {len(chunks_data)} разделов для загрузки.")
            
            print("Разбиение на предложения и сохранение в БД...")
            for i, chunk_data in enumerate(chunks_data, 1):
                print(f"\nОбработка раздела {i}/{len(chunks_data)}: {chunk_data['section_title']}")
                store_sentences_in_db(conn, chunk_data)
            print("\nПроцесс векторизации и сохранения завершен.")

        print("\n--- Демонстрация RAG-запроса ---")
        while True:
            user_query = input("\nВведите ваш вопрос (или 'exit' для выхода): ")
            if user_query.lower() == 'exit':
                break

            print("Получение эмбеддинга для запроса...")
            query_embedding = get_embedding(user_query)
            if not query_embedding:
                print("Не удалось получить эмбеддинг для запроса. Попробуйте еще раз.")
                continue

            print("Поиск релевантных разделов в базе данных...")
            retrieved_sections = retrieve_sections(conn, query_embedding, top_k=10)

            if retrieved_sections:
                print("\nНайдены следующие релевантные разделы:")
                for section in retrieved_sections:
                    print(f"  - {section['section_title']} (сходство: {section['avg_similarity']:.3f})")
                    
                print("\nГенерация ответа с помощью LLM...")
                rag_response = generate_rag_response(user_query, retrieved_sections)
                print("\n--- Ответ RAG ---")
                print(rag_response)
                print("-----------------\n")
            else:
                print("Не найдено релевантных разделов в базе данных.")
                print("Попробуйте переформулировать вопрос или добавьте больше данных в базу.")

    except psycopg2.Error as e:
        print(f"Ошибка подключения к базе данных: {e}")
    finally:
        if conn:
            conn.close()
            print("Соединение с PostgreSQL закрыто.")
            