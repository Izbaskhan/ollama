import os
import re
import psycopg2
import requests
from dotenv import load_dotenv
import json
import uuid
import pandas as pd
from pgvector.psycopg2 import register_vector

# Загружаем переменные окружения из .env файла
load_dotenv()

# --- Конфигурация из .env файла ---
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = "db"
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

OLLAMA_HOST = "ollama"
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_GENERATION_MODEL = os.getenv("OLLAMA_GENERATION_MODEL", "llama3")

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
                "chunk_text": row["Chunk_Text_Content"],
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

def get_embedding(text, model=OLLAMA_EMBEDDING_MODEL):
    """Получает векторное представление текста от Ollama."""
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": text
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        embedding = response.json()["embedding"]
        return embedding
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении эмбеддинга от Ollama: {e}")
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
                chunk_order INTEGER,
                page_number INTEGER,
                section_title TEXT,
                document_id TEXT,
                document_title TEXT,
                document_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # Создаем индекс для поиска по вектору
            cur.execute("""
            CREATE INDEX ON documents_meta USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """)
            
            print("Таблица 'documents_meta' успешно создана.")
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при создании таблицы: {e}")

def store_document_in_db(conn, chunk_data, embedding_vector):
    """Сохраняет документ и его эмбеддинг в PostgreSQL с метаданными."""
    try:
        with conn.cursor() as cur:
            sql = """
            INSERT INTO documents_meta (
                id, content, embedding, metadata, 
                chunk_order, page_number, section_title,
                document_id, document_title, document_date
            )
            VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s);
            """
            doc_id = uuid.uuid4()
            metadata = chunk_data["metadata"]
            cur.execute(sql, (
                str(doc_id),
                chunk_data["chunk_text"],
                embedding_vector,
                json.dumps(metadata),
                chunk_data["chunk_order"],
                chunk_data["page_number"],
                chunk_data["section_title"],
                metadata["document_id"],
                metadata["document_title"],
                metadata["document_date"]
            ))
        conn.commit()
        print(f"Документ (ID: {doc_id}) успешно сохранен в БД.")
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при сохранении документа в БД: {e}")

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

def retrieve_documents(conn, query_embedding, top_k=5):
    """Ищет ближайшие документы в БД по векторному запросу."""
    try:
        with conn.cursor() as cur:
            sql = f"""
            SELECT id, content, metadata, section_title, 
                   embedding <=> %s::vector AS distance
            FROM documents_meta
            WHERE (embedding <=> %s::vector) <= %s
            ORDER BY distance
            LIMIT %s;
            """
            max_distance_threshold = 0.3
            cur.execute(sql, (query_embedding, query_embedding, max_distance_threshold, top_k))
            results = cur.fetchall()
            documents = []
            for row in results:
                doc_id, content, metadata, section_title, distance = row
                documents.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "section_title": section_title,
                    "distance": distance
                })
            return documents
    except Exception as e:
        print(f"Ошибка при извлечении документов из БД: {e}")
        return []

def generate_rag_response(query, retrieved_docs, model=OLLAMA_GENERATION_MODEL):
    """Генерирует ответ с использованием контекста из извлеченных документов."""
    context = "\n\n".join([f"Раздел: {doc['section_title']}\n{doc['content']}" for doc in retrieved_docs])
    prompt = f"""Используй следующий контекст для ответа на вопрос. Если ты не знаешь ответа на основе предоставленного контекста, просто скажи, что не можешь найти ответ. Отвечай кратко и по существу на русском языке.

Контекст:
{context}

Вопрос: {query}

Ответ:"""

    print("\n--- Отправляемый промпт в Ollama ---")
    print(prompt)
    print("------------------------------------\n")
    
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при генерации ответа от Ollama: {e}")
        return "Не удалось сгенерировать ответ."

# --- Основная логика скрипта ---
if __name__ == "__main__":
    excel_file = "Регламент_Технического_Обслуживания_Промышленной_Машины_А-3000_2025-07-01.xlsx"
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
            print(f"Найдено {len(chunks_data)} чанков для загрузки.")
            
            print("Векторизация и сохранение чанков в БД...")
            for i, chunk_data in enumerate(chunks_data):
                print(f"Обработка чанка {i+1}/{len(chunks_data)} (Страница {chunk_data['page_number']}, Раздел: {chunk_data['section_title']})...")
                embedding = get_embedding(chunk_data["chunk_text"])
                if embedding:
                    store_document_in_db(conn, chunk_data, embedding)
                else:
                    print(f"Пропущен чанк {i+1} из-за ошибки векторизации.")
            print("Процесс векторизации и сохранения завершен.")

        print("\n--- Демонстрация RAG-запроса ---")
        while True:
            user_query = input("Введите ваш вопрос (или 'exit' для выхода): ")
            if user_query.lower() == 'exit':
                break

            print("Получение эмбеддинга для запроса...")
            query_embedding = get_embedding(user_query)
            if not query_embedding:
                print("Не удалось получить эмбеддинг для запроса. Попробуйте еще раз.")
                continue

            print("Поиск релевантных документов в базе данных...")
            retrieved_documents = retrieve_documents(conn, query_embedding, top_k=3)

            if retrieved_documents:
                print(f"Найдено {len(retrieved_documents)} релевантных документов.")
                for doc in retrieved_documents:
                    print(f"  - Раздел: {doc['section_title']}, Расстояние: {doc['distance']:.4f}")
                    print(f"    Контент: '{doc['content'][:100]}...'")
                    
                print("Генерация ответа с помощью Ollama...")
                rag_response = generate_rag_response(user_query, retrieved_documents)
                print("\n--- Ответ RAG ---")
                print(rag_response)
                print("-----------------\n")
            else:
                print("Не найдено релевантных документов в базе данных.")
                print("Попробуйте переформулировать вопрос или добавьте больше данных в базу.")

    except psycopg2.Error as e:
        print(f"Ошибка подключения к базе данных: {e}")
    finally:
        if conn:
            conn.close()
            print("Соединение с PostgreSQL закрыто.")