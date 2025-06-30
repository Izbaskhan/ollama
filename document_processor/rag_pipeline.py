import os
import psycopg2
import requests
from dotenv import load_dotenv
import json
import uuid
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

# --- Функции для работы с RAG ---

def extract_text_from_txt(txt_path):
    """Извлекает весь текст из .txt файла."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Ошибка: Файл '{txt_path}' не найден.")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла '{txt_path}': {e}")
        return None

def chunk_text(text, chunk_size=50, overlap=10):
    """Разбивает текст на чанки с перекрытием."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

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

def store_document_in_db(conn, text_content, embedding_vector, source_info=None):
    """Сохраняет документ и его эмбеддинг в PostgreSQL."""
    try:
        with conn.cursor() as cur:
            sql = """
            INSERT INTO documents (id, content, embedding, metadata)
            VALUES (%s, %s, %s, %s::jsonb);
            """
            doc_id = uuid.uuid4()
            metadata = json.dumps(source_info) if source_info else json.dumps({})
            cur.execute(sql, (str(doc_id), text_content, embedding_vector, metadata))
        conn.commit()
        print(f"Документ (ID: {doc_id}) успешно сохранен в БД.")
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при сохранении документа в БД: {e}")

def retrieve_documents(conn, query_embedding, top_k=5):
    """Ищет ближайшие документы в БД по векторному запросу."""
    try:
        with conn.cursor() as cur:
            sql = f"""
            SELECT id, content, metadata, embedding <-> %s::vector AS distance
            FROM documents
            ORDER BY distance
            LIMIT %s;
            """
            cur.execute(sql, (query_embedding, top_k))
            results = cur.fetchall()
            documents = []
            for row in results:
                doc_id, content, metadata, distance = row
                documents.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "distance": distance
                })
            return documents
    except Exception as e:
        print(f"Ошибка при извлечении документов из БД: {e}")
        return []

def check_documents_exist(conn, file_name):
    """Проверяет, есть ли уже чанки для данного файла в базе данных."""
    try:
        with conn.cursor() as cur:
            sql = """
            SELECT COUNT(*) FROM documents
            WHERE metadata->>'file_name' = %s;
            """
            cur.execute(sql, (file_name,))
            count = cur.fetchone()[0]
            return count > 0
    except Exception as e:
        print(f"Ошибка при проверке наличия документов в БД: {e}")
        return False

def generate_rag_response(query, retrieved_docs, model=OLLAMA_GENERATION_MODEL):
    """Генерирует ответ с использованием контекста из извлеченных документов."""
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])
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
    txt_file = "regulations.txt"
    if not os.path.exists(txt_file):
        print(f"Ошибка: Файл '{txt_file}' не найден в директории 'document_processor/'. Пожалуйста, поместите его туда.")
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

        # --- Проверка на существование чанков ---
        if check_documents_exist(conn, txt_file):
            print(f"Чанки для файла '{txt_file}' уже существуют в базе данных. Пропускаем загрузку.")
        else:
            print(f"Извлечение текста из '{txt_file}'...")
            full_text = extract_text_from_txt(txt_file)
            if full_text is None:
                exit(1)
            print(f"Извлечено {len(full_text)} символов.")

            chunks = chunk_text(full_text)
            print(f"Разбито на {len(chunks)} чанков.")

            print("Векторизация и сохранение чанков в БД...")
            for i, chunk in enumerate(chunks):
                print(f"Обработка чанка {i+1}/{len(chunks)}...")
                embedding = get_embedding(chunk)
                if embedding:
                    source_info = {"file_name": txt_file, "chunk_index": i}
                    store_document_in_db(conn, chunk, embedding, source_info)
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