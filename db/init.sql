-- Включение расширения pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Создание таблицы для хранения документов и их векторных представлений
-- Это пример структуры таблицы для RAG
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(), -- Уникальный идентификатор документа
    content TEXT NOT NULL,                         -- Текстовое содержимое документа
    embedding VECTOR(768),                        -- Векторное представление (например, для OpenAI Ada-002)
    metadata JSONB                                 -- Метаданные о документе (источник, дата, автор и т.д.)
);

-- Создание индекса для ускорения поиска по векторам
-- Для небольших наборов данных можно начать с IVFFlat или HNSW для больших
-- Выбор индекса зависит от объема данных и требований к производительности.
-- Для начала IVFFlat_L2_FLAT является хорошим компромиссом.
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING HNSW (embedding vector_l2_ops);

-- Опционально: вставка тестовых данных
-- INSERT INTO documents (content, embedding, metadata) VALUES
-- ('Это первый тестовый документ о Docker Compose.', '[0.1,0.2,0.3,...,0.9]', '{"source": "test_doc_1"}'),
-- ('Второй документ о важности векторных баз данных.', '[0.9,0.8,0.7,...,0.1]', '{"source": "test_doc_2"}');