from sentence_transformers import SentenceTransformer

# Загрузка модели (автоматически скачается с Hugging Face)
model = SentenceTransformer("ai-forever/sbert_large_nlu_ru")

# Пример текста
text = "Кошка спит на ковре."

# Генерация эмбеддинга (вектор 1024d)
embedding = model.encode(text)
print(embedding.shape)  # (1024,)