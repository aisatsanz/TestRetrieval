# Поиск похожих изображений на датасете Flowers Recognition

---

## Архитектура решения

> ! [Архитектура перенести из miro](./architecture.png){width=60%}

---

## Структура проекта

- **extractors/**  
  Классы-экстракторы признаков (ResNet, EfficientNet, CLIP, Metric Learning и др.), все с миксином `FineTuneMixin` (валидация, early stopping, регуляризация).
- **evaluate.py**  
  Функция `evaluate()` для подсчёта Precision@5, Recall@5 и замера задержки поиска.
- **config.yaml**  
  Все гиперпараметры и настройки: пути, lr, batch_size, тип индекса, метрика, verbose и тд.
- **api.py**  
  FastAPI-сервис с одним endpoint `/search-image` (POST image → JSON топ-5 результатов).
- **Dockerfile**  
  Собирает контейнер с зависимостями (torch, timm, transformers, faiss, fastapi).

---

## Сравнение моделей

| Подход                   | ResNet50 FT | EfficientNet-B0 | CLIP (zero-shot) | CLIP (FT) | Metric Learning |
|--------------------------|-------------|-----------------|------------------|-----------|-----------------|
| **Метод обучения**       | CrossEntropy| CrossEntropy    | —                | CrossEntropy (vision) | TripletLoss      |
| **Индекс**               | FAISS HNSW  | FAISS HNSW      | FAISS HNSW       | FAISS HNSW| FAISS HNSW      |
| **Precision@5**          | TBD         | TBD             | TBD              | TBD       | TBD             |
| **mAP@5**                | TBD         | TBD             | TBD              | TBD       | TBD             |
| **Время отклика**        | TBD ms      | TBD ms          | TBD ms           | TBD ms    | TBD ms          |

