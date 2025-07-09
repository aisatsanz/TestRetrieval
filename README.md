# Поиск похожих изображений на датасете Flowers Recognition

---
## Результаты экспериментов

| Подход                  | ResNet50 | EfficientNet-B0 | CLIP (zero-shot) | CLIP (fine-tuned) | Metric Learning | DINOv2 |
|-------------------------|---------:|----------------:|-----------------:|------------------:|----------------:|--------:|
| **Метод поиска**        | FAISS Flat | FAISS Flat    | FAISS Flat       | FAISS Flat        | FAISS Flat      | FAISS Flat |
| **Precision@5**         | 0.926    | 0.938           | 0.548            | 0.331             | 0.903             | 0.974   |
| **mAP@5**               | 0.826    | 0.895           | 0.279            | 0.238             | 0.835           | 0.869   |

> *Метрики рассчитаны на тестовой части Flowers Recognition (≈800×5 изображений). «–» означает, что Precision@5 для Metric Learning требует дополнительной проверки.*

---

## 🚀 Локальный запуск

1. **Клонировать репозиторий**  
   ```bash
   git clone <repo_url>
   cd <repo_folder>