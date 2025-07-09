# Dockerfile для системы поиска изображений цветов
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы требований
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Создаем директории если их нет
RUN mkdir -p models app src

# Открываем порты
EXPOSE 8080 8501

# Команда по умолчанию - запуск API
CMD ["python3", "-m", "app.main"]


