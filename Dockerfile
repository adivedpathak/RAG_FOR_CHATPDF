# Use the exact same Python version as your local environment
FROM python:3.12.5-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system-level dependencies (PDF, OCR, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libgl1-mesa-glx \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project files
COPY . .

# Expose port used by FastAPI or Streamlit etc.
EXPOSE 8000

# Command to run the app (edit this if using something else)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
