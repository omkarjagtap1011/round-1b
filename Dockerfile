# syntax=docker/dockerfile:1
FROM python:3.10-slim

# -------------------------------
# SYSTEM DEPENDENCIES
# -------------------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# SET WORKDIR
# -------------------------------
WORKDIR /app

# -------------------------------
# COPY PROJECT FILES
# -------------------------------
COPY . .

# -------------------------------
# INSTALL PYTHON DEPENDENCIES
# -------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# CLEAR CHROMA FOLDER ON EVERY RUN (optional logic in app code)
# -------------------------------

# -------------------------------
# EXECUTE MAIN SCRIPT
# -------------------------------
CMD ["python", "main.py"]
