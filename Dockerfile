FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY app.py .

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]