# Base image: Python 3.9 Slim (Debian)
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=1000

# Install system dependencies (curl for Node.js)
RUN apt-get update && apt-get install -y curl gnupg && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js 20 (LTS)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy Python requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# --- FRONTEND BUILD ---
WORKDIR /app/frontend
# Install deps and build (output to dist/)
RUN npm install
RUN npm run build

# --- BACKEND SETUP ---
WORKDIR /app/backend
RUN npm install

# --- FINAL SETUP ---
WORKDIR /app

# Expose backend port
EXPOSE 5000

# Start command (Run server from root so CWD = /app)
CMD ["node", "backend/server.js"]
