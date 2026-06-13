FROM python:3.11-slim

# Install system dependencies (git is required for clone-based repository analysis)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose application port
EXPOSE 8001

# Run MCP server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
