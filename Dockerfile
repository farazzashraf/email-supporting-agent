# Use an official lightweight Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the agent code
COPY . .

# Ensure /tmp/chroma_data exists and is writable
RUN mkdir -p /tmp/chroma_data && chmod 777 /tmp/chroma_data

# Expose the port (Cloud Run will override this via the PORT env var)
EXPOSE 8080

# Command to run the application using the PORT environment variable
# We use the shell form to allow environment variable expansion
CMD ["python", "main.py"]