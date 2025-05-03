# Use a more complete base image for TensorFlow and OpenCV compatibility
FROM python:3.9

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, TensorFlow, and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port the Flask app runs on
EXPOSE 8100

# Set environment variables for Flask
ENV FLASK_APP=webapp.py
ENV FLASK_ENV=development

# Command to run the Flask app
CMD ["python", "webapp.py"]