# Use the official Python slim image
FROM python:3.11.3

# Ensure environment variable is properly defined without spaces
ENV PYTHONUNBUFFERED=True

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Command to run the application using the PORT environment variable
CMD uvicorn application.app:fastapp --host 0.0.0.0 --port ${PORT}
