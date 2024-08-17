# Use the official Python slim image
FROM python:3.11.3
ENV PYTHONUNBUFFERED = True
# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "application.app:fastapp", "--host", "0.0.0.0", "--port", "8000"]
