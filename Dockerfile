# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*


# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create a non-root user and switch to it for security
RUN useradd -m myuser
USER myuser

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variable
ENV FLASK_APP=app.py
ENV PORT=7860

# Run the application
CMD ["python", "app.py"]
