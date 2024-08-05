# Use an official Python 3.10 runtime as a parent image
FROM python:3.10-slim-bullseye

# Set environment variables
ENV CLIPAPI_PORT=13339

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install system dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  git \
  ffmpeg && \
  rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir virtualenv && \
  python -m venv venv && \
  . venv/bin/activate && \
  pip install --no-cache-dir -r requirements.txt && \
  deactivate

# Make port 13339 available to the world outside this container
EXPOSE $CLIPAPI_PORT

# Run the application
CMD . venv/bin/activate && exec python main.py