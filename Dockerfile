FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

# Install Python 3.10, pip, and other dependencies
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip python3-dev python-is-python3 curl python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment and install dependencies
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask server using the virtual environment's python
CMD ["python", "server.py"]