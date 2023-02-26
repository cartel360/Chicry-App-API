FROM ubuntu:latest

# Update and install required packages
RUN apt-get update && \
    apt-get install -y python3-pip

# Install additional packages
RUN apt-get install -y libsndfile1

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python packages
RUN pip3 install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the entry point
ENTRYPOINT [ "python3", "app.py" ]
