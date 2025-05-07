FROM python:3.9-slim-buster

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your code and model
COPY . .

# Set the command to run your script
CMD ["python", "pan2025_dataset1.py", "-i", "$inputDataset", "-o", "$outputDir"]