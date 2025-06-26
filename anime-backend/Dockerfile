# Use Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Render uses 10000+)
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
