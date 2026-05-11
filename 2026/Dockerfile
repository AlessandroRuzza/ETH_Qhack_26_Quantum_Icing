# Start with a Python base image
FROM python:3.12-slim

# Install Java (JRE)
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean;

# Set the working directory
WORKDIR /app

# Copy the rest of your app's code
COPY . .
RUN uv sync --active

# Expose the port Render uses (default is 10000 or 8000 depending on your framework)
EXPOSE 10000

# Replace with your actual start command (e.g., for FastAPI, Flask, etc.)
CMD ["py server.py",]