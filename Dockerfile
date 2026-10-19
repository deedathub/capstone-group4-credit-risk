# Use the same Python version as the local environment, where these libraries are known to work
FROM python:3.14-slim

# Set the working folder inside the container
WORKDIR /app

# Copy the pinned requirements first so Docker can reuse this step when only code changes
COPY requirements.txt .

# Install the pinned libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the demo folder, which holds app.py and model.pkl
COPY demo/ demo/

# Streamlit listens on port 8501
EXPOSE 8501

# Start the app and accept connections from outside the container
CMD ["streamlit", "run", "demo/app.py", "--server.port=8501", "--server.address=0.0.0.0"]