FROM python:3.11-slim

# Create a user to avoid running as root
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user . .

# Switch to the non-root user
USER user

# Hugging Face Spaces expose port 7860
EXPOSE 7860

# Run Streamlit on the required port
CMD ["streamlit", "run", "app_streamlit.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
