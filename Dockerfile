FROM python:3.12-slim

# Copy the entire project
COPY . mlops_project/

# Install the package with dependencies
RUN pip install ./mlops_project

# Set the working directory
WORKDIR /mlops_project

# Expose the port gunicorn will listen on
EXPOSE 5001

# Run the API (uvicorn; Dockerfile completo na Fase 4)
CMD ["uvicorn", "--factory", "app.api:create_app", "--host=0.0.0.0", "--port=5001"]
