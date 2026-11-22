FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy pandas scipy fastapi uvicorn fastmcp pydantic
COPY core_btr/ ./core_btr/
COPY server.py .
COPY artifacts/ ./artifacts/
ENV BTR_BUNDLE_PATH="artifacts/duke_pod.pt"
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
