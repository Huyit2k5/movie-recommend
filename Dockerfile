FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY recsys/ recsys/
COPY data/ml-100k/ data/ml-100k/
COPY api.py app.py ./
# fit model lúc build image để container khởi động là chạy ngay
RUN python -c "from recsys.service import RecService; RecService.build()"
EXPOSE 8000 8501
# mặc định chạy API; Streamlit: docker run ... image streamlit run app.py --server.address 0.0.0.0
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
