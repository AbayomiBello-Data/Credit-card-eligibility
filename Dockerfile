FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DATA_PATH=./data/creditcard.csv
ENV MODEL_SAVE_PATH=./model/saved_models/model.pkl

CMD ["python", "model/model.py"]