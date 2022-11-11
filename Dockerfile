FROM python:3.10

RUN mkdir /code

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /code

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=80"]