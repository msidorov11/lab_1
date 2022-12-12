FROM python:3.8

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app

ADD . /app

COPY . .

EXPOSE 8000

ENTRYPOINT [ "python" ]
CMD ["app.py"]
