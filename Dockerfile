FROM python:3.9

WORKDIR /enc

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

RUN export PYTHONPATH='${PYTHONPATH}:/enc'

COPY . .

CMD ["python", "./test_cryptosystem.py"]