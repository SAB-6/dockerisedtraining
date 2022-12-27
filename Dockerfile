From python:3.10

COPY requirements.txt .
COPY code/train.py /opt/ml/code


RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT [ "python", "/opt/ml/code/train.py"]