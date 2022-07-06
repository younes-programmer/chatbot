FROM python:3.9

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt')"

COPY . .

EXPOSE 5000

CMD ["flask", "run", "--host=0000", "--port=5000"]
