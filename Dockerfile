From python:3.9-alpine

WORKDIR /EducatifChatbot

ADD . /EducatifChatbot

RUN pip3 install Flask nltk torch==1.11.0+cu102 torchvision==0.12.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html

CMD ["python","main.py"]
