FROM python:3.9-slim

WORKDIR /app

COPY ["predict.py","model.pickle","requirements.txt","dict_vectorizer.pickle","./"]

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8790

CMD [ "python", "predict.py" ]