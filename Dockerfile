FROM python:3.10.6
WORKDIR /app
COPY flask_app /app/flask_app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "--pythonpath", "flask_app", "-w 2", "-b :5000", "app:app"]
