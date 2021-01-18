# simple-fastAPI-webapp
Use fastAPI to generate html web app that will serve a local directory or S3 bucket of image thumbnails and metadata, and provide links to the full sized images. Docker, python and fastAPI are the core technologies

## Docker build and run
```
docker build -t webapp .
docker run -p 8000:8000 webapp:latest
```
Then visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/). The docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Development in venv
- `python3.7 -m venv venv`
- `source venv/bin/activate`
- `pip3 install -r requirements.txt`
- `uvicorn main:app --reload`
- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## References
- https://fastapi.tiangolo.com/advanced/custom-response/ -> fastAPI HTML docs
- https://eugeneyan.com/writing/how-to-set-up-html-app-with-fastapi-jinja-forms-templates/ article and [code](https://github.com/eugeneyan/fastapi-html)
- https://haltakov.net/simple-photo-gallery/ -> flask app for serving image thumbnails as html app
- [File Management with AWS S3, Python, and Flask](https://stackabuse.com/file-management-with-aws-s3-python-and-flask/) -> flask app for uploading and downloading files to S3, with [code](https://github.com/ro6ley/flask-drive)
- https://github.com/robmarkcole/text-insights-app/tree/master/tesseract-engine -> fastAPI in docker