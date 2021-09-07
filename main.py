"""
https://fastapi.tiangolo.com/advanced/custom-response/#html-response
"""

"""
question: What is 42? 
question: Who wrote the book? 

context: Scott Adams is the author of a book  "Hitchhiker's guide to the galaxy" in which he stated that 42 was the answer to life, the universe and everything; the answer was given by a big computer of a size of the earth.
"""
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from enum import Enum

app = FastAPI()
templates = Jinja2Templates(directory="templates/")

class dropdownChoices(str, Enum):
    translation_en_to_fr = "translation_en_to_fr"
    translation_fr_to_en = "translation_fr_to_en"
    sentiment_analysis = "sentiment-analysis"
    text_generation = "text-generation"
    text2text_generation = "text2text-generation"
    summarization = "summarization"

# models not mentioned here will be [None], so default model will be loaded
specific_models={}
specific_models[dropdownChoices.translation_fr_to_en] = 'Helsinki-NLP/opus-mt-fr-en'
specific_models[dropdownChoices.text_generation] = 'EleutherAI/gpt-neo-1.3B'
specific_models[dropdownChoices.sentiment_analysis] = 'finiteautomata/beto-sentiment-analysis'
specific_models[dropdownChoices.summarization] = 'sshleifer/distilbart-cnn-12-6'

# list of models that need sentence splitting, i.e. translation and sentiment analysis.
models_to_split=[
                    dropdownChoices.translation_en_to_fr,
                    dropdownChoices.translation_fr_to_en,
                    dropdownChoices.sentiment_analysis,
                ]
pipelines={}

def get_pipeline(name, model = None):
    from transformers import pipeline
    
    p = pipelines.get(name)
    if(not p):
        model = specific_models.get(name)
        print(f"creating [{name}] pipeline, with model [{model}]")
        p = pipeline(name, model=model)
        pipelines[name] = p
    return p

@app.get("/", response_class=HTMLResponse)
async def read_items():
    return """
    <html>
        <head>
            <title>Simple HTML app</title>
        </head>
        <body>
            <h1>Navigate to <a href="http://localhost:8000/form">/form</a></h1>
        </body>
    </html>
    """

@app.get("/form")
def form_get(request: Request):
    text = "Enter some text."
    return templates.TemplateResponse(
        "form.html", context={"request": request, "text": text, 'model_names': [e.value for e in dropdownChoices], 'model_name': dropdownChoices.translation_en_to_fr}
    )

@app.post("/form")
def form_post(request: Request, 
              text: str = Form(...),
              model_name: dropdownChoices = Form(dropdownChoices.translation_en_to_fr)):
    p = get_pipeline(model_name)

    if model_name in models_to_split:
        input = map(lambda s: s.strip(), text.split('.'))
        input = list(filter(lambda x: len(x) > 0, input))
    else:
        input = [text]

    result = p(input)

    return templates.TemplateResponse(
        "form.html", context={"request": request, 
                    "text": text, 
                    "model_name": model_name, 
                    'model_names': [e.value for e in dropdownChoices], 
                    'result': result}
    )
