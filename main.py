"""
https://fastapi.tiangolo.com/advanced/custom-response/#html-response
"""

""" question_answering example
What is 42? 
***
Douglas Adams is the author of a book  "Hitchhiker's guide to the galaxy" in which he stated that 42 was the answer to life, the universe and everything; the answer was given by a big computer of a size of the earth.
"""

""" zero-shot-classification example
religion,politics,language,artificial intelligence,geography,science, programming
***
The origins of the document go back to the bishops’ fight with pro-choice Catholic politicians, such as John Kerry, over the legalization of abortion. Some bishops, like Cardinal Raymond Burke, wanted to punish pro-choice Catholic politicians by denying them Communion. Other bishops, such as the late Cardinal Francis George of Chicago, disagreed. George said he did not want his priests playing cop at the Communion rail.
"""


from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from enum import Enum

from transformers import pipeline

# fastAPI setup
app = FastAPI()
templates = Jinja2Templates(directory="templates/")

# dropdown definition
class dropdownChoices(str, Enum):
    translation_en_to_fr = "translate_en_to_fr"
    translation_fr_to_en = "translate_fr_to_en"
    sentiment_analysis = "sentiment_analysis"
    text_generation = "generate_text_gpt-neo-125M"
    text_generation2 = "generate_text_gpt-neo-1.3B"
    question_answering = "question_answering"
    summarization = "summarize"
    zero_shot_classification = "zero-shot-classification"

# definition of models used by the dropdown choices
specific_models={}
specific_models[dropdownChoices.translation_en_to_fr] = {'task':'translation_en_to_fr', 'model': 'Helsinki-NLP/opus-mt-en-fr'}
specific_models[dropdownChoices.translation_fr_to_en] = {'task':'translation_fr_to_en', 'model': 'Helsinki-NLP/opus-mt-fr-en'}
specific_models[dropdownChoices.sentiment_analysis] = {'task':'sentiment-analysis', 'model': 'finiteautomata/beto-sentiment-analysis'}
specific_models[dropdownChoices.text_generation] = {'task':'text-generation', 'model': 'EleutherAI/gpt-neo-125M'}
specific_models[dropdownChoices.text_generation2] = {'task':'text-generation', 'model': 'EleutherAI/gpt-neo-1.3B'}
specific_models[dropdownChoices.question_answering] = {'task':'question-answering', 'model': None}
specific_models[dropdownChoices.summarization] = {'task':'summarization', 'model': 'sshleifer/distilbart-cnn-12-6'}
specific_models[dropdownChoices.zero_shot_classification] = {'task':'zero-shot-classification', 'model': 'facebook/bart-large-mnli'}

# list of models that need sentence splitting, i.e. translation and sentiment analysis.
models_to_split=[
                    dropdownChoices.translation_en_to_fr,
                    dropdownChoices.translation_fr_to_en,
                    dropdownChoices.sentiment_analysis,
                ]
pipelines={}

# creation and caching of models
def get_pipeline(name):
    tuple = pipelines.get(name)

    if(not tuple):
        m = specific_models.get(name)
        model = m['model']
        task = m['task']

        print(f"[{name}]: creating [{task}] pipeline, with model [{model}]")
        p = pipeline(task, model=model)
        pipelines[name] = model,task,p
    else:
        model,task,p = tuple
    
    return model,task,p

# index page
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

# get request logic
@app.get("/form")
def form_get(request: Request):
    text = "Enter some text."
    return templates.TemplateResponse(
        "form.html", context={"request": request, "text": text, 'model_names': [e.value for e in dropdownChoices], 'model_name': dropdownChoices.translation_en_to_fr}
    )

# post request logic
@app.post("/form")
def form_post(request: Request, 
              text: str = Form(...),
              model_name: dropdownChoices = Form(dropdownChoices.translation_en_to_fr),
              send_json_response: bool = Form(False)):
    import time

    t0 = time.time()

    model,task,p = get_pipeline(model_name)

    if model_name in models_to_split:
        # for models that only use single sentences, we need to slit them
        input = map(lambda s: s.strip(), text.split('.'))
        input = list(filter(lambda x: len(x) > 0, input))
        result = p(input)
    elif model_name == dropdownChoices.zero_shot_classification:
        #for zero classification we need two parameters, so we split input by ***
        t1,t2 = [x.strip() for x in text.split('***')]
        labels = [x.strip() for x in t1.split(',')]
        input = [t2]
        print(f"input={input}\nlabels={labels}")
        result = p(input, labels, multi_class=True)
    elif model_name == dropdownChoices.question_answering:
        #for question asswering we need two parameters, so we split input by ***
        question,context = [x.strip() for x in text.split('***')]
        print(f"question={question}\ncontext={context}")
        input = [context]
        result = p(question=question, context=context)
    else:
        # any other case (no splitting sentences, just one parameter needed for the pipeline)
        input = [text]
        result = p(input)

    info = {'time': round(time.time()-t0, 2) , 
            'input': input,
            'task':task,
            'model':model}

    return templates.TemplateResponse(
        "form.html", context={"request": request, 
                    "text": text, 
                    "model_name": model_name, 
                    'model_names': [e.value for e in dropdownChoices], 
                    'result': result,
                    'info': info}
    )

