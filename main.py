import re 
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("Aayushban-0330/t5-dialogue-summarizer")
tokenizer = T5Tokenizer.from_pretrained("Aayushban-0330/t5-dialogue-summarizer")

# Device configuration (Fixed typo here)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():  # <-- Fixed typo
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # clean lines
    text = re.sub(r"\s+", " ", text) # clean spaces
    text = re.sub(r"<.*?>", " ", text) # clean html tags
    return text.strip() 

def summarize_dialogue(dialogue : str) -> str:
    cleaned_dialogue = clean_data(dialogue)
    
    # Optional: If you used "summarize: " prefix during training, uncomment the line below
    # cleaned_dialogue = "summarize: " + cleaned_dialogue

    # Tokenize
    inputs = tokenizer(
        cleaned_dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Generate the summary
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    
    # Decode output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")