import sys
import json
from typing import List, Dict, Any
from enum import Enum
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import torch
import numpy as np

# Load the pre-trained BERT model and tokenizer
# model_name = 'bert-base-uncased'
model_name = 'neuralmind/bert-large-portuguese-cased'
#tokenizer = BertTokenizer.from_pretrained(model_name)
#model = BertModel.from_pretrained(model_name, output_attentions=True)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

class ModelName(str, Enum):
    # Enum of the available models. This allows the API to raise a more specific
    # error if an invalid model is provided.
    en = "english"
    pt = "portuguese"
    pt_gsd = "portuguese-gsd"


DEFAULT_MODEL = ModelName.pt

class Article(BaseModel):
    # Schema for a single article in a batch of articles to process
    text: str

class RequestModel(BaseModel):
    articles: List[Article]
    tokens: List[str]
    #model: ModelName = DEFAULT_MODEL
    model: str

class ResponseModel(BaseModel):
    # This is the schema of the expected response and depends on what you
    # return from get_data.
    embedding: List

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Function to get word embeddings from the model
def get_word_embedding(word, model_name='bert-base-uncased'):

    # Tokenize the input word and convert to tensor
    inputs = tokenizer(word, return_tensors='pt')

    # Get the model's outputs (hidden states)
    with torch.no_grad():
        outputs = model(**inputs)

    # Take the embeddings for the [CLS] token (sentence representation)
    # BERT embeddings: outputs[0] -> (batch_size, sequence_length, hidden_size)
    # We take the hidden state corresponding to the first token ([CLS] token in BERT).
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    #embedding = embedding.cpu().numpy()

    return embedding

# Function to calculate cosine similarity between two word embeddings
def calculate_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Main function to compute semantic similarity between two words
def semantic_similarity(word1, word2, model_name='bert-base-uncased'):
    # Get embeddings for each word
    embedding1 = get_word_embedding(word1, model_name)
    embedding2 = get_word_embedding(word2, model_name)

    # Calculate cosine similarity
    similarity = calculate_similarity(embedding1, embedding2)
    return similarity


# You can switch between BERT, XLM-Roberta, or any other model
#model_name = 'xlm-roberta-base'  # For XLM-Roberta
# model_name = 'bert-base-uncased'  # For BERT

@app.post("/embedding/", summary="Process batches of text", response_model=ResponseModel)
def embedding(query: RequestModel):
    text = query.articles[0].text
    embedding = get_word_embedding(text, model_name)
    return {"embedding" :embedding.tolist()}

@app.post("/similarity/", summary="Process batches of text", response_model=ResponseModel)
def stanza(query: RequestModel):
    similarity_score = semantic_similarity(word1, word2, model_name)
    print(query.tokens)
    all = p.posdep(query.tokens, is_sent=True)
    return {"result": all}
