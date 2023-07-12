import pinecone
import os
import getpass
from langchain.vectorstores import Pinecone
import sys
# from embedding.text import Document, embedding

import hashlib
from pydantic import BaseModel
from typing import Any, Optional
from transformers import AutoTokenizer, AutoModel

import json

from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean, replace_unicode_quotes, clean_non_ascii_chars
from unstructured.staging.huggingface import chunk_by_attention_window
from unstructured.staging.huggingface import stage_for_transformers

import hashlib
from pydantic import BaseModel
from typing import Any, Optional

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

class Document(BaseModel):
    id: str
    group_key: Optional[str] = None
    metadata: Optional[dict] = {}
    text: Optional[list]
    embeddings: Optional[list] = []

# create embedding and store in vector db
def embedding(document):
    inputs = tokenizer(document.text, padding=True, truncation=True, return_tensors="pt", max_length=384)
    result = model(**inputs)
    embeddings = result.last_hidden_state[:, 0, :].cpu().detach().numpy()
    lst = embeddings.flatten().tolist()
    document.embeddings.append(lst)
    return document


PINECONE_API_KEY = "insert here"
PINECONE_ENV = "insert here"

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
)

index = pinecone.Index("quickstart")


example_doc = Document(
    id = "1",
    group_key = None,
    metadata = None,
    text = ["The square root of 4 is 2. 2 times 2 is 4. 4 is the square of 2."],
    embeddings = []
)

embedded_doc = embedding(document=example_doc)

index.upsert([
    ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
    ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
    ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    ("F", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ("G", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ("H", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
    ("I", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
    ("J", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
])
    
print(pinecone.list_indexes())



