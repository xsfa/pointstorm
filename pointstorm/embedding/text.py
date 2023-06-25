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