import hashlib
from typing import List, Optional
from pydantic import BaseModel
from typing import Any, Optional
from numpy import ndarray
from transformers import AutoTokenizer, AutoModel

import json

from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean, replace_unicode_quotes, clean_non_ascii_chars
from unstructured.staging.huggingface import chunk_by_attention_window
from unstructured.staging.huggingface import stage_for_transformers

import hashlib
from pydantic import BaseModel
from typing import Any, Optional

class Document(BaseModel):
    """
    Document object to be used for generating embeddings.
    @params:
        id: Unique identifier for the document.
        group_key: Group key for the document.
        metadata: Metadata for the document.
        text: The text.
        embeddings: Generated embeddings.
    """
    id: str
    group_key: Optional[str] = None
    metadata: Optional[dict] = {}
    text: Optional[list]
    embeddings: Optional[list] = []

def generate_embedding(document: Document, tokenizer: AutoTokenizer, model: AutoModel) -> Document:
    """
    Generate embedding for a given document using a pretrained model.
    @params: 
        document: Document for which to generate the embeddings.
    returns: 
        Document: Document object with updated embeddings.
    """
    try:
        inputs = tokenizer(
            document.text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=384
        )
        result = model(**inputs)
        embeddings = result.last_hidden_state[:, 0, :].cpu().detach().numpy()
        flattened_embeddings = embeddings.flatten().tolist()
        document.embeddings.append(flattened_embeddings)

        return document
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None