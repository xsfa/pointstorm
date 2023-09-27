import os
from dotenv import load_dotenv, dotenv_values
import requests
from typing import List, Optional
from pydantic import BaseModel
from typing import Any, Optional
from numpy import ndarray
from transformers import AutoTokenizer, AutoModel

from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean, replace_unicode_quotes, clean_non_ascii_chars
from unstructured.staging.huggingface import chunk_by_attention_window
from unstructured.staging.huggingface import stage_for_transformers

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

def get_openai_models():
    """
    Retrieve all currently available embedding models from OpenAI.
    @params:
        none
    returns:
        An array containing strings of the embedding model names.
    """
    try:
        response = requests.get(
                url = "https://api.openai.com/v1/models",
                headers = {
                        'Authorization': f'Bearer {os.getenv("OPENAI_API_TOKEN")}',
                        'Content-Type': 'application/json'
                    }
            )
        if response.status_code == 200:
            data = response.json()['data']
            return [entry['id'] for entry in data]
        elif response.status_code == 401:
            print(f"You didn't provide your API key. You need to provide your API key in an Authorization header using Bearer auth.")
    except Exception as e:
        raise e

def generate_embedding(document: Document, tokenizer: AutoTokenizer = None, model: AutoModel = None, embedding_type: str = None, model_id: str = "text-embedding-ada-002") -> Document:
    """
    Generate embedding for a given document using a pretrained model.
    @params: 
        document: Document for which to generate the embeddings.
    returns: 
        Document: Document object with updated embeddings.
    """
    if embedding_type == "openai":
        load_dotenv()
        if model_id == "text-embedding-ada-002" or model_id in get_openai_models():
            pass
        else:
            raise Exception("Not a valid model id. Please choose a valid model to embed your data with.")
        response = requests.post(
            url='https://api.openai.com/v1/embeddings',
            headers = {
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {os.getenv("OPENAI_API_TOKEN")}'
                    },
            json = {
                'input': document.text,
                'model': model_id
            }
        )
        if response.status_code == 200:
            embedding_array = response.json().get('data', [])[0].get('embedding', [])
            document.embeddings.append(embedding_array)
            return document
        elif response.status_code == 401:
            raise Exception("You didn't provide your API key. You need to provide your API key in an Authorization header using Bearer auth.")
        else:
            raise Exception(f"Failed to get data. Code: {response.status_code} Content: {response.content}")
    else:
        if not tokenizer or not model:
            raise TypeError("generate_embedding() missing 2 required positional arguments: 'tokenizer' and 'model'")
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