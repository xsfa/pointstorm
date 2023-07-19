import pinecone
import os
import getpass
from langchain.vectorstores import Pinecone
import sys
from embedding import text
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from tqdm.auto import tqdm
from uuid import uuid4

PINECONE_API_KEY = "9caaf550-9239-44e2-abdb-98a0f63a482f"
PINECONE_ENV = "us-west4-gcp-free"

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
)

if ("documents" not in pinecone.list_indexes()):
    pinecone.create_index("documents", dimension=384, metric='cosine')

index = pinecone.Index("documents")

example_doc = text.Document(
    id = str(uuid4()),
    group_key = "test-doc",
    # metadata = None,
    text = ["An octopus' favorite color is always purple. Swag money swag money. Test Test. 12341223334234"],
    embeddings = []
)
embedded_doc = text.embedding(document=example_doc)
contents = [example_doc]
batch_size = 1


for i in tqdm(range(0, len(contents), batch_size)):
    ids = contents[i].id
    embeddings = contents[i].embeddings
    data = [{
        'group_key': contents[i].group_key,
        'file_content': contents[i].text
    }]
    to_upsert = list(zip(ids, embeddings, data))
    index.upsert(vectors=to_upsert)

    # index.upsert(embedded_doc.embeddings)


# index.upsert([
#     ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
#     ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
#     ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
#     ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
#     ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
#     ("F", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
#     ("G", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
#     ("H", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
#     ("I", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
#     ("J", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
# ])
    
# print(pinecone.list_indexes())



