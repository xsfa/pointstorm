import pinecone
from langchain.vectorstores import Pinecone
from embedding.text import Document, embedding
from tqdm.auto import tqdm
from uuid import uuid4
from typing import Union, List

PINECONE_API_KEY = "9caaf550-9239-44e2-abdb-98a0f63a482f"
PINECONE_ENV = "us-west4-gcp-free"

class PineconePipeline:
    api_key: str
    environment: str
    documents: Union[Document, List[Document], str]

    def __init__(self, api_key: str, environment: str) -> None:
        self.api_key = api_key
        self.environment = environment
        pinecone.init(
            api_key=self.api_key,
            environment=self.environment,
        )

    def set_data(self, input_data: Union[Document, List[Document], str]) -> None:
        if isinstance(input_data, Document):
            self.documents = [input_data]
        elif isinstance(input_data, list) and all(isinstance(doc, Document) for doc in input_data):
            self.documents = input_data
        elif isinstance(input_data, str):
            self.documents = [Document(id=str(uuid4()), text=[input_data])]
        else:
            raise ValueError("Input data should be a Document object, a list of Document objects, or a raw string.")

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
)

if ("documents" not in pinecone.list_indexes()):
    pinecone.create_index("documents", dimension=384, metric='cosine')

index = pinecone.Index("documents")

contents = []
i = 0
with open('/Users/andrewjumanca/GitHub/pointstorm-docs/pointstorm/destinations/octopi.txt', 'r') as file:
    for line in file:
        contents.append(
            Document(
                id = str(uuid4()),
                group_key = "test-doc",
                text = [line],
                embeddings = []
            )
        )
        contents[i] = embedding(document=contents[i])
        i += 1

# print([len(x.embeddings[0]) for x in contents])
# print(contents[0].embeddings[0])

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



