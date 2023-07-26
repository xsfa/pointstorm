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

    # pinecone.init(
    #     api_key=PINECONE_API_KEY,
    #     environment=PINECONE_ENV,
    # )

    def initialize_contents(self, txt_path):
        contents = []
        with open(txt_path, 'r') as file:
            for line in file:
                doc = Document(
                    id=str(uuid4()),
                    group_key="test-doc",
                    text=[line],
                    embeddings=[]
                )
                doc = self.embedding(document=doc)
                contents.append(doc)
        return contents


    def upsert(self, index_name, txt_path):
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=384, metric='cosine')

        index = pinecone.Index(index_name)
        contents = self.initialize_contents(txt_path)

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





