import unittest
from unittest.mock import patch, MagicMock
from destinations import PineconePipeline
from embedding.text import Document, embedding

PINECONE_API_KEY = "9caaf550-9239-44e2-abdb-98a0f63a482f"
PINECONE_ENV = "us-west4-gcp-free"

class TestCreatePineconeModule(unittest.TestCase):
    connection: PineconePipeline

    def test_pinecone_pipeline_init(self):
        connection = PineconePipeline(
            api_key = PINECONE_API_KEY,
            environment = PINECONE_ENV
        )
        self.connection = connection
        self.assertEqual(connection.api_key, PINECONE_API_KEY)
        self.assertEqual(connection.environment, PINECONE_ENV)
        
    def test_pinecone_import_data(self):
        

if __name__ == '__main__':
    unittest.main()