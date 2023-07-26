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
        # Test that the document model we want to import as our
        # data input is in the correct format and has all 
        # necessary attributes
        pass

    def test_pinecone_data_upsert(self):
        # Test whether we can correctly upsert the data we want
        pass

    def test_pinecone_index(self):
        # Verify if pinecone index is what we want it to be
        # and whether we can get the name of it
        pass


if __name__ == '__main__':
    unittest.main()