import unittest
from unittest.mock import patch, MagicMock
from destinations.PineconePipeline import PineconePipeline
from embedding.text import Document, embedding

PINECONE_API_KEY = "9caaf550-9239-44e2-abdb-98a0f63a482f"
PINECONE_ENV = "us-west4-gcp-free"

class TestCreatePineconeModule(unittest.TestCase):
    connection = PineconePipeline(
            api_key = PINECONE_API_KEY,
            environment = PINECONE_ENV
        )

    def test_pinecone_pipeline_init(self):
        self.assertEqual(self.connection.api_key, PINECONE_API_KEY)
        self.assertEqual(self.connection.environment, PINECONE_ENV)
        
    def test_pinecone_import_data(self):
 
        # Verifying if the set data is a Document object
        doc_mock, string_mock = MagicMock(spec=Document), MagicMock(spec=str)
        
        # Test case 1: Setting data with a single Document object
        self.connection.set_data(doc_mock)
        self.assertEqual(self.connection.documents, [doc_mock])

        # Test case 2: Setting data with a list of Document objects
        self.connection.set_data([doc_mock, doc_mock])
        self.assertEqual(self.connection.documents, [doc_mock, doc_mock])

        # Test case 3: Setting data with a string
        self.connection.set_data("This is a test string.")
        self.assertEqual(len(self.connection.documents), 1)
        self.assertIsInstance(self.connection.documents[0], Document)

        # Test case 4: Setting data with an unsupported input type
        with self.assertRaises(ValueError):
            self.connection.set_data(12345)  # Pass an unsupported input type, like an integer


    def test_pinecone_data_upsert(self):
        # Test whether we can correctly upsert the data we want
        pass

    def test_pinecone_index(self):
        # Verify if pinecone index is what we want it to be
        # and whether we can get the name of it
        pass


if __name__ == '__main__':
    unittest.main()