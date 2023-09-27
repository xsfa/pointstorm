import unittest, os, pinecone
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock
from pointstorm.destinations.PineconePipeline import PineconePipeline
from pointstorm.embedding.text import Document
from typing import List

class TestCreatePineconeModule(unittest.TestCase):
    connection = PineconePipeline()

    def test_pinecone_pipeline_init(self):
        self.assertIsNotNone(self.connection.pinecone_api_key)
        self.assertEqual(self.connection.pinecone_api_key, os.getenv('PINECONE_API_KEY'))
        self.assertIsNotNone(self.connection.pinecone_environment)
        self.assertEqual(self.connection.pinecone_environment, os.getenv('PINECONE_ENV'))
        
    def test_pinecone_import_data(self):
        # Verifying if the set data is a Document object
        doc_mock = MagicMock(spec=Document)
        docs_mock = [MagicMock(spec=Document) for _ in range(3)]
        string_mock = MagicMock(spec=Document)

        self.connection.set_data(doc_mock)
        self.assertEqual(self.connection.documents, [doc_mock])

        self.connection.set_data(docs_mock)
        self.assertEqual(self.connection.documents, docs_mock)

        self.connection.set_data(string_mock)
        print(self.connection.documents)
        self.assertEqual(self.connection.documents, [string_mock])



    def test_pinecone_data_upsert(self):
        with patch("pinecone.list_indexes") as list_indexes_mock, \
            patch("pinecone.create_index") as create_index_mock, \
            patch("pinecone.Index") as IndexMock:

            list_indexes_mock.return_value = []  # Returning an empty list so that the index will be created
            index_mock = MagicMock()
            IndexMock.return_value = index_mock

            # Mock the initialize_contents method and make it return a list of documents
            with patch.object(PineconePipeline, "initialize_contents") as init_contents_mock:
                doc_mock = MagicMock(spec=Document)
                doc_mock.id = "test_id"
                doc_mock.embeddings = []  # Mock embeddings as empty list
                doc_mock.group_key = "test_group_key"
                doc_mock.text = "example_text"

                contents_mock = [doc_mock]

                init_contents_mock.return_value = contents_mock

                connection = PineconePipeline()

                # Call the upsert method
                connection.upsert("test-index", "pointstorm/destinations/octopi.txt")

                # Ensure the pinecone functions were called correctly
                list_indexes_mock.assert_called_once()
                create_index_mock.assert_called_once_with("test-index", dimension=384, metric='cosine')
                IndexMock.assert_called_once_with("test-index")

    def test_pinecone_index(self):
        # Verify if pinecone index is what we want it to be
        # and whether we can get the name of it

        # Mock the pinecone.list_indexes function
        with patch("pinecone.list_indexes") as list_indexes_mock:
            list_indexes_mock.return_value = ["index1", "index2"]

            indexes = pinecone.list_indexes()

            # Test the name of the index
            self.assertIn("index1", indexes)

            # Test that the pinecone.list_indexes() function was called
            list_indexes_mock.assert_called_once()


if __name__ == '__main__':
    unittest.main(exit=False)