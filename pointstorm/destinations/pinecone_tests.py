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
        doc_mock, docs_mock, string_mock = MagicMock(spec=Document), MagicMock(spec=list(Document)), MagicMock(spec=str)

        self.connection.set_data(doc_mock)
        self.assertEqual(self.connection.documents, [doc_mock])

        self.connections.set_data(docs_mock)
        self.assertEqual(self.connection.documents, [docs_mock])

        self.connections.set_data(string_mock)
        self.assertEqual(self.connection.documents, [string_mock])



    def test_pinecone_data_upsert(self):
        # Test whether we can correctly upsert the data we want
        # For this test, we need to mock the pinecone functions and initialize_contents method

        # Mock the pinecone.Index class
        with patch("pinecone.Index") as IndexMock:
            index_mock = MagicMock()
            IndexMock.return_value = index_mock

            # Mock the initialize_contents method and make it return a list of documents
            with patch.object(PineconePipeline, "initialize_contents") as init_contents_mock:
                contents_mock = [MagicMock(spec=Document)]
                init_contents_mock.return_value = contents_mock

                connection = PineconePipeline(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

                # Call the upsert method
                connection.upsert("test-index", "path/to/txt/file.txt")

                # Ensure the pinecone functions were called correctly
                IndexMock.assert_called_once_with("test-index")
                index_mock.upsert.assert_called_once_with(vectors=([(contents_mock[0].id, contents_mock[0].embeddings, [{'group_key': contents_mock[0].group_key, 'file_content': contents_mock[0].text}])]))

    def test_pinecone_index(self):
        # Verify if pinecone index is what we want it to be
        # and whether we can get the name of it

        # Mock the pinecone.list_indexes function
        with patch("pinecone.list_indexes") as list_indexes_mock:
            list_indexes_mock.return_value = ["index1", "index2"]

            connection = PineconePipeline(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

            # Test the name of the index
            self.assertEqual(connection.index_name, "index1")

            # Test that the pinecone.list_indexes function was called
            list_indexes_mock.assert_called_once()


if __name__ == '__main__':
    unittest.main()