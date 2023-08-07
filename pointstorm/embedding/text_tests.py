import unittest
from pointstorm.embedding.text import Document, generate_embedding
from unittest.mock import patch, MagicMock
from transformers import AutoTokenizer, AutoModel
import torch

class TestDocumentModel(unittest.TestCase):
    def test_document_model_creation(self):
        doc = Document(
            id="123",
            group_key="group1",
            metadata={"author": "John Doe"},
            text=["Hello, world!"],
            embeddings=[[]]
        )
        self.assertEqual(doc.id, "123")
        self.assertEqual(doc.group_key, "group1")
        self.assertEqual(doc.metadata, {"author": "John Doe"})
        self.assertEqual(doc.text, ["Hello, world!"])
        self.assertEqual(doc.embeddings, [[]])


@patch("builtins.print")
class TestGenerateEmbedding(unittest.TestCase):
    @patch.object(AutoTokenizer, 'from_pretrained')
    @patch.object(AutoModel, 'from_pretrained')
    def setUp(self, mock_model, mock_tokenizer):
        self.mock_model = mock_model
        self.mock_tokenizer = mock_tokenizer
        self.document = Document(
            id="123",
            group_key="group1",
            metadata={"author": "John Doe"},
            text=["Hello, world!"],
            embeddings=[[]]
        )

    def test_generate_embedding_success(self, mock_print):
        # Mocking the tokenizer and model return values
        self.mock_tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        self.mock_model.return_value = MagicMock(last_hidden_state=torch.tensor([[1, 2, 3]]))

        # Create fake tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

        result = generate_embedding(document=self.document, tokenizer=tokenizer, model=model)

        # Check that the function returned a Document
        self.assertIsInstance(result, Document)

        # Check that embeddings are there
        self.assertGreaterEqual(len(result.embeddings), 1)

# Run the tests
if __name__ == '__main__':
    unittest.main()