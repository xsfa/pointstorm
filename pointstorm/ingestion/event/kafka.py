# Generic imports
import json
import logging
import warnings
import os
import uuid
warnings.filterwarnings(action = 'ignore')

# Ingestion Imports
from bytewax.testing import run_main
from bytewax.dataflow import Dataflow
from bytewax.connectors.kafka import KafkaInput
from bytewax.connectors.stdio import StdOutput
# from kafka import KafkaConsumer, TopicPartition

# ML imports
from transformers import AutoTokenizer, AutoModel
import torch

# Local imports
from pointstorm.config import abstract_logger as kafka_logger
from pointstorm.embedding.text import Document, generate_embedding

class KafkaIngestionException(Exception):
    pass

class KafkaTextEmbeddings():
    def __init__(self, kafka_topic, kafka_bootstrap_server, kafka_config, huggingface_model_name):
        self.kafka_topic = kafka_topic
        self.kafka_bootstrap_server = kafka_bootstrap_server
        self.kafka_config = kafka_config
        self.model_name = huggingface_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def set_output(self, output):
        # TODO: Add output
        self.output = output

    def embedding(self, message) -> Document:
        """
        Generating embedding using text embedding class
        """

        if "raw_text" in json.loads(message[1]):
            message = str(json.loads(message[1])["raw_text"])
        else:
            raise KafkaIngestionException("Message does not contain the specified text field: " + self.text_field)
        
        doc = Document(
            id=str(uuid.uuid4()),
            group_key="group1",
            metadata={},
            text=[message],
            embeddings=[[]]
        )
        
        doc = generate_embedding(
            document=doc, 
            tokenizer=self.tokenizer, 
            model=self.model
        )

        kafka_logger.info(f"Generated embeddings for message: {message} ({doc.id}): {doc.embeddings}")
        return doc
    
    def run(self):
        input_config = KafkaInput(
            brokers=[self.kafka_bootstrap_server],
            topics=[self.kafka_topic],
            add_config=self.kafka_config
        )

        kafka_logger.info("Started KafkaTextEmbeddings for topic: " + self.kafka_topic)
        flow = Dataflow()
        flow.input(self.kafka_topic, input_config)
        flow.map(self.embedding)
        flow.output("stdout", StdOutput())
        run_main(flow)