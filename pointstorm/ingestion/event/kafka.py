# Generic imports
import json
import logging
import warnings
import os
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
from pointstorm.embedding.text import Document, embedding

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
        # self.previous_messages = self.get_previous_messages()

    #     def set_topic(self, kafka_topic):
    #         self.kafka_topic = kafka_topic

    # def get_previous_messages(self):
    #     previous_messages = []

    #     consumer = KafkaConsumer(
    #         self.kafka_topic,
    #         bootstrap_servers=self.kafka_bootstrap_server,
    #         group_id=self.kafka_group_id,
    #         auto_offset_reset='earliest',
    #         enable_auto_commit=True,
    #         value_deserializer=lambda x: x.decode('utf-8')
    #     )

    #     partitions = consumer.partitions_for_topic(self.kafka_topic)
    #     topic_partitions = [TopicPartition(self.kafka_topic, p) for p in partitions]
    #     end_offsets = consumer.end_offsets(topic_partitions)

    #     for message in consumer:

    #         previous_messages.append(json.loads(message.value)[self.text_field]) if self.text_field in json.loads(message.value) else previous_messages.append(json.loads(message.value))

    #         topic_partition = TopicPartition(message.topic, message.partition)
    #         if message.offset == end_offsets[topic_partition] - 1:
    #             topic_partitions.remove(topic_partition)

    #             if not topic_partitions:
    #                 consumer.close()
    #                 break
        
    #     return previous_messages

    def set_output(self, output):
        # TODO: Add output
        self.output = output

    def embedding(self, message):
        """
        Generating embedding using text embedding class
        """

        if "raw_text" in json.loads(message[1]):
            message = str(json.loads(message[1])["raw_text"])
        else:
            raise KafkaIngestionException("Message does not contain the specified text field: " + self.text_field)
        
        # doc = Document(
        #     id=
        #     text=message,
        #     )
        inputs = self.tokenizer(message, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        kafka_logger.info(f"Generated embeddings for message: {message}, {embeddings}")
    
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