from bytewax import Dataflow, cluster_main, spawn_cluster
from bytewax.inputs import KafkaInputConfig
from kafka import KafkaConsumer, TopicPartition

import warnings
warnings.filterwarnings(action = 'ignore')

from transformers import AutoTokenizer, AutoModel
import torch

import json
import logging

from config import abstract_logger as kafka_logger

class KafkaIngestionException(Exception):
    pass

class KafkaSentenceVectorizer():
    def __init__(self, kafka_topic, kafka_bootstrap_server, kafka_group_id, messages_per_epoch, text_field, huggingface_model_name):
        self.kafka_topic = kafka_topic
        self.kafka_bootstrap_server = kafka_bootstrap_server
        self.kafka_group_id = kafka_group_id
        self.messages_per_epoch = messages_per_epoch
        self.text_field = text_field
        self.model_name = huggingface_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.previous_messages = self.get_previous_messages()

    def set_topic(self, kafka_topic):
        self.kafka_topic = kafka_topic

    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer

    def get_previous_messages(self):
        previous_messages = []

        consumer = KafkaConsumer(
            self.kafka_topic,
            bootstrap_servers=self.kafka_bootstrap_server,
            group_id=self.kafka_group_id,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            value_deserializer=lambda x: x.decode('utf-8')
        )

        partitions = consumer.partitions_for_topic(self.kafka_topic)
        topic_partitions = [TopicPartition(self.kafka_topic, p) for p in partitions]
        end_offsets = consumer.end_offsets(topic_partitions)

        for message in consumer:

            previous_messages.append(json.loads(message.value)[self.text_field]) if self.text_field in json.loads(message.value) else previous_messages.append(json.loads(message.value))

            topic_partition = TopicPartition(message.topic, message.partition)
            if message.offset == end_offsets[topic_partition] - 1:
                topic_partitions.remove(topic_partition)

                if not topic_partitions:
                    consumer.close()
                    break
        
        return previous_messages

    def vectorize(self, message):
        """
        Vectorize the message using the trained model
        """
        message = str(json.loads(message[1])[self.text_field]) if self.text_field in json.loads(message[1]) else str(json.loads(message[1]))
        inputs = self.tokenizer(message, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        kafka_logger.info(f"Generated embeddings for message: {message}, {embeddings}")

    def run(self):
        kafka_logger.info("Started KafkaSentenceVectorizer for topic: " + self.kafka_topic)
        flow = Dataflow()

        flow.map(self.vectorize)
        flow.capture()

        input_config = KafkaInputConfig(
            self.kafka_bootstrap_server,
            self.kafka_group_id,
            self.kafka_topic,
            messages_per_epoch=self.messages_per_epoch
        )

        def print_output(worker_index, worker_count):
            def destination_helper(feed):
                offset = feed[0]
                data = feed[1]
                # . . .
            return destination_helper
        spawn_cluster(flow, input_config, print_output)