import os
from pointstorm.ingestion.event.kafka import KafkaTextEmbeddings

if __name__ == "__main__":
    kafka_consumer_config = {
        'group.id': f"kafka_text_vectorizer",
        'auto.offset.reset': 'largest',
        'enable.auto.commit': True
    }

    kafka_embeddings = KafkaTextEmbeddings(
        kafka_topic="user-tracker2",
        kafka_bootstrap_server="localhost:9092",
        kafka_config=kafka_consumer_config,
        huggingface_model_name= "sentence-transformers/paraphrase-MiniLM-L6-v2"
    )
    kafka_embeddings.run()