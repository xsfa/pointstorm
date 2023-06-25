# pointstorm 🌩️

Real-time embeddings for data on the move

```shell
pip install pointstorm
```

```py
from pointstorm.ingestion.event.kafka import KafkaTextEmbeddings

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
```

## Kafka Example Quickstart

### Prerequisites

- Setup Kafka
- `pip install -r requirements.txt`

### Steps

1. Configure `examples/kafka/run_kafka.py`
2. Run `python3 examples/kafka/run_kafka.py`
3. Run `python3 examples/kafka/kafka_producer.py`
