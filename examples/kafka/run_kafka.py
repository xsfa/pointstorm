import sys
sys.path.append('/Users/tesfa/Developer/github/pointstorm')
import pointstorm.ingestion.event.kafka as kafka
import json
import os

if __name__ == "__main__":
    kafka_sentence_vectorizer = kafka.KafkaSentenceVectorizer(
        kafka_topic="user-tracker",
        kafka_bootstrap_server="localhost:9092",
        kafka_group_id=f"kafka_text_vectorizer_{os.getpid()}",
        messages_per_epoch=1,
        text_field="text",
        huggingface_model_name= "sentence-transformers/paraphrase-MiniLM-L6-v2"
    )
    kafka_sentence_vectorizer.run()