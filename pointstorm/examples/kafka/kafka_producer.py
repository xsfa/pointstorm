from pointstorm.producers.text_producer import TextProducer
from faker import Faker

fake = Faker()

# Create Kafka Producer
kafka_config = {
    'bootstrap.servers':'localhost:9092',
}

kafka_topic = "user-tracker2"
p = TextProducer(kafka_topic, kafka_config)
print('Kafka Producer has been initiated...')

# Producing 10 random sentences to Kafka topic
def produce():    
    for i in range(10):
        # generating fake sentence
        sentence = fake.sentence(nb_words=6, variable_nb_words=True, ext_word_list=None)
        p.produce(sentence)

produce()