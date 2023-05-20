import json
import time
import logging
import random 
from confluent_kafka import Producer
from faker import Faker

fake = Faker()
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='producer.log',
                    filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create Kafka Producer
p = Producer({'bootstrap.servers':'localhost:9092'})
print('Kafka Producer has been initiated...')

def receipt(err,msg):
    if err is not None:
        print('Error: {}'.format(err))
    else:
        message = 'Produced message on topic {} with value of {}\n'.format(msg.topic(), msg.value().decode('utf-8'))
        logger.info(message)
        print(message)

# Producing 10 random sentences to Kafka topic
def produce():    
    for i in range(10):
        # generating fake sentence, should be somewhat related
        sentence = fake.sentence(nb_words=6, variable_nb_words=True, ext_word_list=None)
        data = {
            'msg_time': float(time.time()),
            'text': sentence    
        }

        m = json.dumps(data)

        # producing data to kafka topic, user-tracker
        p.poll(1)
        p.produce('user-tracker', m.encode('utf-8'),callback=receipt)
        p.flush()

        # time.sleep(0.1)

produce()