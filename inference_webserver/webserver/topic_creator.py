import os
from kafka.admin import KafkaAdminClient, NewTopic
import kafka

if __name__=='__main__':
    kafka_server=os.getenv('kafka_server',
                            "kafka:9092")
    
    kafka_topic_name=os.getenv('kafka_topic_name',
                                'isic_topic')
    kafka_topic_num_partitions=\
            int(os.getenv('kafka_topic_num_partitions',
                                1))
    kafka_topic_replication_factor=\
            int(os.getenv('kafka_topic_replication_factor',
                                1))

    admin_client = KafkaAdminClient(
        bootstrap_servers=kafka_server, 
        client_id='isic_topic_creator'
    )

    topic_list = []
    topic_list.append(NewTopic(name=kafka_topic_name, 
            num_partitions=kafka_topic_num_partitions, 
            replication_factor=kafka_topic_replication_factor))
    try:
        admin_client.create_topics(new_topics=topic_list, 
                            validate_only=False)
    except kafka.errors.TopicAlreadyExistsError:
        pass