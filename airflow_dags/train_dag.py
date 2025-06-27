from airflow.sdk import dag, task

@dag()
def isic_train_model_dag():

    @task()
    def init_pipeline():
        print('hello')

    init_pipeline()

isic_train_model_dag()