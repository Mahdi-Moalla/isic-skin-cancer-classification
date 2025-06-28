from airflow.decorators import dag, task

@dag()
def train_model_pipeline():
    @task()
    def init_pipeline():
        print('init_pipeline')
        return 1
    @task()
    def train_model(x):
        print(f'train_model {x}')

    train_model(init_pipeline())

train_model_pipeline()
