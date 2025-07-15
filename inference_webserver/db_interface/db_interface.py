import psycopg
import logging
import json



class  db_connector:
    def __init__(self,psycopg_conn_str):
        self.psycopg_conn_str=psycopg_conn_str
        self.conn=psycopg.connect(self.psycopg_conn_str, autocommit=True)

    def __del__(self):
        self.conn.close()

    def sql_query(self,
                  query):
        logging.info(query.strip())
        res = self.conn.execute(query)
        print(res)
        return res
    

class db_isic_data:

    
    def __init__(self, db_connector: db_connector):
        self.db_connector=db_connector

    def create_table(self):
    
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS isic_data(
                isic_id SERIAL PRIMARY KEY,
                patient_id SERIAL,
                target INTEGER,
                created_at TIMESTAMP NOT NULL,
                record JSON NOT NULL
            );
        """
        self.db_connector.sql_query(create_table_query)

    def insert_into_db(self,
                    record):
        isic_id=int(record["isic_id"].split("_")[1])
        patient_id=int(record["patient_id"].split("_")[1])
        target=None
        if 'target' in record.keys():
            target=record['target']
        
        insert_query=f"""
        insert into isic_data (isic_id, patient_id, target, created_at, record)
        values ({isic_id}, {patient_id}, {'null' if target is None else target}, '{record['timestamp']}' ,'{json.dumps(record).replace('NaN','null')}');
        """
        self.db_connector.sql_query(insert_query)



class db_isic_inference:

    
    def __init__(self, db_connector: db_connector):
        self.db_connector=db_connector

    def create_table(self):
    
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS isic_inference(
            isic_id SERIAL PRIMARY KEY,
            score REAL
        );
        """
        self.db_connector.sql_query(create_table_query)

    def insert_into_db(self,
                       isic_id,
                       score):
        insert_query=f"""
            insert into isic_inference (isic_id, score)
            values ({isic_id}, {score});
            """
        self.db_connector.sql_query(insert_query)






