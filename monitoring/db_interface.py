import psycopg
import logging
import json

from datetime import datetime

class  db_connector:
    def __init__(self,psycopg_conn_str):
        self.psycopg_conn_str=psycopg_conn_str
        self.conn=psycopg.connect(self.psycopg_conn_str, 
                                  autocommit=True,
                                  row_factory=psycopg.rows.dict_row)

    def __del__(self):
        self.conn.close()

    def sql_query(self,
                  query):
        logging.info(query.strip())
        res = self.conn.execute(query)
        print(res)
        return res



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

    def query_single_record(self, isic_id):

        query=f"""
        SELECT * FROM isic_inference
        WHERE isic_id={isic_id}
        """
        res_cur=self.db_connector.sql_query(query)
        return res_cur.fetchone()
    
    def query_multi_records(self, isic_ids: list):

        query=f"""
        SELECT * FROM isic_inference
        WHERE isic_id IN ({','.join( [ str(x) for x in isic_ids ] )})
        """
        res_cur=self.db_connector.sql_query(query)
        return res_cur.fetchall()


