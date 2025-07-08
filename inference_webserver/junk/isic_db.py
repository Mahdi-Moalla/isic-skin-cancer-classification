import os
import json
import datetime

import pandas as pd
import psycopg
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# https://stackoverflow.com/questions/68405302/how-to-insert-json-data-into-postgres-database-table

def run_query(psycopg_conn_str,
              query):
    print(query.strip())
    with psycopg.connect(psycopg_conn_str, autocommit=True) as conn:
        res = conn.execute(query)
        print(res)
    return res
    
        

def create_user(psycopg_conn_str, 
                user_name, 
                user_password):
    res=run_query(psycopg_conn_str,
                  f"SELECT COUNT(*) FROM pg_catalog.pg_user WHERE usename = '{user_name}';")
    if res.fetchone()[0]==1:
        return
    
    run_query(psycopg_conn_str,
              f"CREATE USER {user_name} WITH LOGIN PASSWORD '{user_password}';")
    

def create_db(psycopg_conn_str,
              user_name,
              db_name):
    
    run_query(psycopg_conn_str,
              f"DROP DATABASE IF EXISTS  {db_name};")
    
    # from IPython import embed; embed(colors='Linux')    
    
    run_query(psycopg_conn_str,
              f"CREATE DATABASE {db_name} OWNER {user_name};")


def create_table(psycopg_conn_str):
    
    create_table_query = """
        DROP TABLE IF EXISTS isic_data;
        CREATE TABLE isic_data(
            isic_id SERIAL PRIMARY KEY,
            target INTEGER,
            data JSON NOT NULL,
            image_uuid VARCHAR UNIQUE NOT NULL,
            created_at TIMESTAMP NOT NULL
        );
    """
    run_query(psycopg_conn_str,
              create_table_query)


def insert_into_db(psycopg_conn_str,
                   data_dict):
    isic_id=int(data_dict["isic_id"].split("_")[1])
    insert_query=f"""
    insert into isic_data (isic_id, data,image_uuid,created_at)
    values ('{isic_id}','{json.dumps(data_dict).replace('NaN','null')}','1-1-1-1','{str(datetime.datetime(2025,1,1,0,0,0))}');
    """
    run_query(psycopg_conn_str,
              insert_query)



if __name__=='__main__':
    psycopg_conn_str="host=postgres-db port=5432 user=postgres password=postgres"
    user_name='isic_db_user'
    user_password='isic_db_password'
    db_name='isic_db'
    create_user(psycopg_conn_str, 
                user_name=user_name,
                user_password=user_password)
    create_db(psycopg_conn_str, 
                user_name=user_name,
                db_name=db_name)
    
    psycopg_conn_str_newuser=f"host=postgres-db port=5432 dbname={db_name} user={user_name} password={user_password}"

    create_table(psycopg_conn_str_newuser)

    data=pd.read_csv('test-metadata.csv')

    print(json.dumps(data.iloc[0].to_dict(), indent=4))

    insert_into_db(psycopg_conn_str_newuser,
                   data.iloc[0].to_dict())
