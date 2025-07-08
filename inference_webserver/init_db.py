import os
import json
import datetime

import pandas as pd
import psycopg
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


def run_query(psycopg_conn_str,
              query):
    logging.info(query)
    with psycopg.connect(psycopg_conn_str, autocommit=True) as conn:
        res = conn.execute(query)
        logging.info(str(res))
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
    
    run_query(psycopg_conn_str,
              f"CREATE DATABASE {db_name} OWNER {user_name};")


if __name__=='__main__':
    
    postgres_server=os.getenv("postgres_server",
                              "postgres-db")
    postgres_port=os.getenv("postgres_port",
                            5432)
    postgres_password=os.getenv("postgres_password",
                                "postgres")
    
    postgres_db_name=os.getenv("postgres_db_name",
                               "isic_db")
    postgres_db_user=os.getenv("postgres_db_user",
                               "isic_db_user")
    postgres_db_password=os.getenv("postgres_db_password",
                                   "isic_db_password")
    
    psycopg_conn_str=f"host={postgres_server} port={postgres_port} user=postgres password={postgres_password}"
    
    logging.info(f"psycopg_conn_str= {psycopg_conn_str}")

    create_user(psycopg_conn_str, 
                user_name=postgres_db_user,
                user_password=postgres_db_password)
    create_db(psycopg_conn_str,
                user_name=postgres_db_user,
                db_name=postgres_db_name)
