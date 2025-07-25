"""
interface to postgres db
"""

# pylint: disable=invalid-name
import json
import logging
from datetime import datetime

import psycopg


class db_connector:
    """
    db connector with psycopg
    """

    def __init__(self, psycopg_conn_str):
        self.psycopg_conn_str = psycopg_conn_str
        self.conn = psycopg.connect(
            self.psycopg_conn_str, autocommit=True, row_factory=psycopg.rows.dict_row
        )

    def __del__(self):
        self.conn.close()  # pylint: disable=no-member

    def sql_query(self, query):
        """
        execute query using opened connection
        """
        logging.info(query.strip())
        res = self.conn.execute(query)  # pylint: disable=no-member
        print(res)
        return res


class db_isic_data:
    """
    data  persistance table  manager
    """

    def __init__(self, db_conn: db_connector):
        self.db_conn = db_conn

    def create_table(self):
        """
        table creator
        """
        create_table_query = """
            CREATE TABLE IF NOT EXISTS isic_data(
                isic_id SERIAL PRIMARY KEY,
                patient_id SERIAL,
                target INTEGER,
                created_at TIMESTAMP NOT NULL,
                record JSON NOT NULL
            );
        """
        self.db_conn.sql_query(create_table_query)

    def insert_into_db(self, record):
        """
        insert record into table
        """
        isic_id = int(record["isic_id"].split("_")[1])
        patient_id = int(record["patient_id"].split("_")[1])
        target = None
        if 'target' in record.keys():
            target = record['target']

        insert_query = f"""
        insert into isic_data (isic_id, patient_id, target, created_at, record)
        values ({isic_id}, {patient_id},
                {'null' if target is None else target},
                '{record['timestamp']}',
                '{json.dumps(record).replace('NaN','null')}');
        """
        self.db_conn.sql_query(insert_query)

    def query_single_record(self, isic_id):
        """
        query single record
        """
        query = f"""
        SELECT * FROM isic_data
        WHERE isic_id={isic_id}
        """
        res_cur = self.db_conn.sql_query(query)
        return res_cur.fetchone()

    def query_multi_records(self, isic_ids: list):
        """
        query multiple records from a list of isic ids
        """
        query = f"""
        SELECT * FROM isic_data
        WHERE isic_id IN ({','.join( [ str(x) for x in isic_ids ] )})
        """
        res_cur = self.db_conn.sql_query(query)
        return res_cur.fetchall()

    def query_records_by_timestamp(
        self, start_timestamp: datetime, end_timestamp: datetime
    ):
        """
        query records with upload time between start and
        end timestamps
        """
        query = f"""
        SELECT * FROM isic_data
        WHERE created_at BETWEEN '{str(start_timestamp)}' AND '{str(end_timestamp)}'
        """
        res_cur = self.db_conn.sql_query(query)
        return res_cur.fetchall()

    def query_records_by_day(self, date: datetime):
        """
        query records uploaded in a certain day
        """
        date_str = date.strftime("%Y-%m-%d")
        query = f"""
        SELECT * FROM isic_data
        WHERE created_at BETWEEN '{date_str} 00:00:00' AND '{date_str} 23:59:59'
        """
        res_cur = self.db_conn.sql_query(query)
        return res_cur.fetchall()

    def query_records_with_score_by_day(self, date: datetime):
        """
        query records with inference score
        of a certain day
        """
        date_str = date.strftime("%Y-%m-%d")
        query = f"""
        SELECT * 
        FROM isic_data as d
        LEFT JOIN isic_inference as i
        ON d.isic_id=i.isic_id
        WHERE d.created_at BETWEEN '{date_str} 00:00:00' AND '{date_str} 23:59:59'
        """
        res_cur = self.db_conn.sql_query(query)
        return res_cur.fetchall()


class db_isic_inference:
    """
    inference table  manager
    """

    def __init__(self, db_conn: db_connector):
        self.db_conn = db_conn

    def create_table(self):
        """
        inference table manager
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS isic_inference(
            isic_id SERIAL PRIMARY KEY,
            score REAL
        );
        """
        self.db_conn.sql_query(create_table_query)

    def insert_into_db(self, isic_id, score):
        """
        insert into inference table
        """
        insert_query = f"""
            insert into isic_inference (isic_id, score)
            values ({isic_id}, {score});
            """
        self.db_conn.sql_query(insert_query)

    def query_single_record(self, isic_id):
        """
        query single record
        """
        query = f"""
        SELECT * FROM isic_inference
        WHERE isic_id={isic_id}
        """
        res_cur = self.db_conn.sql_query(query)
        return res_cur.fetchone()

    def query_multi_records(self, isic_ids: list):
        """
        query multiple records
        """
        query = f"""
        SELECT * FROM isic_inference
        WHERE isic_id IN ({','.join( [ str(x) for x in isic_ids ] )})
        """
        res_cur = self.db_conn.sql_query(query)
        return res_cur.fetchall()
