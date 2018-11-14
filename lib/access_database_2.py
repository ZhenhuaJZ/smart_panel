import sqlite3
import requests
import csv

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    """init pid = 0, -1 male and femal """
    init_value_proj = [['None', 0, -1], ['None', 0, 0], ['None', 0, 1], ['None', 0, 2], ['None', 0, 3],
                        ['None', -1, -1], ['None', -1, 0], ['None', -1, 1], ['None', -1, 2], ['None', -1, 3]] #gmt, pid, project
    init_value_gender = [['None', 0, 0, 0], ['None', -1, 0, 1]] #gmt, pid, age, gender

    try:
        c = conn.cursor()
        c.execute(create_table_sql[0])
        c.execute(create_table_sql[1])
        c.executemany("INSERT INTO smrtpnl VALUES (?,?,?);", init_value_proj)
        c.executemany("INSERT INTO atrbts VALUES (?,?,?,?);", init_value_gender)
        conn.commit()
    except Exception as e:
        raise
#read tabel content
def export_tabel(conn, tabel_name):
    cur = conn.cursor()
    with conn:
        cur.execute("SELECT * FROM {}".format(tabel_name))
        rows = cur.fetchall()
        f = csv.writer(open('smtpnl.csv', 'a'))
        for row in rows:
            f.writerow(row)

def main():
    database = "../db/database.db"

    pid_project_table = """ CREATE TABLE IF NOT EXISTS smrtpnl (
                                        gmt_occur text not null,
                                        pid integer NOT NULL,
                                        project integer NOT NULL
                                    ); """
    attributes_table = """ CREATE TABLE IF NOT EXISTS atrbts (
                                        gmt_occur text not null,
                                        pid integer NOT NULL,
                                        age integer NOT NULL,
                                        gender integer NOT NULL
                                    ); """
    # sql_create_tasks_table = """CREATE TABLE IF NOT EXISTS tasks (
    #                                 id integer PRIMARY KEY,
    #                                 name text NOT NULL,
    #                                 priority integer,
    #                                 status_id integer NOT NULL,
    #                                 project_id integer NOT NULL,
    #                                 begin_date text NOT NULL,
    #                                 end_date text NOT NULL,
    #                                 FOREIGN KEY (project_id) REFERENCES projects (id)
    #                             );"""
    crt_table = [pid_project_table, attributes_table]
    # create a database connection
    conn = create_connection(database)
    if conn is not None:
        # create projects table
        create_table(conn, crt_table)
        print("Created database")
    else:
        print("Error! cannot create the database connection.")

    # export_tabel(conn, "smrtpnl")

if __name__ == '__main__':
    main()
