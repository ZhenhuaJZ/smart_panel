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
    """ gmt_occur, pid, proj_a, proj_b, proj_c, proj_d, age, gender, enter_t, exit_t, dur"""
    value = [['None', 0, 1, 1, 1 ,1, 0, 0, 0, 0, 0], ['None', -1, 1, 1, 1 ,1, 0, 1, 0, 0, 0]]

    try:
        c = conn.cursor()
        c.execute(create_table_sql)
        c.executemany("INSERT INTO smrtpnl VALUES (?,?,?,?,?,?,?,?,?,?,?);", value)
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

    sql_create_projects_table = """ CREATE TABLE IF NOT EXISTS smrtpnl (
                                        gmt_occur text not null,
                                        pid integer NOT NULL,
                                        proj_a integer NOT NULL,
                                        proj_b integer NOT NULL,
                                        proj_c integer NOT NULL,
                                        proj_d integer NOT NULL,
                                        age integer NOT NULL,
                                        gender integer NOT NULL,
                                        enter_t integer NOT NULL,
                                        exit_t integer NOT NULL,
                                        dur integer NOT NULL
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

    # create a database connection
    conn = create_connection(database)
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_projects_table)
        print("Created database")
    else:
        print("Error! cannot create the database connection.")

    # export_tabel(conn, "smrtpnl")

if __name__ == '__main__':
    main()
