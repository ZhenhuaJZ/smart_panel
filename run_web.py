import numpy as np
from flask import Flask, jsonify, make_response, request
import sqlite3 as sql

from bokeh.plotting import figure, show
from bokeh.models import AjaxDataSource, CustomJS
from datetime import datetime
import time
# datetime.now().strftime("%x")#date
# datetime.now().strftime("%X") #time
# datetime.now().strftime("%H:%M") #24-H : M

DATABASE = 'db/database.db'
count_project = "SELECT pid, COUNT(project) FROM smrtpnl GROUP BY pid"


def write_database(value):
    #connect to DB
    with sql.connect(DATABASE) as con:
        cur = con.cursor()
        #use executemay if have multipuls input
        cur.executemany("INSERT INTO smrtpnl VALUES (?,?,?,?,?);", value)
        con.commit()

#read tabel content
def access_database():
    with sql.connect(DATABASE) as con:
        cur = con.cursor()
        # cur.execute("SELECT * FROM smrtpnl")
        cur.execute(count_project)
        rows = cur.fetchall()
        con.commit()
        print(rows)
    return rows

# Bokeh related code
adapter = CustomJS(code="""
    const result = {x: [], y: []}
    const pts = cb_data.response.points
    for (i=0; i<pts.length; i++) {
        result.x.push(pts[i][0])
        result.y.push(pts[i][1])
    }
    return result
""")

source = AjaxDataSource(data_url='http://127.0.0.1:5000/data',
                        polling_interval=100, adapter=adapter)

p = figure(plot_height=300, plot_width=800, background_fill_color="lightgrey",
           title="Streaming Noisy sin(x) via Ajax")
p.line(x = 'x', y = 'y', line_width=2, source=source)

p.x_range.follow = "end"
p.x_range.follow_interval = 10

# Flask related code

app = Flask(__name__)

def crossdomain(f):
    def wrapped_function(*args, **kwargs):
        resp = make_response(f(*args, **kwargs))
        h = resp.headers
        h['Access-Control-Allow-Origin'] = '*'
        h['Access-Control-Allow-Methods'] = "GET, OPTIONS, POST"
        h['Access-Control-Max-Age'] = str(21600)
        requested_headers = request.headers.get('Access-Control-Request-Headers')
        if requested_headers:
            h['Access-Control-Allow-Headers'] = requested_headers
        return resp
    return wrapped_function

@app.route('/data', methods=['GET', 'OPTIONS', 'POST'])
@crossdomain
def data():

    if request.method == 'POST':
        post_list = []
        for key in request.form.getlist('key_order'):
            val = request.form.getlist(key)
            post_list.append(val)
        #jsonify(request.form.to_dict())
        post_data = np.array(post_list).transpose().tolist()
        write_database(post_data)
        print("Ok")

    rows = access_database()
    x = [row[0] for row in rows]
    y = [row[1] for row in rows]
    return jsonify(points=list(zip(x,y)))
# show and run
show(p)
app.run(port=5000)
