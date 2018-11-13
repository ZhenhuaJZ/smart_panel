from math import pi
import pandas as pd

import numpy as np
from flask import Flask, jsonify, make_response, request, render_template
import sqlite3 as sql

from bokeh.plotting import figure, show
from bokeh.models import AjaxDataSource, CustomJS, NumeralTickFormatter, DatetimeTickFormatter
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
from bokeh.palettes import Category20c
from bokeh.transform import cumsum
from datetime import datetime, timedelta
import time
# datetime.now().strftime("%x")#date
# datetime.now().strftime("%X") #time
# datetime.now().strftime("%H:%M") #24-H : M
DATABASE = 'db/database.db'
ajax_refresh_inter = 2 #second


count_project = "SELECT pid, COUNT(project) FROM smrtpnl GROUP BY pid"
#how many ppl watched
total_num_of_ppl_viewing = "SELECT COUNT(*) FROM (SELECT *, COUNT(pid) FROM smrtpnl GROUP BY pid)"
#how many ppl viewing different project
project_total_viewing = "SELECT project, COUNT(DISTINCT pid) as num_project FROM smrtpnl GROUP BY project"
#gender percentage
gender_ratio = "SELECT avg_gender, COUNT(avg_gender) FROM (SELECT pid,  CAST(ROUND(AVG(gender)) AS INT) as avg_gender FROM smrtpnl GROUP BY pid) GROUP BY avg_gender"

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

        cur.execute(total_num_of_ppl_viewing)
        output1 = cur.fetchall()

        cur.execute(project_total_viewing)
        output2 = cur.fetchall()

        cur.execute(gender_ratio)
        output3 = cur.fetchall()

        # cur.execute("SELECT * FROM smrtpnl")
        # output5 = cur.fetchall()
        # print(output5)
        con.commit()

    return output1, output2, output3

# Bokeh related code
# adapter = CustomJS(code="""
#     const result = {x: [], y: []}
#     const pts = cb_data.response.points
#     for (i=0; i<pts.length; i++) {
#         result.x.push(pts[i][0])
#         result.y.push(pts[i][1])
#     }
#     return result
# """)
app = Flask(__name__)

@app.route("/")
def graph():
    ajax_input = dict(x=[], x1=[], y=[], y1=[], y2=[], y3=[], y4=[], y5=[], y6=[])

    source = AjaxDataSource(data = ajax_input, data_url='http://127.0.0.1:5000/data',
                            polling_interval= ajax_refresh_inter * 1000) #adapter=adapter)

    p = figure(plot_height=300, plot_width=800, x_axis_type = "datetime", y_range = (0,10), 
               title="People Viewing Statistics")
    p.line(x= 'x', y= 'y', line_dash="4 4", line_width=3, color='gray', source=source)
    p.vbar(x= 'x', top='y5', width=200, alpha=0.5, color='red', legend='female', source=source) #female
    p.vbar(x= 'x1', top='y6', width=200, alpha=0.5, color='blue', legend='male', source=source) #male
    p.xaxis.formatter = DatetimeTickFormatter(milliseconds=["%X"],seconds=["%X"],minutes=["%X"],hours=["%X"])
    # p.x_range.follow = "end"
    # p.x_range.follow_interval = timedelta(seconds=30)

    p2 = figure(plot_height=300, plot_width=800, x_axis_type = "datetime",
               title="Project Viewing Statistics")
    p2.line(x = 'x', y = 'y1', line_width=3, color='#FB9A99', legend='p1', source=source) #p1
    p2.line(x = 'x', y = 'y2', line_width=3, color='#A6CEE3', legend='p2', source=source) #p2
    p2.line(x = 'x', y = 'y3', line_width=3, color='black', legend='p3', source=source) #p3
    p2.line(x = 'x', y = 'y4', line_width=3, color='#33A02C', legend='p4', source=source) #p4
    p2.xaxis.formatter = DatetimeTickFormatter(milliseconds=["%X"],seconds=["%X"],minutes=["%X"],hours=["%X"])
    # p2.x_range.follow = "end"
    # p2.x_range.follow_interval = timedelta(seconds=30)



    #p3 = figure(plot_height=350,
               #toolbar_location=None, tools="")


    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    script, div = components((p,p2))

    html = render_template(
                           'embed.html',
                           plot_script=script,
                           plot_div=div[0],
                           plot_div2=div[1],
                          # plot_div3=div[2],
                           js_resources=js_resources,
                           css_resources=css_resources,
                           )

    return encode_utf8(html)

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

""" total_num_of_ppl_viewing_y"""
x = []
y = []
""" project_total_viewing"""
y1 = []
y2 = []
y3 = []
y4 = []
"""gender ratio"""
x1 = []
y5 = []
y6 = []

@app.route('/data', methods=['GET', 'OPTIONS', 'POST'])
@crossdomain
def data():
    global x, x1, y, y1 ,y2 ,y3, y4, y5, y6
    if len(x) > 5:
        """ total_num_of_ppl_viewing_y"""
        x.pop(0)
        y.pop(0)
        """ project_total_viewing"""
        y1.pop(0)
        y2.pop(0)
        y3.pop(0)
        y4.pop(0)
        """gender ratio"""
        x1.pop(0)
        y5.pop(0)
        y6.pop(0)

    sql_init = 2 #pid =0,-1 / male female

    x_in_aust = (time.time()+11*60*60)*1000 #Australia UTC+8 11h*60*60 = S #BOKEH IN ms !!!!!!
    if request.method == 'POST':
        post_list = []
        for key in request.form.getlist('key_order'):
            val = request.form.getlist(key)
            post_list.append(val)
        post_data = np.array(post_list).transpose().tolist()
        ## DEBUG:  print(post_data)
        write_database(post_data)
        print("Ok")

    o1, o2, o3 = access_database()

    x.append(x_in_aust)
    y.append(o1[0][0]-sql_init)  #total
    y1.append(o2[1][1]-sql_init) #p1
    y2.append(o2[2][1]-sql_init) #p2
    y3.append(o2[3][1]-sql_init) #p3
    y4.append(o2[4][1]-sql_init) #p4

    x1.append(x_in_aust+200)
    y5.append(o3[0][1]-sql_init+1) #female
    y6.append(o3[1][1]-sql_init+1) #male

    if len(x) < 2: #init
        return jsonify(x=[], x1=[], y=[], y1=[], y2=[], y3=[], y4=[], y5=[], y6=[])

    return jsonify(x=x, x1=x1, y=y, y1=y1, y2=y2, y3=y3, y4=y4, y5=y5, y6=y6)

# show(p)
app.run(host="0.0.0.0", port=5000, debug=True)
