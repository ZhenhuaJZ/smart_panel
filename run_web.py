from math import pi
import pandas as pd

import numpy as np
from flask import Flask, jsonify, make_response, request, render_template
import sqlite3 as sql

from bokeh.plotting import figure, show
from bokeh.models import AjaxDataSource, CustomJS, DatetimeTickFormatter, Range1d, DataRange1d
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

f2_vbar_interval = 0.55

#how many ppl watched
total_num_of_ppl_viewing = "SELECT COUNT(*) FROM smrtpnl"
#gender percentage
gender_ratio = "SELECT gender, COUNT(gender) FROM  smrtpnl GROUP BY gender"
#how many ppl viewing different project
proj_a_viewing = "SELECT gender, COUNT(*) FROM smrtpnl WHERE proj_a > 0 GROUP BY gender"
proj_b_viewing = "SELECT gender, COUNT(*) FROM smrtpnl WHERE proj_b > 0 GROUP BY gender "
proj_c_viewing = "SELECT gender, COUNT(*) FROM smrtpnl WHERE proj_c > 0 GROUP BY gender"
proj_d_viewing = "SELECT gender, COUNT(*) FROM smrtpnl WHERE proj_d > 0 GROUP BY gender"

def write_database(value):
    #connect to DB
    with sql.connect(DATABASE) as con:
        cur = con.cursor()
        #use executemay if have multipuls input
        cur.executemany("INSERT INTO smrtpnl VALUES (?,?,?,?,?,?,?,?,?,?,?);", value)
        con.commit()
#read tabel content
def access_database():
    with sql.connect(DATABASE) as con:
        cur = con.cursor()

        cur.execute(total_num_of_ppl_viewing)
        output1 = cur.fetchall()

        cur.execute(gender_ratio)
        output2 = cur.fetchall()

        cur.execute(proj_a_viewing)
        proj_a = cur.fetchall()
        cur.execute(proj_b_viewing)
        proj_b = cur.fetchall()
        cur.execute(proj_c_viewing)
        proj_c = cur.fetchall()
        cur.execute(proj_d_viewing)
        proj_d = cur.fetchall()
        con.commit()

    return output1, output2, [proj_a, proj_b, proj_c, proj_d]

app = Flask(__name__)

@app.route("/")
def graph():
    ajax_input = dict(x_time=[], x1_time=[], y=[], y_female=[], y_male=[], x_female_proj=[], y_female_proj=[], x_male_proj=[], y_male_proj=[])

    source = AjaxDataSource(data = ajax_input, data_url='http://127.0.0.1:5000/data',
                            polling_interval= ajax_refresh_inter * 1000) #adapter=adapter)

    p = figure(plot_height=300, plot_width=800, x_axis_type = "datetime", tools="wheel_zoom,reset",
               title="People Viewing Statistics")
    p.line(x= 'x_time', y= 'y', line_dash="4 4", line_width=3, color='gray', source=source)
    p.vbar(x= 'x_time', top='y_female', width=200, alpha=0.5, color='red', legend='female', source=source) #female
    p.vbar(x= 'x1_time', top='y_male', width=200, alpha=0.5, color='blue', legend='male', source=source) #male
    p.xaxis.formatter = DatetimeTickFormatter(milliseconds=["%X"],seconds=["%X"],minutes=["%X"],hours=["%X"])
    p.y_range = DataRange1d(start = 0, range_padding = 5) #padding leave margin on the top
    p.legend.orientation = "horizontal" #legend horizontal
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Number'
    # p.x_range.follow = "end"
    # p.x_range.follow_interval = timedelta(seconds=30)

    p2 = figure(plot_height=300, plot_width=800,tools="wheel_zoom,reset",
               title="Project Viewing Statistics")
    p2.vbar(x='x_female_proj', top='y_female_proj', width=f2_vbar_interval, alpha=0.5, color='red', legend='female', source=source) #female
    p2.vbar(x='x_male_proj', top='y_male_proj', width=f2_vbar_interval, alpha=0.5, color='blue', legend='male', source=source) #male
    p2.xaxis.ticker = [2, 6, 10, 14]
    p2.xaxis.major_label_overrides = {2: 'P1', 6: 'P2', 10: 'P3', 14: 'P4'}

    p2.x_range = DataRange1d(start = 0, end = 16) #padding leave margin on the top
    p2.y_range = DataRange1d(start = 0, range_padding = 5) #padding leave margin on the top
    p2.legend.orientation = "horizontal"
    p2.xaxis.axis_label = 'Project'
    p2.yaxis.axis_label = 'Number'

    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    script, div = components((p,p2))

    html = render_template(
                           'embed.html',
                           plot_script=script,
                           plot_div=div[0],
                           plot_div2=div[1],
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
x_time = []
y = []
"""gender ratio"""
x1_time = []
y_female = []
y_male = []

""" project_total_viewing"""
x_female_proj = []
x_male_proj = []
y_female_proj = []
y_male_proj = []


@app.route('/data', methods=['GET', 'OPTIONS', 'POST'])
@crossdomain
def data():
    global x_time, y, x1_time, y_female, y_male, x_female_proj ,y_female_proj ,x_male_proj, y_male_proj
    if len(x_time) > 5:
        """ total_num_of_ppl_viewing_y"""
        x_time.pop(0)
        y.pop(0)
        """gender ratio"""
        x1_time.pop(0)
        y_female.pop(0)
        y_male.pop(0)

        """ project_total_viewing"""
        x_female_proj.pop(0)
        y_female_proj.pop(0)
        x_male_proj.pop(0)
        y_male_proj.pop(0)


    sql_init = 2 #pid =0,-1 / male female
    sql_init_gender = 1

    x_in_aust = (time.time()+11*60*60)*1000 #Australia UTC+8 11h*60*60 = S #BOKEH IN ms !!!!!!
    if request.method == 'POST':
        post_list = []
        for key in request.form.getlist('key_order'):
            val = request.form.getlist(key)
            post_list.append(val)
        post_data = np.array(post_list).transpose().tolist()
        ## DEBUG:
        print(post_data)
        write_database(post_data)
        print("Ok")

    o1, o2, o3 = access_database()

    x_time.append(x_in_aust)
    y.append(o1[0][0]-sql_init)  #total
    x1_time.append(x_in_aust+200)
    y_female.append(o2[0][1]-sql_init+1) #female
    y_male.append(o2[1][1]-sql_init+1) #male

    print(o3)

    for i, o in enumerate(o3):
        for o_inner in o:
            if o_inner[0] == 0 : y_female_proj.append(o_inner[1] -sql_init_gender); x_female_proj.append(i*4 + 1.75) #[#1, #2, #3, #4] how many female view p1-p4
            if o_inner[0] == 1 : y_male_proj.append(o_inner[1] -sql_init_gender); x_male_proj.append(i*4 + 1.75 + f2_vbar_interval) #how many male view p1-p

    if len(x_time) < 2: #init
        return jsonify(x_time=[], y=[], x1_time=[], y_female=[], y_male=[], x_female_proj=[], y_female_proj=[], x_male_proj=[], y_male_proj=[])

    # return jsonify(x=x, x1=x1, y=y, y1=y1, y2=y2, y3=y3, y4=y4, y5=y5, y6=y6)
    return jsonify(x_time=x_time, y=y, x1_time=x1_time, y_female=y_female, y_male=y_male, x_female_proj=x_female_proj, y_female_proj=y_female_proj, x_male_proj=x_male_proj, y_male_proj=y_male_proj)

# show(p)
app.run(host="0.0.0.0", port=5000, debug=True)
