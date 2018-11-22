"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import time
import logging as log
from inference_engine import IENetwork, IEPlugin
import numpy as np
import pandas as pd
import math
import _thread
from _datetime import datetime
import requests
import screeninfo
from multiprocessing import Process
'''Import custom class'''
from AreaOfInterest import *
from Person import *
from Camera import *
from Ads_panel import *

'''Function definition'''
def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="Path to video file or image. 'cam' for capturing video stream from camera", required=True,
                        type=str)

    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-m", "--model", required=True, type=str) #Path to an .xml file with a trained model
    parser.add_argument("-m_fc", "--model_fc", required=True, type=str)
    parser.add_argument("-m_ag", "--model_ag", required=True, type=str)
    parser.add_argument("-m_attri", "--model_attri", required=True, type=str)
    parser.add_argument("-m_hp", "--model_hp", required=True, type=str)

    parser.add_argument("-d", "--device", default="CPU",type=str) # CPU, GPU, FPGA or MYRIAD
    parser.add_argument("-d_hp", "--device_hp", default="CPU",type=str)

    parser.add_argument("-pt", "--prob_threshold", default=0.55, type=float)
    parser.add_argument("-pt_face", "--prob_threshold_face", default=0.65, type=float)

    parser.add_argument("--labels", help="Labels mapping file", default=None, type=str)

    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    return parser

'''frames capturing and manage threading functions'''
def capture_frame(cam, frames):
    while cam.video.isOpened():
        frames.insert(0, cam.frameDetections());

def frames_manage(frames):
    while 1:
        length = len(frames)
        if length > 5:
            frames.pop(-1)

'''thread for store and transmit data'''
def transmit_data(persons, stored_data):
    transmit_time = 10
    start_time = time.time()
    while 1:
        if time.time() - start_time > transmit_time:
            for i in range(len(persons)):
                list = persons[i].getAttris()
                list.insert(0,str(datetime.now().strftime("%x-%X")))
                stored_data.loc[len(stored_data)] = list
            transmit_data = {"key_order": stored_data.columns} # Save dataframe order first
            stored_data.fillna(-1,inplace = True) # Process None data
            for key in stored_data.columns:
                transmit_data[key] = stored_data[key].values.tolist()
            try:
                r = requests.post('http://127.0.0.1:5000/data',data = transmit_data)
            except Exception as e:
                # print('unable to connect')
                pass
            start_time = time.time()
            stored_data.drop(stored_data.index, inplace = True)

def frame_process(frame, n, c, h, w):
    in_frame = cv2.resize(frame, (w, h))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n, c, h, w))
    return in_frame

def eular_to_image(frame,eular_angle,center,scale):
    ##### Define camera property
    cam_matrix = np.array([[973, 0, 648],
                   [0, 973, 344],
                   [0, 0, 1]], dtype = np.float64)
    ##### Convert from degree to radian
    eular_angle = eular_angle
    # yaw = eular_angle['angle_y_fc'][0] * math.pi/180
    # pitch = eular_angle['angle_p_fc'][0] * math.pi/180
    # roll = eular_angle['angle_r_fc'][0] * math.pi/180
    yaw = eular_angle[0,0] * math.pi/180
    pitch = eular_angle[0,1] * math.pi/180
    roll = eular_angle[0,2] * math.pi/180
    ##### convert the eular_angle to each rotational axis matrix
    rx = np.array([[1,0,0],
                    [0,math.cos(pitch),-math.sin(pitch)],
                    [0,math.sin(pitch),math.cos(pitch)]])

    ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                    [0, 1, 0],
                    [math.sin(yaw), 0, math.cos(yaw)]])

    rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1]])
    r_mat = rz.dot(ry).dot(rx)

    o = np.array([0,0,frame.shape[1]])
    xAxis = r_mat.dot(np.array([scale,0,0]))+o
    yAxis = r_mat.dot(np.array([0,-scale,0]))+o
    zAxis = r_mat.dot(np.array([0,0,-scale]))+o
    zAxis2 = r_mat.dot(np.array([0,0,-700]))+o

    x_p = cam_matrix.dot(xAxis)/xAxis[2]
    x_p = x_p.astype(np.int)
    y_p = cam_matrix.dot(yAxis)/yAxis[2]
    y_p = y_p.astype(np.int)
    z_p = cam_matrix.dot(np.transpose(zAxis))/zAxis[2]
    z_p = z_p.astype(np.int)
    z_p2 = cam_matrix.dot(zAxis2)/zAxis2[2]
    z_p2 = z_p2.astype(np.int)
    center = center.astype(np.int)

    # cv2.line(frame,(center[0],center[1]),(x_p[0],x_p[1]),[255,0,0],4)
    # cv2.line(frame,(center[0],center[1]),(y_p[0],y_p[1]),[0,255,0],4)
    # cv2.line(frame,(center[0],center[1]),(z_p[0],z_p[1]),[0,0,255],4)
    # cv2.line(frame,(center[0],center[1]),(z_p2[0],z_p2[1]),[0,125,255],4)

    # cv2.circle(frame,(z_p2[0],z_p2[1]), 50, [0,125,255],2)
    return z_p2

def draw_detection_info(frame, cam, personContours, end_points):

    #fixed bounding box out of inActiveZone and still draw on the graph
    index_not_in_active = [index for index, attri in enumerate(personContours) if attri['rect'][0] not in range(cam.rangeLeft, cam.rangeRight)]
    for i in index_not_in_active:
        try:
            personContours.pop(i)
            end_points.pop(i)
        except Exception as e:
            pass


    for p, attri, end_point in zip(cam.display_pid, personContours, end_points):
        xCenter_fc, yCenter_fc, width_fc, height_fc = attri['rect']
        age = attri['age']
        gender = attri['gender']

        xmin_fc = int(xCenter_fc - width_fc/2)
        ymin_fc = int(yCenter_fc - height_fc/2)
        xmax_fc = xmin_fc + width_fc
        ymax_fc = ymin_fc + height_fc

        #head pose track
        cv2.line(frame,(xCenter_fc,yCenter_fc),(end_point[0],end_point[1]),p[1],1)
        cv2.circle(frame,(end_point[0],end_point[1]), 50, p[1],1)
        #bounding box and pid
        cv2.circle(frame, (xCenter_fc, yCenter_fc), 4, p[1], 1)
        cv2.rectangle(frame, (xmin_fc, ymin_fc), (xmax_fc, ymax_fc), p[1], 1)
        #age gender text
        cv2.putText(frame, "pid" + str(p[0]), (xmin_fc, ymin_fc-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, p[1], 1)
        cv2.putText(frame, str(gender) +", "+str(age), (xmin_fc + 50, ymin_fc - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (200, 10, 10), 1)

def draw_UI_info(frame, render_time, cam):

    #draw inActiveZone
    cv2.line(frame, (cam.rangeLeft, 0), (cam.rangeLeft, cam.h), (100, 100, 100), 2)
    cv2.line(frame, (cam.rangeRight, 0), (cam.rangeRight, cam.h), (100, 100, 100), 2)

    # Draw performance stats
    FPS = "FPS: {}".format(int(1/render_time))
    statistics_population ="Total {} people pass the screen".format(cam.entered)
    cv2.putText(frame, FPS, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
    cv2.putText(frame, statistics_population, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                (10, 10, 200), 1)

def main():

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    model_xml_fc = args.model_fc
    model_bin_fc = os.path.splitext(model_xml_fc)[0] + ".bin"

    model_xml_attri = args.model_attri
    model_bin_attri = os.path.splitext(model_xml_attri)[0] + ".bin"

    model_xml_ag = args.model_ag
    model_bin_ag = os.path.splitext(model_xml_ag)[0] + ".bin"

    model_xml_hp = args.model_hp
    model_bin_hp = os.path.splitext(model_xml_hp)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    plugin_hp = IEPlugin(device=args.device_hp, plugin_dirs=args.plugin_dir)

    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)

    # Read IR
    log.info("Reading IR...")
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    net_fc = IENetwork.from_ir(model=model_xml_fc, weights=model_bin_fc)
    net_attri = IENetwork.from_ir(model=model_xml_attri, weights=model_bin_attri)
    net_ag = IENetwork.from_ir(model=model_xml_ag, weights=model_bin_ag)
    net_hp = IENetwork.from_ir(model=model_xml_hp, weights=model_bin_hp)

    # if "CPU" in plugin.device:
    #     supported_layers = plugin.get_supported_layers(net)
    #     # supported_layers_ag = plugin_ag.get_supported_layers(net_ag)
    #     not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    #     # not_supported_layers_ag = [l for l in net.layers.keys() if l not in supported_layers_ag]
    #     if len(not_supported_layers) != 0:
    #         log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
    #                   format(plugin.device, ', '.join(not_supported_layers)))
    #         log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
    #                   "or --cpu_extension command line argument")
    #         sys.exit(1)
    # assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    # assert len(net.outputs) == 1, "Sample supports only single output topologies"

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    input_blob_fc = next(iter(net_fc.inputs))
    out_blob_fc = next(iter(net_fc.outputs))
    input_blob_attri = next(iter(net_attri.inputs))
    out_blob_attri = next(iter(net_attri.outputs))
    input_blob_ag = next(iter(net_ag.inputs))
    input_blob_hp = next(iter(net_hp.inputs))

    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)
    exec_net_face = plugin.load(network=net_fc, num_requests=2)
    exec_net_attri = plugin.load(network=net_attri, num_requests=2)
    exec_net_age = plugin_hp.load(network=net_ag, num_requests=2)
    exec_net_hp = plugin_hp.load(network=net_hp, num_requests=2)

    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob]
    n_fc, c_fc, h_fc, w_fc = net_fc.inputs[input_blob_fc]
    n_attri, c_attri, h_attri, w_attri = net_attri.inputs[input_blob_attri]
    n_ag, c_ag, h_ag, w_ag = net_ag.inputs[input_blob_ag]
    n_hp, c_hp, h_hp, w_hp = net_hp.inputs[input_blob_hp]
    del net, net_ag, net_fc, net_attri, net_hp

    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    """Cam Info"""
    cam = Camera(input_stream)
    camera_are = cam.w * cam.h

    """Ads Info"""
    ads_path = './ads/'
    ads = Advertisment(ads_path)

    """Screen Info"""
    screen = screeninfo.get_monitors()[0]
    window_name = "SmartPanel"
    """full screen"""
    # cv2.namedWindow(window_name,cv2.WND_PROP_FULLSCREEN)
    # cv2.moveWindow(window_name, screen.x - 20, screen.y - 20)
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
    #                   cv2.WINDOW_FULLSCREEN)
    """small screen"""
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 400,400)
    cv2.moveWindow(window_name, 1900,1000)


    '''Variable Definition'''
    #request id, render time
    cur_request_id = 0
    next_request_id = 1
    render_time = 0

    start_store_time = time.time()
    start_transmit_time = time.time()
    stored_data = pd.DataFrame(columns = ['gmt_occur', 'pid', 'proj_a', 'proj_b', 'proj_c',
                                            'proj_d', 'age', 'gender','enter_t','exit_t','dur'])
    transmit_interval = 10 # Define server transmission interval
    sample_interval = 1 # Define data collecting intervals
    frames = [] # Buffer container for video frame stream
    head_pose_array = np.zeros((1,3))
    head_pose_mean = np.zeros((1,3))

    '''Define projects and project area of interest'''
    project_key = ["a","b","c","d"]
    boxes = np.array([
    [0, cam.h/2, cam.w/2, 0],
    [cam.w/2, cam.h/2, cam.w, 0],
    [0,cam.h, cam.w/2, cam.h/2],
    [cam.w/2, cam.h, cam.w, cam.h/2]
    ])
    aoi = AreaOfInterest(boxes)

    try:
        _thread.start_new_thread(capture_frame,(cam,frames,))
        _thread.start_new_thread(frames_manage,(frames,))
        _thread.start_new_thread(transmit_data,(cam.valid_persons,stored_data,))
        # _thread.start_new_thread(capture_frame,(ads,video_frames,))
        # _thread.start_new_thread(frames_manage,(video_frames,))
        # _thread.start_new_thread(ads.display_ads_video,())
        p = Process(target = ads.display_ads_video, args = ())
        p.start()
    except:
        raise

    while 1:
        try:
            ret, frame = frames[0];
            # frame = cv2.flip(frame,1)
        except:
            continue
        if not ret:
            break

        in_frame = frame_process(frame, n, c, h, w)
        in_face = frame_process(frame, n_fc, c_fc, h_fc, w_fc)
        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = time.time()
        exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
        exec_net_face.start_async(request_id=next_request_id, inputs={input_blob_fc: in_face})

        if exec_net.requests[cur_request_id].wait(-1) == 0 and exec_net_face.requests[cur_request_id].wait(-1) == 0:
            # Parse detection results of the current request
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            res_fc = exec_net_face.requests[cur_request_id].outputs[out_blob_fc]

            personContours = []
            end_points = []

            for obj, obj_fc in zip(res[0][0], res_fc[0][0]):
                # Draw only objects when probability more than specified threshold
                if obj_fc[2] > args.prob_threshold_face:

                    personAttributes = {} #muset inside of the face loop otherwise it wont update

                    #if no person skip
                    size = 0
                    xmin_fc = int(obj_fc[3] * cam.w) - size
                    ymin_fc = int(obj_fc[4] * cam.h) - size
                    xmax_fc = int(obj_fc[5] * cam.w) + size
                    ymax_fc = int(obj_fc[6] * cam.h) + size
                    width_fc = xmax_fc-xmin_fc
                    height_fc = ymax_fc-ymin_fc
                    face_area = width_fc * height_fc

                    #get rid off small face
                    # DEBUG: # print(camera_are/face_area)
                    #if camera_are/face_area < 100:
                    if True:
                        #central of the face
                        xCenter_fc = int(xmin_fc + (width_fc)/2)
                        yCenter_fc = int(ymin_fc + (height_fc)/2)
                        rect = (xCenter_fc, yCenter_fc, width_fc, height_fc)

                        try:
                            #crop face
                            face = frame[ymin_fc:ymax_fc,xmin_fc:xmax_fc] #crop the face
                            face = cv2.medianBlur(face,5) # Medium blur to reduce noise in image
                            # cv2.imwrite('./face/leo.png',frame[ymin_fc + 5 :ymax_fc + 5 ,xmin_fc + 5 :xmax_fc + 5 ])
                            in_face = frame_process(face, n_ag, c_ag, h_ag, w_ag)
                            res_ag = exec_net_age.infer({input_blob_ag : in_face})
                            gender = np.argmax(res_ag['prob'])
                            age = int(res_ag['age_conv3']*100)

                            personAttributes["rect"] = rect
                            personAttributes["age"] =age
                            personAttributes["gender"] = gender

                            '''Head pose recognition and process'''
                            in_face_hp = frame_process(face, n_hp, c_hp, h_hp, w_hp)
                            res_hp = exec_net_hp.infer({input_blob_hp : in_face_hp})

                            '''Noise reduction for detected eular angle'''
                            # TODO: Improve accuracy
                            head_pose = np.array([[res_hp['angle_y_fc'][0][0],res_hp['angle_p_fc'][0][0],res_hp['angle_r_fc'][0][0]]])
                            head_pose_rmse = np.sqrt(np.mean(head_pose - head_pose_mean)**2)
                            if head_pose_rmse < 2:
                                head_pose_array = np.vstack([head_pose_array,head_pose])
                                head_pose_mean = np.mean(head_pose_array,0).reshape(1,3)
                            else:
                                head_pose_array = head_pose
                                head_pose_mean = head_pose

                            '''Remove excessive head pose from array'''
                            if len(head_pose_array) > 15:
                                head_pose_array = np.delete(head_pose_array, 0, 0)

                            '''Calculate end point position of the given pose and determine
                               focusing project and its duration for the frame'''
                            end_point = eular_to_image(frame,head_pose_mean,np.array([xCenter_fc, yCenter_fc]), 300)
                            end_points.append(end_point)

                            # keys = ads.get_key_name()
                            # p_id = ads.get_proj_id()
                            proj = aoi.check_project(end_point)
                            projects = {"a": 0, "b": 0, "c": 0, "d": 0}
                            if proj != None:
                                stayed_time = round(time.time()-inf_start,2)
                                projects[project_key[proj]] = stayed_time
                                personAttributes["project"] = projects
                            else:
                                personAttributes["project"] = projects
                            personContours.append(personAttributes)

                        except Exception as e:
                            pass

                        if obj[2] > args.prob_threshold:
                            xmin = int(obj[3] * cam.w)
                            ymin = int(obj[4] * cam.h)
                            xmax = int(obj[5] * cam.w)
                            ymax = int(obj[6] * cam.h)

                            width = xmax-xmin
                            height = ymax-ymin
                            person_area = width * height
                            # central of the person
                            xCenter = xmin + int((width)/2)
                            yCenter = ymin + int((height)/2)

                            #ratio head and person calibration
                            if person_area/face_area > 5 and abs(xCenter - xCenter_fc) < (width + width_fc)/2:

                                class_id = int(obj[1])
                                color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
                                cv2.circle(frame, (xCenter, yCenter), 4, (0,255,0), 1)

                                try:
                                    person = frame[ymin:ymax,xmin:xmax]
                                    #person attribution
                                    in_attri = frame_process(person, n_attri, c_attri, h_attri, w_attri)
                                    res_attri = exec_net_attri.infer({input_blob_attri : in_attri})[out_blob_attri][0].reshape(-1)
                                except Exception as e:
                                    pass

            cam.people_tracking(personContours)
            aoi.check_box(end_points)
            # aoi.update_info(frame)
            # aoi.draw_bounding_box(frame)

            #display the pid icon and draw face bounding box
            draw_detection_info(frame, cam, personContours, end_points)

        inf_end = time.time()
        det_time = inf_end - inf_start
        draw_UI_info(frame, det_time, cam)

        cv2.imshow(window_name,frame)
        # key = cv2.waitKey(1)
        # if key == 27:
        #     break

        cur_request_id, next_request_id = next_request_id, cur_request_id
    p.join()
    cv2.destroyAllWindows()
    del exec_net
    del plugin
if __name__ == '__main__':
    sys.exit(main() or 0)
