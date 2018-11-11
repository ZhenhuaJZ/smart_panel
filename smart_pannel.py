#!/usr/bin/env python
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
'''Import custom class'''
from AreaOfInterest import *
from Person import *
from Camera import *

'''Global definition'''
frames = []

'''Function definition'''
def transmit_data(data):
    transmit_data = {"key_order": data.columns} # Save dataframe order first
    data.fillna(-1,inplace = True) # Process None data
    print(data)
    for key in data.columns:
        transmit_data[key] = data[key].values.tolist()
    try:
        r = requests.post('http://127.0.0.1:5000/data',data = transmit_data)
    except Exception as e:
        return
    return

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

def capture_frame(cam):
    while cam.video.isOpened():
        frames.insert(0, cam.frameDetections());

def frames_manage():
    print(len(frames))
    if len(frames) > 5:
        frames.pop(-1)

def frame_process(frame, n, c, h, w):
    in_frame = cv2.resize(frame, (w, h))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n, c, h, w))
    return in_frame

def get_3d_landmark():
    land_mark = np.array([
    [-170, 170, -135], # Left eye left corner
    [170, 170, -135], # Right eye right corner
    [0, 0,  0], # Nose
    [-150, -150, -125], # Left Mouth corner
    [150, -150, -125],
    ]
    )

    return land_mark

def eular_to_image(frame,eular_angle,center,scale):
    ##### Define camera property
    cam_matrix = np.array([[950, 0, center[0]],
                   [0, 950, center[1]],
                   [0, 0, 1]], dtype = np.float64)
    ##### Convert from degree to radian
    eular_angle = eular_angle
    yaw = eular_angle['angle_y_fc'][0] * math.pi/180
    pitch = eular_angle['angle_p_fc'][0] * math.pi/180
    roll = eular_angle['angle_r_fc'][0] * math.pi/180
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
    cv2.line(frame,(center[0],center[1]),(z_p2[0],z_p2[1]),[0,125,255],4)
    return z_p2

'''
Convert given 3d reference point and 2d landmark to obtain
3d to 2d projection properties
'''
def landmark_3d_to_2d(img, landmark_2d):
    cam_matrix = np.array([[950, 0, img.shape[1]/2],
                   [0, 950, img.shape[0]/2],
                   [0, 0, 1]], dtype = np.float64)

    dist_coeffs = np.zeros((4,1))
    landmark_2d = landmark_2d.astype(np.float64)

    retval, rvec, tvec = cv2.solvePnP(get_3d_landmark().astype(np.float64), landmark_2d, cam_matrix, dist_coeffs)

    end_point_3d = np.array([[0.0, 0.0, 1000.0]]) # 0.0, 0.0, 2000.0

    end_point_2d, _ = cv2.projectPoints(end_point_3d, rvec, tvec, cam_matrix, dist_coeffs)
    end_point_2d = end_point_2d.astype(np.int)
    landmark_2d = landmark_2d.astype(np.int)
    cv2.line(img,(landmark_2d[2,:][0],landmark_2d[2,:][1]),(end_point_2d[0][0][0],end_point_2d[0][0][1]),[255,0,0],4)

    return end_point_2d

def main():
    # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
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
    exec_net_age = plugin.load(network=net_ag, num_requests=2)
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

    cur_request_id = 0
    next_request_id = 1

    log.info("Starting inference in async mode...")
    log.info("To switch between sync and async modes press Tab button")
    log.info("To stop the sample execution press Esc button")
    is_async_mode = True
    render_time = 0

    frame_detection_interval = 0 #ms
    detection_end_time = 0
    cam = Camera(input_stream)
    camera_are = cam.w * cam.h
    ####@ Define array for landmarks
    # landmark_2d_array = np.zeros((5,2))
    # landmark_2d_array = []

    '''Variable Definition'''
    start_store_time = time.time()
    start_transmit_time = time.time()
    stored_data = pd.DataFrame(columns = ['gmt', 'pid', 'project', 'age', 'gender'])
    transmit_interval = 10 # Define server transmission interval
    sample_interval = 1 # Define data collecting intervals

    '''Define area of interest'''
    boxes = np.array([
    [0, cam.h/2, cam.w/2, 0],
    [cam.w/2, cam.h/2, cam.w, 0],
    [0,cam.h, cam.w/2, cam.h/2],
    [cam.w/2, cam.h, cam.w, cam.h/2]
    ])
    aoi = AreaOfInterest(boxes)

    try:
        _thread.start_new_thread(capture_frame,(cam,))
        _thread.start_new_thread(frames_manage,())
    except:
        raise

    while 1:
        try:
            ret, frame = frames[0];
            frame = cv2.flip(frame,1)
        except:
            continue
        if not ret:
            break
        cv2.line(frame, (cam.rangeLeft, 0), (cam.rangeLeft, cam.h), (0,255,0), 2)
        cv2.line(frame, (cam.rangeRight, 0), (cam.rangeRight, cam.h), (0,255,0), 2)
        cv2.line(frame, (cam.midLine, 0), (cam.midLine, cam.h), (0,255,0), 2)
        in_frame = frame_process(frame, n, c, h, w)
        in_face = frame_process(frame, n_fc, c_fc, h_fc, w_fc)

        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = time.time()

        if is_async_mode:
            exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
            exec_net_face.start_async(request_id=next_request_id, inputs={input_blob_fc: in_face})
        else:
            exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
            exec_net_face.start_async(request_id=cur_request_id, inputs={input_blob_fc: in_face})

        #if (inf_start-detection_end_time)*1000 >= frame_detection_interval:
        aoi.draw_bounding_box(frame)
        if exec_net.requests[cur_request_id].wait(-1) == 0 and exec_net_face.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

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
                    if camera_are/face_area < 100:
                        cv2.rectangle(frame, (xmin_fc, ymin_fc), (xmax_fc, ymax_fc), (10, 10, 200), 2)
                        #central of the face
                        xCenter_fc = int(xmin_fc + (width_fc)/2)
                        yCenter_fc = int(ymin_fc + (height_fc)/2)
                        rect = (xCenter_fc, yCenter_fc, width_fc, height_fc)

                        # Face centre point
                        cv2.circle(frame, (xCenter_fc, yCenter_fc), 5, (0,255,0), 3)

                        try:
                            #crop face
                            face = frame[ymin_fc:ymax_fc,xmin_fc:xmax_fc] #crop the face
                            in_face = frame_process(face, n_ag, c_ag, h_ag, w_ag)
                            res_ag = exec_net_age.infer({input_blob_ag : in_face})
                            sex = np.argmax(res_ag['prob'])
                            age = int(res_ag['age_conv3']*100)

                            personAttributes["rect"] = rect
                            personAttributes["age"] =age
                            personAttributes["gender"] = sex

                            ##### head pose
                            in_face_hp = frame_process(face, n_hp, c_hp, h_hp, w_hp)
                            res_hp = exec_net_hp.infer({input_blob_hp : in_face_hp})
                            # head_pose = []
                            # # print(res_hp)
                            # for key in res_hp.keys():
                            #     # print(key)
                            #     head_pose.append(res_hp[key][0])
                            #     # print(res_hp[key][0])
                            end_point = eular_to_image(frame,res_hp,np.array([xCenter_fc, yCenter_fc]), 300)
                            end_points.append(end_point)
                            proj = aoi.check_project(end_point)
                            personAttributes["project"] = proj

                            personContours.append(personAttributes)
                            # aoi.check_box(end_point)

                            cv2.putText(frame, str(sex) +", "+str(age), (xmin_fc + 100, ymin_fc - 7),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (200, 10, 10), 1)

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
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                                cv2.circle(frame, (xCenter, yCenter), 5, (0,255,0), 3)

                                try:
                                    person = frame[ymin:ymax,xmin:xmax]
                                    #person attribution
                                    in_attri = frame_process(person, n_attri, c_attri, h_attri, w_attri)
                                    res_attri = exec_net_attri.infer({input_blob_attri : in_attri})[out_blob_attri][0].reshape(-1)
                                except Exception as e:
                                    pass

                    detection_end_time = time.time()

            aoi.check_box(end_points)
            aoi.update_info(frame)
            cam.people_tracking(personContours)
            final_data = cam.persons

            #only pass the data while getting detection
            if len(personContours) != 0:
                '''Convert attributes into dataframe for processing'''
                if int(time.time() - start_store_time) > sample_interval:
                    for i in range(len(final_data)):
                        list = final_data[i].getAttris()
                        list.insert(0,str(datetime.now().strftime("%x-%X")))
                        stored_data.loc[len(stored_data)] = list
                    start_store_time = time.time()

                '''Transmit process data at every 5 second'''
                if int(time.time() - start_transmit_time) > transmit_interval:
                    transmit_data(stored_data)
                    stored_data = pd.DataFrame(columns = ['gmt', 'pid', 'project', 'age', 'gender'])

                    start_transmit_time = time.time()

        # Draw performance stats
        inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
            "Inference time: {:.3f} ms".format(det_time * 1000)
        render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
        async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
            "Async mode is off. Processing request {}".format(cur_request_id)
        statistics_population ="Total {} people pass the screen".format(cam.entered)
        cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
        cv2.putText(frame, statistics_population, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (10, 10, 200), 1)

        render_start = time.time()
        cv2.imshow("Detection Results", frame)
        render_end = time.time()
        render_time = render_end - render_start

        key = cv2.waitKey(1)
        if key == 27:
            break
        if (9 == key):
            is_async_mode = not is_async_mode
            log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id

        '''if frames container larger than 5 frames pop last'''
        if len(frames) > 5:
            numToPop = len(frames)-5
            for _ in range(numToPop): frames.pop();
    cv2.destroyAllWindows()
    del exec_net
    del plugin

if __name__ == '__main__':
    sys.exit(main() or 0)
