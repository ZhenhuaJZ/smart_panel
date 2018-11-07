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
import math
import _thread

persons = []
pid = 1
entered = 0
exited = 0
frames = []

check_key = True
last_checklist = {}
start_time = time.time()



def capture_frame(cam):
    while cam.video.isOpened():
        frames.insert(0, cam.frameDetections());

def frames_manage():
    print(len(frames))
    if len(frames) > 5:
        frames.pop(-1)

class Person:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
    def getId(self):
        return self.id
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def updateCoords(self, newX, newY):
        self.x = newX
        self.y = newY


class Camera(object):
    def __init__(self, input):
        self.input = input
        self.video = cv2.VideoCapture(self.input)
        self.w = int(self.video.get(3))
        self.h = int(self.video.get(4))
        self.rangeLeft = int(1*self.w/8)
        self.rangeRight = int(7*self.w/8)
        self.midLine = int(4*self.w/8)
        self.counter= []
    def __del__(self):
        self.video.release()

    def frameDetections(self):
        return self.video.read()

    def people_tracking(self, rects):
        global pid
        global entered
        global check_key
        global last_checklist

        for xCenter, yCenter, w, h in rects:
            new = True
            inActiveZone= xCenter in range(self.rangeLeft,self.rangeRight)

            #check every ppl location in every 5s
            if int(time.time() - start_time) % 10 == 0 and check_key == True:
                cur_checklist = {}
                cur_indexlist = {}
                for index, p in enumerate(persons):
                    fix_dist = math.sqrt((p.getX())**2 + (p.getY())**2)
                    cur_checklist["pid" + str(p.getId())] = fix_dist
                    cur_indexlist["pid" + str(p.getId())] = index
                # print(last_checklist)

                check_key = False

                for key in last_checklist.keys():
                    if cur_checklist[key] == last_checklist[key]:
                        print("pop : ", key)
                        persons.pop(cur_indexlist[key])
                        cur_checklist.pop(key)
                        print(cur_checklist)

                last_checklist = cur_checklist

            elif int(time.time() - start_time) % 5 != 0:
                check_key = True

            for index, p in enumerate(persons):
                dist = math.sqrt((xCenter - p.getX())**2 + (yCenter - p.getY())**2)

                if dist <= w and dist <=h:
                    if inActiveZone:
                        new = False
                        if p.getX() < self.midLine and  xCenter >= self.midLine:
                            print("[INFO] person {} going left ".format(str(p.getId())))
                            # entered += 1 #count ppl in and out
                        if p.getX() > self.midLine and  xCenter <= self.midLine:
                            print("[INFO] person {} going right ".format(str(p.getId())))
                            # exited += 1
                        p.updateCoords(xCenter,yCenter)
                        break
                    else:
                        print("[INFO] person {} removed ".format(str(p.getId())))
                        entered += 1 #count ppl in the area
                        try:
                            last_checklist.pop("pid" + str(p.getId())) #pop last frame dict id --- > 1st then pop list
                        except Exception as e:
                            pass
                        persons.pop(index)
                        for p in persons:
                            print("left pid = ", p.getId())
                            print("------")

            #make sure the total persons number wont excess the bounding box
            if new == True and inActiveZone and len(persons) + 1 <= len(rects) :
                print("[INFO] new person " + str(pid))
                p = Person(pid, xCenter, yCenter)
                persons.append(p)
                pid += 1


def frame_process(frame, n, c, h, w):
    in_frame = cv2.resize(frame, (w, h))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n, c, h, w))
    return in_frame

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
    parser.add_argument("-m_landmark", "--model_landmark", required=True, type=str)
    # parser.add_argument("-m_gaze", "--model_gaze", required=True, type=str)
    parser.add_argument("-m_hp", "--model_hp", required=True, type=str)

    parser.add_argument("-d", "--device", default="CPU",type=str) # CPU, GPU, FPGA or MYRIAD
    parser.add_argument("-d_fc", "--device_fc", default="CPU",type=str)
    parser.add_argument("-d_ag", "--device_ag", default="CPU",type=str)
    parser.add_argument("-d_attri", "--device_attri", default="CPU", type=str)
    parser.add_argument("-d_landmak", "--device_landmark", default="CPU",type=str)
    # parser.add_argument("-d_gaze", "--device_gaze", default="CPU",type=str)
    parser.add_argument("-d_hp", "--device_hp", default="CPU",type=str)


    parser.add_argument("-pt", "--prob_threshold", default=0.55, type=float)
    parser.add_argument("-pt_face", "--prob_threshold_face", default=0.65, type=float)

    parser.add_argument("--labels", help="Labels mapping file", default=None, type=str)

    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    return parser


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

    # modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f)); //The first must be (0,0,0) while using POSIT
    # modelPoints.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));
    # modelPoints.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));
    # modelPoints.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));
    # modelPoints.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));
    # modelPoints.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));

def eular_to_image(frame,eular_angle,center,scale):
    ##### Define camera property
    cam_matrix = np.array([[frame.shape[1], 0, center[0]],
                   [0, frame.shape[1], center[1]],
                   [0, 0, 1]], dtype = np.float64)

    ##### Convert from degree to radian
    eular_angle = eular_angle * 1.57/180
    # print(eular_angle)
    ##### convert the eular_angle to each rotational axis matrix
    rx = np.array([[1,0,0],
                    [0,math.cos(eular_angle[1]),-math.sin(eular_angle[1])],
                    [0,math.sin(eular_angle[1]),math.cos(eular_angle[1])]])

    ry = np.array([[math.cos(eular_angle[0]), 0, -math.sin(eular_angle[0])],
                    [0, 1, 0],
                    [math.sin(eular_angle[0]), 0, math.cos(eular_angle[0])]])

    rz = np.array([[math.cos(eular_angle[2]), -math.sin(eular_angle[2]), 0],
                    [math.sin(eular_angle[2]), math.cos(eular_angle[2]), 0],
                    [0, 0, 1]])
    r_mat = np.matmul(np.matmul(rz,ry),rx)
    # r_mat = rz.dot(ry).dot(rx)
    # print(r_mat)
    o = np.array([0,0,frame.shape[1]])
    xAxis = r_mat.dot(np.array([scale,0,0]))+o
    yAxis = r_mat.dot(np.array([0,-scale,0]))+o
    zAxis = r_mat.dot(np.array([0,0,-scale]))+o
    zAxis2 = r_mat.dot(np.array([0,0,scale]))+o

    x_p = cam_matrix.dot(xAxis)/xAxis[2]
    x_p = x_p.astype(np.int)
    y_p = cam_matrix.dot(yAxis)/yAxis[2]
    y_p = y_p.astype(np.int)
    z_p = cam_matrix.dot(np.transpose(zAxis))/zAxis[2]
    z_p = z_p.astype(np.int)
    z_p2 = cam_matrix.dot(zAxis2)/zAxis2[2]
    z_p2 = z_p2.astype(np.int)
    center = center.astype(np.int)

    cv2.line(frame,(center[0],center[1]),(x_p[0],x_p[1]),[255,0,0],4)
    cv2.line(frame,(center[0],center[1]),(y_p[0],y_p[1]),[0,255,0],4)
    cv2.line(frame,(center[0],center[1]),(z_p[0],z_p[1]),[0,0,255],4)
    # cv2.line(frame,(center[0],center[1]),(z_p2[0],z_p2[1]),[0,0,255],4)




def landmark_3d_to_2d(img, landmark_2d):
    cam_matrix = np.array([[img.shape[1], 0, img.shape[1]/2],
                   [0, img.shape[1], img.shape[0]/2],
                   [0, 0, 1]], dtype = np.float64)
    # rot_mat = np.zeros(4,4)
    # trans_mat = np.zeros(4,4)
    dist_coeffs = np.zeros((4,1))
    landmark_2d = landmark_2d.astype(np.float64)
    # print(landmark_2d)
    # print(get_3d_landmark().shape)
    # print(landmark_2d.shape)
    retval, rvec, tvec = cv2.solvePnP(get_3d_landmark().astype(np.float64), landmark_2d, cam_matrix, dist_coeffs)

    end_point_3d = np.array([[0,0,2000]]);

    end_point_2d, _ = cv2.projectPoints(end_point_3d.astype(np.float32), rvec, tvec, cam_matrix, dist_coeffs)

    landmark_2d = landmark_2d.astype(np.int)
    # cv2.line(img,(landmark_2d[2,:][0],landmark_2d[2,:][1]),(end_point_2d[0][0][0],end_point_2d[0][0][1]),[255,0,0],4)

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

    model_xml_landmark = args.model_landmark
    model_bin_landmark = os.path.splitext(model_xml_landmark)[0] + ".bin"

    # model_xml_gaze = args.model_gaze
    # model_bin_gaze = os.path.splitext(model_xml_gaze)[0] + ".bin"

    model_xml_hp = args.model_hp
    model_bin_hp = os.path.splitext(model_xml_hp)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    plugin_fc = IEPlugin(device=args.device_fc, plugin_dirs=args.plugin_dir)
    plugin_attri = IEPlugin(device=args.device_attri, plugin_dirs=args.plugin_dir)
    plugin_ag = IEPlugin(device=args.device_ag, plugin_dirs=args.plugin_dir)
    plugin_landmark = IEPlugin(device=args.device_landmark, plugin_dirs=args.plugin_dir)
    # plugin_gaze = IEPlugin(device=args.device_gaze, plugin_dirs=args.plugin_dir)
    plugin_hp = IEPlugin(device=args.device_hp, plugin_dirs=args.plugin_dir)

    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)

    # Read IR
    log.info("Reading IR...")
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    net_fc = IENetwork.from_ir(model=model_xml_fc, weights=model_bin_fc)
    net_attri = IENetwork.from_ir(model=model_xml_attri, weights=model_bin_attri)
    net_ag = IENetwork.from_ir(model=model_xml_ag, weights=model_bin_ag)
    net_landmark = IENetwork.from_ir(model=model_xml_landmark, weights=model_bin_landmark)
    # net_gaze = IENetwork.from_ir(model=model_xml_gaze, weights=model_bin_gaze)
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
    input_blob_landmark = next(iter(net_landmark.inputs))
    out_blob_landmark = next(iter(net_landmark.outputs))
    # input_blob_gaze = next(iter(net_gaze.inputs))
    # out_blob_gaze = next(iter(net_gaze.outputs))
    input_blob_ag = next(iter(net_ag.inputs))
    input_blob_hp = next(iter(net_hp.inputs))

    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)
    exec_net_face = plugin.load(network=net_fc, num_requests=2)
    exec_net_attri = plugin_ag.load(network=net_attri, num_requests=2)
    exec_net_age = plugin_ag.load(network=net_ag, num_requests=2)
    exec_net_landmark = plugin_landmark.load(network=net_landmark, num_requests=2)
    # exec_net_gaze = plugin_gaze.load(network=net_gaze, num_requests=2)
    exec_net_hp = plugin_landmark.load(network=net_hp, num_requests=2)

    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob]
    n_fc, c_fc, h_fc, w_fc = net_fc.inputs[input_blob_fc]
    n_attri, c_attri, h_attri, w_attri = net_attri.inputs[input_blob_attri]
    n_ag, c_ag, h_ag, w_ag = net_ag.inputs[input_blob_ag]
    n_lm, c_lm, h_lm, w_lm = net_landmark.inputs[input_blob_landmark]
    n_hp, c_hp, h_hp, w_hp = net_hp.inputs[input_blob_hp]
    del net, net_ag, net_fc, net_attri, net_landmark, net_hp

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
    last_frame_p_num = 0
    statistics_person = 0
    frame_detection_interval = 0 #ms
    detection_end_time = 0
    cam = Camera(input_stream)
    camera_are = cam.w * cam.h
    ####@ Define array for landmarks
    # landmark_2d_array = np.zeros((5,2))
    # landmark_2d_array = []
    try:
        _thread.start_new_thread(capture_frame,(cam,))
        _thread.start_new_thread(frames_manage,())
    except:
        raise

    while 1:
        try:
            ret, frame = frames[0];
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

        if exec_net.requests[cur_request_id].wait(-1) == 0 and exec_net_face.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            # Parse detection results of the current request
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            res_fc = exec_net_face.requests[cur_request_id].outputs[out_blob_fc]

            cur_frame_p_num = 0
            personContours = []

            for obj, obj_fc in zip(res[0][0], res_fc[0][0]):
                # Draw only objects when probability more than specified threshold
                if obj_fc[2] > args.prob_threshold_face:
                    #if no person skip
                    xmin_fc = int(obj_fc[3] * cam.w)
                    ymin_fc = int(obj_fc[4] * cam.h)
                    xmax_fc = int(obj_fc[5] * cam.w)
                    ymax_fc = int(obj_fc[6] * cam.h)

                    width_fc = xmax_fc-xmin_fc
                    height_fc = ymax_fc-ymin_fc
                    face_area = width_fc * height_fc

                    #get rid off small face
                    # DEBUG: # print(camera_are/face_area)
                    if camera_are/face_area < 100:
                        # print(camera_are/face_area)
                        cv2.rectangle(frame, (xmin_fc, ymin_fc), (xmax_fc, ymax_fc), (10, 10, 200), 2)
                        #central of the face
                        xCenter_fc = int(xmin_fc + (width_fc)/2)
                        yCenter_fc = int(ymin_fc + (height_fc)/2)
                        rect = (xCenter_fc, yCenter_fc, width_fc, height_fc)

                        # Face centre point
                        cv2.circle(frame, (xCenter_fc, yCenter_fc), 5, (0,255,0), 3)

                        personContours.append(rect)
                        cam.counter = personContours

                        try:
                            #crop face
                            face = frame[ymin_fc:ymax_fc,xmin_fc:xmax_fc] #crop the face
                            in_face = frame_process(face, n_ag, c_ag, h_ag, w_ag)
                            res_ag = exec_net_age.infer({input_blob_ag : in_face})
                            sex = np.argmax(res_ag['prob'])
                            age = int(res_ag['age_conv3']*100)

                            ##### head pose
                            in_face_hp = frame_process(face, n_hp, c_hp, h_hp, w_hp)
                            res_hp = exec_net_hp.infer({input_blob_hp : in_face_hp})
                            head_pose = []
                            # print(res_hp)
                            for key in res_hp.keys():
                                # print(key)
                                head_pose.append(res_hp[key][0])
                                # print(res_hp[key][0])
                            eular_to_image(frame,np.asarray(head_pose),np.array([xCenter_fc, yCenter_fc]), 300)


                            ##### draw landmark
                            face_landmark = frame_process(face, n_lm, c_lm, h_lm, w_lm)
                            res_landmark = exec_net_landmark.infer({input_blob_landmark : face_landmark})[out_blob_landmark][0].reshape(-1)
                            _w_fc = xmax_fc - xmin_fc
                            _h_fc = ymax_fc - ymin_fc

                            ##### landmark, left_eye, right_eye, nose, lip left, lip right
                            x_lm = [xmin_fc + x * _w_fc for x in res_landmark[0::2]]
                            # print('*************')
                            y_lm = [ymin_fc + y * _h_fc for y in res_landmark[1::2]]
                            landmark_2d = np.stack((np.asarray(x_lm),np.asarray(y_lm)), axis = -1)
                            ##### Data filtering by average several landmark points
                            # landmark_2d_array = np.concatenate((landmark_2d_array,landmark_2d), axis = 2)
                            # landmark_2d_array.append(landmark_2d)
                            # print(landmark_2d_array)
                            # landmark_2d = np.mean(np.array(landmark_2d_array), axis = 0)
                            # print(landmark_2d, landmark_2d.shape)
                            # end_point_2d = landmark_3d_to_2d(frame, landmark_2d).astype(np.int)
                            # if len(landmark_2d_array) > 3:
                            #     landmark_2d_array.pop(0)
                            # x_lm = [xmin_fc + x * _w_fc for x in res_landmark[0:4:2]]
                            # y_lm = [ymin_fc + y * _h_fc for y in res_landmark[1:5:2]]

                            eyes = []
                            for _x_lm, _y_lm in zip(x_lm, y_lm):
                                cv2.circle(frame, (int(_x_lm), int(_y_lm)), 3, (125,255,0), 2)
                                # eye_y is 1/8 of the face, eye_x is 1/4 of the face
                                eye_ymin = int(_y_lm - 1/7 * _h_fc)
                                eye_ymax = int(_y_lm + 1/7 * _h_fc)
                                eye_xmin = int(_x_lm - 1/3 * _w_fc )
                                eye_xmax = int(_x_lm + 1/3 * _w_fc)
                                cv2.rectangle(frame, (eye_xmin, eye_ymin), (eye_xmax, eye_ymax), (125,255,0), 1)
                                eyes.append(frame[eye_ymin : eye_ymax, eye_xmin : eye_xmax])

                            ##### face gaze model
                            right_eye = eyes[0]
                            left_eye = eyes[1]
                            #### binary face
                            height, width, _ = frame.shape
                            binary_face = np.ones((height,width))
                            mask = cv2.rectangle(binary_face, (xmin_fc, ymin_fc), (xmax_fc, ymax_fc), 0, -1)
                            # binary_face = cv2.resize(mask, (25, 25)).reshape(1,-1,1,1)
                            # in_right_eye = frame_process(right_eye, 1, 3, 224, 224)
                            # in_left_eye = frame_process(left_eye, 1, 3, 224, 224)
                            # in_face_gaze = frame_process(face, 1, 3, 224, 224)
                            # res_gaze = exec_net_gaze.infer({"image_left": in_left_eye, "image_right":in_right_eye, "image_face": in_face_gaze, "face_grid":binary_face})['fc3'][0]
                            # print(res_gaze)
                            # points = [width/2 + (1280/30)*res_gaze[0], height/2 - (720/15)*res_gaze[1]]
                            # cv2.circle(frame, (int(points[0]), int(points[1])), 2, (125,255,0), 1)
                            cv2.putText(frame, str(sex) +", "+str(age), (xmin + 100, ymin - 7),
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
                                # rect = (xCenter, yCenter, width, height)
                                cv2.circle(frame, (xCenter, yCenter), 5, (0,255,0), 3)

                                try:
                                    person = frame[ymin:ymax,xmin:xmax]
                                    #person attribution
                                    in_attri = frame_process(person, n_attri, c_attri, h_attri, w_attri)
                                    res_attri = exec_net_attri.infer({input_blob_attri : in_attri})[out_blob_attri][0].reshape(-1)
                                except Exception as e:
                                    pass

                                # print(res_attri)

                    #det_label = labels_map[class_id] if labels_map else str(class_id)
                    # cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                    #             cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
                    detection_end_time = time.time()

            cam.people_tracking(cam.counter)
        # Draw performance stats
        inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
            "Inference time: {:.3f} ms".format(det_time * 1000)
        render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
        async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
            "Async mode is off. Processing request {}".format(cur_request_id)
        statistics_population ="Total {} people pass the screen".format(entered)
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
        if len(frames) > 5:
            numToPop = len(frames)-5
            for _ in range(numToPop): frames.pop();
    cv2.destroyAllWindows()
    del exec_net
    del plugin


if __name__ == '__main__':
    sys.exit(main() or 0)
