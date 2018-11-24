import cv2
import screeninfo
import os, sys
import time
# from moviepy.editor import *
import numpy as np
from subprocess import call
import _thread
from multiprocessing import Process, Manager

class Advertisment (object):
    def __init__(self, path):
        self.path = path
        self.video = "Video path"
        self.window_name = 'Ads'
        self.screen = screeninfo.get_monitors()[0]
        self.time_interval = 10
        self.cur_id = None
        self.key_name = []
        self.frames = []

##########################"""Normal Fucntion"""#################################

    def set_ads_refresh_time(self, new_time_interval):
        self.time_interval = new_time_interval

    def set_proj_id(self, file):
        self.cur_id = file

    """pass the key to the smart panel"""
    def get_key_name(self):
        key_names = os.listdir(self.path)
        self.key_name = [os.path.splitext(key)[0] for key in key_names]
        return self.key_name

    def write_queue(self, id, q):
        q.put(id)
#########################"""Display Ads"""######################################
    """Picture display"""
    def display_ads(self):
        cv2.namedWindow(self.window_name,cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(self.window_name, self.screen.x, self.screen.y)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
        files = os.listdir(self.path)
        while 1:
            for file in files:
                id = os.path.splitext(file)[0]
                self.set_proj_id(id)
                img_path = os.path.join(self.path, file)
                img = cv2.imread(img_path)
                cv2.imshow(self.window_name, img)
                start = time.time()
                while 1:
                    if (time.time() - start) > self.time_interval:
                        break
                # cv2.waitKey(1)

    """frames capturing and manage threading functions"""
    def capture_frame(self):
        #Must define in here
        while 1:
            files = os.listdir(self.path)
            for file in files:
                if file.endswith(".mp4"):
                    self.video = cv2.VideoCapture(os.path.join(self.path, file))
                    video_frames_len = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
                    frame_counter = 0
                    while frame_counter < video_frames_len:
                        _, frame = self.video.read()
                        self.frames.insert(0, frame)
                        frame_counter += 1

    def frames_manage(self):
        while 1:
            length = len(self.frames)
            if length > 10:
                self.frames.pop(-1)

    def play_audio(self):
        music = pyglet.resource.media("adv2.mp3")
        music.play()
        pyglet.app.run()

    """This is using Opencv to open video"""
    def display_ads_video(self):

        cv2.namedWindow(self.window_name,cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(self.window_name, self.screen.x, self.screen.y)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

        _thread.start_new_thread(self.capture_frame,())
        _thread.start_new_thread(self.frames_manage,())

        fps = 10

        while 1:
            frame_start_time = time.time()
            try:
                frame = self.frames[0]
            except Exception as e:
                continue
            cv2.imshow(self.window_name,frame)
            frame_time = time.time() - frame_start_time
            delay_time = int((1/fps-frame_time)*1000)
            # print(delay_time)
            if delay_time <= 0:
                delay_time = 1

            key = cv2.waitKey(delay_time)
            if key == 27:
                break
        # print(os.path.join(self.path, 'adv1.mp4'))
        # clip = VideoFileClip(os.path.join(self.path, 'adv1.mp4'))
        # clip.preview()
        # pygame.quit()

    """This is using ffmepg to open video"""
    def call_cmd(self, q):

        files = os.listdir(self.path)
        videos = [file for file in files if file.endswith(".mp4") or file.endswith(".mkv") or file.endswith(".webm")]

        cv2.namedWindow(self.window_name,cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(self.window_name, self.screen.x, self.screen.y)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

        while 1:

            #black canvas
            img = cv2.imread("bg.png")
            cv2.imshow(self.window_name, img)
            key = cv2.waitKey(100)

            for v in videos:
                id = os.path.splitext(v)[0] #set video id
                # self.set_proj_id(id)
                self.write_queue(id,q)
                v = os.path.join(self.path, v)
                call(["ffplay", str(v), '-autoexit', '-fs', '-loglevel', 'quiet']) #full screen
