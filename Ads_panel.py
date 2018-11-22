import cv2
import screeninfo
import os
import time
# from moviepy.editor import *
# import pygame
import pyglet
from subprocess import call
from threading import Thread
import _thread
from multiprocessing import Process, Queue

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

    def set_window_name(self, new_name):
        self.window_name = new_name

    def update_ads_path(self, new_path):
        self.path = new_path

    def set_ads_refresh_time(self, new_time_interval):
        self.time_interval = new_time_interval

    def set_proj_id(self, file):
        self.cur_id = file

    def frameDetections(self):
        return self.video.read()

    def get_proj_id(self):
        return self.cur_id

    def get_proj_id(self):
        return self.cur_id

    def get_key_name(self):
        key_names = os.listdir(self.path)
        self.key_name = [os.path.splitext(key)[0] for key in key_names]
        return self.key_name

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

    '''frames capturing and manage threading functions'''
    def capture_frame(self):
        #Must define in here
        self.video = cv2.VideoCapture(os.path.join(self.path, "adv1.mp4"))
        while 1:
            _, frame = self.video.read()
            self.frames.insert(0, frame)

    def frames_manage(self):
        while 1:
            length = len(self.frames)
            if length > 10:
                self.frames.pop(-1)
                
    def play_audio(self):
        music = pyglet.resource.media("adv1.mp3")
        music.play()
        pyglet.app.run()

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

            key = cv2.waitKey(29)
            if key == 27:
                break

        # clip = VideoFileClip(os.path.join(self.path, 'adv1.mp4'))
        # clip.preview()
        # pygame.quit()

    def call_cmd(self):
        call(["xdg-open", str(os.path.join(self.path, "adv1.mp4"))])
