import cv2
import screeninfo
import os
import time

class Advertisment (object):
    def __init__(self, path):
        self.path = path
        self.window_name = 'Ads'
        self.screen = screeninfo.get_monitors()[0]
        self.time_interval = 10
        self.cur_id = "-1"
        self.key_name = []
    def set_window_name(self, new_name):
        self.window_name = new_name
    def update_ads_path(self, new_path):
        self.path = new_path
    def set_ads_refresh_time(self, new_time_interval):
        self.time_interval = new_time_interval
    def set_proj_id(self, file):
        self.cur_id = file
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
