import numpy as np
import cv2
from polytrack.config import pt_cfg
import os
import datetime as dt



def cal_dist(x,y,px,py):
    type(x)
    edx = float(x) - float(px)
    edy = float(y) - float(py)
    error = np.sqrt(edx**2+edy**2)
    
    return error


def predict_next(_for_predictions):
    
    
    _predicted = []
    for _insect in _for_predictions:
        _insect_num = _insect[0]
        _x0 = float(_insect[1])
        _y0 = float(_insect[2])
        _x1 = float(_insect[3])
        _y1 = float(_insect[4])
        
               
        Dk1 = np.transpose([_x0, _y0])
        Dk2 = np.transpose([_x1, _y1])
        A = [[2,0,-1,0],  [0,2,0,-1]]
        Dkc = np.concatenate((Dk1,Dk2))
        
#         print(Dk1,Dk2,Dkc)
        Pk = np.dot(A,Dkc.T)
        
        _predicted.append([_insect_num, Pk[0], Pk[1]])
        
    
    return _predicted


def check_idle(_nframe, _predicted_position):
    if ((_nframe >pt_cfg.POLYTRACK.INITIAL_FRAMES) and (bool(_predicted_position) == False)):
        _idle = True

    else:
        _idle=False
        

        
    return _idle

def get_video_details(vid):
    pt_cfg.POLYTRACK.FRAME_WIDTH = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    pt_cfg.POLYTRACK.FRAME_HEIGHT = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #pt_cfg.POLYTRACK.FPS = int(vid.get(cv2.CAP_PROP_FPS))
    pt_cfg.POLYTRACK.FRAME_COUNT = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    pt_cfg.POLYTRACK.FPS = 30
    #pt_cfg.POLYTRACK.FRAME_COUNT = 9000

    print('Video dimensions: ', pt_cfg.POLYTRACK.FRAME_WIDTH, ' x ', pt_cfg.POLYTRACK.FRAME_HEIGHT)
    print('Frame rate: ', pt_cfg.POLYTRACK.FPS, 'fps')
    print('Video length: ', round(pt_cfg.POLYTRACK.FRAME_COUNT), 'frames')


    return pt_cfg.POLYTRACK.FRAME_WIDTH, pt_cfg.POLYTRACK.FRAME_HEIGHT, pt_cfg.POLYTRACK.FRAME_COUNT

def get_video_list(directory, video_extension):
    video_list = []
    for video in os.listdir(directory):
        if video.endswith(video_extension):
            video_list.append(video)

    video_list.sort()

    return video_list

def get_video_start_time(video_name, total_frames):
    start_tim_str = dt.datetime.strptime(video_name.split('_')[3].split('.')[0], '%H%M%S').time()
    video_start_time = [start_tim_str, total_frames]

    return video_start_time


def cal_abs_time(_nframe, video_start):
    current_frame_in_video = _nframe - video_start[1]
    time_in_video = str(dt.timedelta(seconds=round(current_frame_in_video/pt_cfg.POLYTRACK.FPS))) 

    video_start_time = dt.datetime.strptime(str(video_start[0]), '%H:%M:%S')
    time_in_video = dt.datetime.strptime(time_in_video, '%H:%M:%S')
    time_zero = dt.datetime.strptime('00:00:00', '%H:%M:%S')
    absolute_time = ((video_start_time - time_zero + time_in_video).time())

    return absolute_time
