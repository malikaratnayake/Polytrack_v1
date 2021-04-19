import cv2
import random
import colorsys
import numpy as np
import pandas as pd
from polytrack.config import pt_cfg
from polytrack.deep_learning import detect_deep_learning
from polytrack.record import track_frame
flowers = pd.DataFrame(columns = ['flower_num', 'x0','y0','radius','species','confidence'])



def record_flowers(vid, video_name):

    vid.set(1, 1)
    ret, frame = vid.read()
    flower_positions = sorted(detect_deep_learning(frame, True), key=lambda x: float(x[0]))

    for position in flower_positions:
        flower_num = len(flowers)
        _x = int(float(position[0]))
        _y = int(float(position[1]))
        _radius = int(float(position[2]))
        _species = position[3]
        _confidence = float(position[4])

        flower_record = [flower_num, _x, _y, _radius, _species, _confidence]
        flowers.loc[len(flowers)] = flower_record

        cv2.circle(track_frame, (_x,_y), _radius, (0,255,255), 4)
        cv2.putText(track_frame, 'F' + str(flower_num), (_x+_radius, _y), cv2.FONT_HERSHEY_DUPLEX , 0.7, (0,255,255), 1, cv2.LINE_AA)

    
    
    flowers.to_csv(str(pt_cfg.POLYTRACK.OUTPUT)+str(video_name)+'_flowers.csv', sep=',') #Save the csv file with insect track

    cv2.imwrite(str(pt_cfg.POLYTRACK.OUTPUT)+str(video_name)+'_flowers.png', cv2.add(frame,track_frame))

    print(str(len(flowers))+' flowers Recorded')

    

    return flowers, bool(len(get_flower_details()))

def get_flower_details():
    flower_details = flowers[flowers.columns[1:4]].values  

    return flower_details
