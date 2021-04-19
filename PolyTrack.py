import os
import time
import sys
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from core.functions import *
from polytrack.track import track
from polytrack.bg_subtraction import foreground_changes
from polytrack.record import record_track, complete_tracking
from polytrack.config import pt_cfg
from polytrack.flowers import record_flowers
from polytrack.general import *
import cv2
import numpy as np
# import warnings; warnings.simplefilter('ignore')
from datetime import datetime
from absl import app
processing_details= open(str(pt_cfg.POLYTRACK.OUTPUT)+ "videoprocessing_details.txt","w+")


def main(_argv):
    start_time = datetime.now()
    start_time_py = time.time()
    print("Start:  " + str(start_time))
    nframe = 0
    total_frames = 0
    flowers_recorded = False
    predicted_position =[]

    video_list = get_video_list(pt_cfg.POLYTRACK.INPUT_DIR, pt_cfg.POLYTRACK.VIDEO_EXT)

    for video_name in video_list:
        
        print('===================' + str(video_name) + '===================')
        processing_details.write("Name: "+str(video_name)+"\n"+ "Start: "+ str(start_time)+ "\n")

        video = str(pt_cfg.POLYTRACK.INPUT_DIR) + str(video_name)

        # begin video capture
        try:
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        
        width, height, video_frames = get_video_details(vid)
        processing_details.write("video_frames: "+str(video_frames)+"\n")
        
        if pt_cfg.POLYTRACK.SIGHTING_TIMES: 
            try:
                pt_cfg.POLYTRACK.VIDEO_START_TIME = get_video_start_time(video_name, total_frames)
            except:
                print('Invalied filename format. Try renaming the file or setting the value of SIGHTING_TIMES to False in Configuration file')
                pt_cfg.POLYTRACK.SIGHTING_TIMES = False

        total_frames += video_frames

        if not flowers_recorded: flowers, flowers_recorded = record_flowers(vid, video_name)

        while True:
            return_value, frame = vid.read()
            if return_value:
                nframe += 1

                # if not flowers_recorded: flowers_recorded = record_flowers()

                idle = check_idle(nframe, predicted_position)
                insectsBS =  foreground_changes(frame, width, height, nframe, idle)
                associated_det_BS, associated_det_DL, missing,new_insect = track(frame, predicted_position, insectsBS)
                for_predictions = record_track(frame, nframe,associated_det_BS, associated_det_DL, missing, new_insect, idle)
                predicted_position = predict_next(for_predictions)

                fps = round(nframe/ (time.time() - start_time_py),2)
                print(str(nframe) + ' out of ' + str(total_frames) + ' frames processed | ' + str(fps) +' FPS     ' , end='\r')


                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
            else:
                print()
                print('Video has ended')
                print(pt_cfg.POLYTRACK.RECORDED_DARK_SPOTS)
                break

        if not pt_cfg.POLYTRACK.CONTINUOUS_VIDEO:
            complete_tracking(predicted_position)
            predicted_position =[]
            pt_cfg.POLYTRACK.RECORDED_DARK_SPOTS = []
            flowers_recorded = False

    cv2.destroyAllWindows()
    complete_tracking(predicted_position)
    end_time = datetime.now()
    print()
    print("End:  " + str(end_time))
    print("Processing Time:  " + str(end_time-start_time))
    processing_details.write("End:  " + str(end_time)+ "\n" + "Processing Time:  " + str(end_time-start_time)+ "\n")
    processing_details.close()




if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
