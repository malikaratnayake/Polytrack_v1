import cv2
import numpy as np
from polytrack.config import pt_cfg


fgbg = cv2.createBackgroundSubtractorKNN() #Use KNN background subtractor
fgbg_ld = cv2.createBackgroundSubtractorKNN() #Use KNN background subtractor
idle_width, idle_height = pt_cfg.POLYTRACK.LOWERES_FRAME_WIDTH, pt_cfg.POLYTRACK.LOWERES_FRAME_HEIGHT
insect_thresh = [pt_cfg.POLYTRACK.MIN_INSECT_AREA, pt_cfg.POLYTRACK.MAX_INSECT_AREA]

#Run background subtraction and related filteres
def Extract_cont(_frame,_idle):
    if (_idle):
        fgmask = fgbg_ld.apply(_frame)
    else:
        fgmask = fgbg.apply(_frame)
        
    median = cv2.medianBlur(fgmask,9)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(median,kernel,iterations = 1)
    contours, hier = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # cv2.imshow("bg", erosion)
    

    return contours



#Filter detected foreground changes to identify possible insects
def filter_contours(_nframe, _contours, _dim_factor):
    _insects = np.zeros(shape=(0,3))
    
    for c in _contours:
        (_x,_y), (_w, _h), _ = cv2.minAreaRect(c)
        _area = _w*_h*_dim_factor

        if _area > insect_thresh[0] and _area<insect_thresh[1]:
            _insects = np.vstack([_insects,(_x,_y,_area)])
            
#     print(_nframe, len(_insects), _insects)

    return _insects
    

#Detect foreground changes and filter them
def foreground_changes(_frame, width, height,_nframe, _idle):

    if(_idle == True):
        _frame_BS = cv2.resize(_frame, (idle_width, idle_height))
        _dim_factor = (width*height)/(idle_width*idle_height)
        
    else:
        _dim_factor = 1
        _frame_BS = _frame
        
    _contours = Extract_cont(_frame_BS, _idle)
    insects = filter_contours(_nframe, _contours, _dim_factor)

    
    return insects



#Check whether there are changes in the foreground and return boolean
def changes_in_foreground(_all_changes):
    bool_changes = False
    if (len(_all_changes) > 0):
        bool_changes = True
        
    return bool_changes
        
