import numpy as np
from polytrack.general import cal_dist
# import polytrack.bg_subtraction as polytrack_bgs
import itertools as it
from scipy.optimize import linear_sum_assignment
from polytrack.deep_learning import detect_deep_learning
from polytrack.config import pt_cfg




max_dist_bs = pt_cfg.POLYTRACK.MAX_DIST_BS #Maximum detection threshold for background subtraction-based detection
max_dist_dl = pt_cfg.POLYTRACK.MAX_DIST_DL
max_bg_changes = pt_cfg.POLYTRACK.MAX_BG_CHANGES #Threshold value of background changes (if background changes exceeds this value, deep learning-based detection will be used)




def track(frame, _predicted_position, _insectsBS):
    _predictions = _predicted_position
    _detections_BS = _insectsBS
    
    if(len(_insectsBS)<=max_bg_changes):
        _associated_det_BS, _missing_BS, _not_associated_BS = associate_detections_BS(_insectsBS, _predicted_position)
    else:
        _missing_BS = [i[0] for i in _predicted_position]
        _associated_det_BS = []
        _not_associated_BS  =[]
         
  
    if (run_DL(_missing_BS, _not_associated_BS)):
        
        _predictions_DL = np.zeros(shape=(0,3))
        
        for pred in np.arange(len(_missing_BS)):
            _predictions_DL = np.vstack([_predictions_DL,([row for row in _predictions if _missing_BS[pred] == row[0]])])
            
        _detections_DL = detect_deep_learning(frame)
        
        _associated_det_DL, _missing, _pot_new_insect = associate_detections_DL(_detections_DL, _predictions_DL)
        _new_insect = verify_new_insect(_pot_new_insect,_associated_det_BS)
    else:
        _associated_det_DL, _missing, _new_insect =[],[], []
        

        
    return _associated_det_BS, _associated_det_DL, _missing, _new_insect

def run_DL(_missing_BS, _not_associated_BS):
    _run_DL =False
    n_missing = len(_missing_BS)
    n_not_assciated_BS = len(_not_associated_BS)
    
#     print('run_DL', nframe, n_missing, n_not_assciated_BS)
    if((n_missing>0) or (n_not_assciated_BS>0)):
        _run_DL = True
    else:
        _run_DL = False
        
    return _run_DL
        


def verify_new_insect(_pot_new_insect,_associated_det_BS):
    recorded_BS = []
    for bsdet in _associated_det_BS:
        for pni in np.arange(len(_pot_new_insect)):
            dist = cal_dist(bsdet[1],bsdet[2],int(float(_pot_new_insect[pni][0])),int(float(_pot_new_insect[pni][1])))
            if dist <= max_dist_bs:
                recorded_BS.append(pni)

    _pot_new_insect = np.delete(_pot_new_insect,recorded_BS,0 )
    
    return _pot_new_insect


def associate_detections_DL(_detections, _predictions):
    _missing = [] 
    _assignments = Hungarian_method(_detections, _predictions)
    _insects = [i[0] for i in _predictions]
    
    _not_associated = np.zeros(shape=(0,5))
    for _nass in (_assignments[len(_insects):]):
        _not_associated = np.vstack([_not_associated,(_detections[_nass])])
                               
    
    _associations_DL = np.zeros(shape=(0,6))
    for ass in np.arange(len(_insects)):
        _record = _assignments[ass]

        if (_record <= len(_detections)-1):
            _xc, _yc, _area, _lable, _conf = _detections[_assignments[ass]][0],_detections[_assignments[ass]][1],_detections[_assignments[ass]][2],_detections[_assignments[ass]][3],_detections[_assignments[ass]][4]
            _dist = cal_dist(_xc,_yc,_predictions[ass][1],_predictions[ass][2])
            if(_dist>max_dist_dl):
                _missing.append(_predictions[ass][0])
            else:
                _associations_DL = np.vstack([_associations_DL,(_predictions[ass][0],_detections[_assignments[ass]][0],_detections[_assignments[ass]][1],_detections[_assignments[ass]][2],_detections[_assignments[ass]][3],_detections[_assignments[ass]][4])])
                
        else:
            _missing.append(_predictions[ass][0])

    return _associations_DL, _missing, _not_associated

    
#not associated = when there are more detections than predictions
#missing = when there are less detections than predictions. 1) less number of insects detected, 2) detected insects cannot be associated with a detection
def associate_detections_BS(_detections, _predictions):
    _assignments = Hungarian_method(_detections, _predictions)
    _missing = []
    _insects = [i[0] for i in _predictions]
    _not_associated = np.zeros(shape=(0,3))
    for _nass in (_assignments[len(_insects):]):
        _not_associated = np.vstack([_not_associated,(_detections[_nass])])       
                              
    _associations_BS = np.zeros(shape=(0,4))
    for ass in np.arange(len(_insects)):
        _record = _assignments[ass]
        if (_record <= len(_detections)-1):
            _xc, _yc, _area = _detections[_assignments[ass]][0],_detections[_assignments[ass]][1],_detections[_assignments[ass]][2]
            _dist = cal_dist(_xc,_yc,_predictions[ass][1],_predictions[ass][2])
            if(_dist>max_dist_bs):
                _missing.append(_predictions[ass][0])
            else:
                _associations_BS = np.vstack([_associations_BS,(int(float(_predictions[ass][0])),int(_xc), int(_yc), int(_area))])
        else:
            _missing.append(_predictions[ass][0])

    return _associations_BS, _missing,_not_associated


def Hungarian_method(_detections, _predictions):
    num_detections, num_predictions = len(_detections), len(_predictions)
    mat_shape = max(num_detections, num_predictions)
    hun_matrix = np.full((mat_shape, mat_shape),0)
    for p in np.arange(num_predictions):
        for d in np.arange(num_detections):
            hun_matrix[p][d] = cal_dist(_predictions[p][1],_predictions[p][2],_detections[d][0],_detections[d][1])
    
    row_ind, col_ind = linear_sum_assignment(hun_matrix)

    return col_ind  
