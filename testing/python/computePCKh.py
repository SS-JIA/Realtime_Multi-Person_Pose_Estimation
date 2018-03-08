#!/bin/python
import pandas as pd
import numpy as np
import h5py
import os
import os.path
import argparse

os.environ['GLOG_minloglevel'] = '3' 

import predictor

## Retrive the ground truth annotations
ground_truth = h5py.File('../sample_image/hockey1/annot.h5', 'r')

def createGroundTruthModel(img_num):
    ## Extract the keypoint annotations
    parts_gt = ground_truth['part'][img_num-1]
    
    ## Match format
    points_gt = np.array(parts_gt[[8,12,11,10,13,14,15,2,1,0,3,4,5]])
    points_gt = np.insert(points_gt, 0, [[np.nan, np.nan]], axis=0)
    nan_arr = np.repeat([[np.nan, np.nan]], 4, axis=0)
    points_gt = np.concatenate([points_gt, nan_arr, [parts_gt[9]]])

    ## Create PoseModel
    pose_model_gt = predictor.PoseModel(points_gt)
    return pose_model_gt

def detectPoseModels(img_num):
    ## Get image path
    img_path = '../sample_image/hockey1/' + str(img_num).zfill(3) + '.jpg'
    ## Run detection
    predictor_instance = predictor.OpenPosePredictor()
    pose_models_detected = predictor_instance.getPoseModels(img_path)

    return pose_models_detected

if __name__ == '__main__':
    ## Load in current record of PCKh scores
    if os.path.isfile('pckhrecord.h5'):
        pckh_record = pd.read_csv('pckhrecord.csv')
    else:
        pckh_record = pd.DataFrame(columns=['num_correct', 'num_total'])

    ## Determine where to start and end
    if len(pckh_record) == 0:
        start_img = 1
    else:
        start_img = pckh_record.index[-1]+1

    end_img = len(ground_truth['imgname']) + 1

    ## Compute PCKh for each image
    num_processed = 0
    for img_num in range(start_img, end_img):
        filename = ground_truth['imgname'][img_num-1]
        filename = ''.join(chr(int(i)) for i in filename)
        print("Processing {}...".format(filename))
        ## Create PoseModel
        pose_model_gt = createGroundTruthModel(img_num)
        ## Detect PoseModels
        pose_models_dt = detectPoseModels(img_num)
        print("\t{} poses detected!".format(len(pose_models_dt)))

        ## Calculate PCKh between PoseModels
        max_score = np.nan
        max_num_corr = np.nan
        max_num_total = np.nan
        for pose_model_dt in pose_models_dt:
            num_corr, num_total = predictor.computePCKh(pose_model_gt, pose_model_dt)
            if num_total != 0 and float(num_corr)/float(num_total) > max_score:
                max_score = float(num_corr)/float(num_total)
                max_num_corr = num_corr
                max_num_total = num_total
        
        ## Record the PCKh score
        print("\tCorrect:{} \tTotal:{}".format(num_corr, num_total))
        pckh_record.loc[img_num] = {'num_corect': num_corr, 'num_total': num_total}

        num_processed += 1

        if num_processed%15 == 0:
            pckh_record.to_csv('pckhrecord.csv')
