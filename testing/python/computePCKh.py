#!/bin/python
import pandas as pd
import numpy as np
import h5py
import os
import os.path
import argparse
import copy

os.environ['GLOG_minloglevel'] = '3' 

import predictor

## Retrive the ground truth annotations
ground_truth = h5py.File('/home/stephen/Datasets/harpe/annot.h5', 'r')

np = 15
limb_from = [0, 1, 2, 3, 1, 5, 6, 1, 14, 8, 9,  14, 11, 12]
limb_to = [1, 2, 3, 4, 5, 6, 7, 14, 8, 9, 10, 11, 12, 13]
limb_order = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]

def createGroundTruthModel(img_num):
    ## Extract the keypoint annotations
    parts_gt = ground_truth['part'][img_num-1]
    
    ## Reorder to expected format
    points_gt = parts_gt[[9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 7]]
    points_gt[-1] = (parts_gt[7] + parts_gt[6])/2

    ## Create PoseModel
    pose_model_gt = predictor.PoseModel(np, limb_from, limb_to)
    pose_model_gt.setCoords(points_gt)

    return pose_model_gt

def detectPoseModels(img_num):
    ## Get image path
    img_path = '/home/stephen/Datasets/harpe/' + str(img_num).zfill(3) + '.jpg'
    ## Run detection
    predictor_instance = predictor.OpenPosePredictor()
    pose_models_detected = predictor_instance.getPoseModels(img_path)

    return pose_models_detected

def processRecord(pckh_record):
    pckh_record = pckh_record[182:]
    results = dict()

    ## Get percentages for each part type
    for part_type in predictor.part_names:
        part_results = pckh_record[part_type]
        ## Filter out instances where the part was not detected
        part_results = part_results[part_results.notnull()]

        ## If part does not exist, record nothing
        if len(part_results) == 0:
            results[part_type] = np.nan
        ## Otherwise record the percentage of detections that were correct
        else:
            results[part_type] = part_results.sum() / len(part_results)
            print("{: >20}: {: <10}".format(part_type, round(results[part_type], 2)))

    ## Record the overall percentage of detected parts that were correct
    results['PCKh'] = pckh_record['correct'].sum() / pckh_record['found'].sum()
    print("{: >20}: {: <10}".format("PCKh", round(results["PCKh"], 2)))

    return results

if __name__ == '__main__':
    ## Load in current record of PCKh scores
    if os.path.isfile('pckhrecord.csv'):
        pckh_record = pd.read_csv('pckhrecord.csv', index_col=0)
    else:
        columns = copy.deepcopy(predictor.part_names)
        columns.append("found")
        columns.append("correct")
        pckh_record = pd.DataFrame(columns=columns)

    ## Determine where to start and end
    if len(pckh_record) == 0:
        start_img = 185
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

        ## Compare ground truth model with all detected and choose best PCKh
        keeper = None
        for pose_model_dt in pose_models_dt:
            parts_found = predictor.computePCKh(pose_model_gt, pose_model_dt)

            proceed = False
            if keeper is None:
                proceed = True
            elif parts_found['correct'] > max_correct_keypoints:
                proceed = True
            elif parts_found['correct'] == max_correct_keypoints and parts_found['found'] > max_found_keypoints:
                proceed = True

            if proceed:
                keeper = parts_found
                max_correct_keypoints = parts_found['correct']
                max_found_keypoints = parts_found['found']
        
        ## Record the PCKh score
        if keeper is not None:
            print("\tCorrect:{} \tFound:{}".format(keeper['correct'], keeper['found']))
            pckh_record.loc[img_num] = keeper

        num_processed += 1

        if num_processed%5 == 0:
            pckh_record.to_csv('pckhrecord.csv')

    pckh_record.to_csv('pckhrecord.csv')

    processRecord(pckh_record)
