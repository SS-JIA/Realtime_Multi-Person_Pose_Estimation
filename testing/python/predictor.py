import cv2 as cv 
import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter
import PIL.Image
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import os.path

base_path = os.path.dirname(os.path.abspath(__file__))
default_config = os.path.join(base_path, 'config')


## Limb Definitions
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
          [55,56], [37,38], [45,46]]

class PoseModel(object):
    """
    Stores the keypoints of a detected human pose as a list of (x, y) coordinates.

    The indices corresponding to each body part are as follows:
    0 - Nose
    1 - Neck
    2 - Right Shoulder
    3 - Right Elbow
    4 - Right Hand
    5 - Left Shoulder
    6 - Left Elbow
    7 - Left Hand
    8 - Right Hip
    9 - Right Knee
    10 - Right Ankle
    11 - Left Hip
    12 - Left Knee
    13 - Left Ankle
    14 - Right Eye
    15 - Left Eye
    16 - Right Ear
    17 - Left Ear

    """
    def __init__(self, coordinates=None):
        if coordinates is None:
            self.keypoints = np.repeat(np.array([np.nan, np.nan], ndmin=2), 18, axis=0)
        elif not isinstance(coordinates, np.ndarray) and coordinates.shape != (18, 2):
            raise Exception("Please input a 18x2 array")
        else:
            self.keypoints = coordinates

    def setPart(self, part_type, coordinates):
        self.keypoints[part_type][0] = coordinates[0]
        self.keypoints[part_type][1] = coordinates[1]

    def getPart(self, part_type):
        return self.keypoints[part_type][0], self.keypoints[part_type][1]

    def overLay(self, image, colors=None):
        canvas = image.copy()
        ## Set colors
        if colors is None:
            colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                      [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                      [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        elif len(colors) != 18:
            raise Exception("Not enough colors defined")

        ## Draw body parts
        for part_type in range(18):
            part_x, part_y = self.getPart(part_type)
            if not np.isnan(part_x):
                cv.circle(canvas, (int(part_x), int(part_y)), 4, colors[part_type], thickness=-1)

        ## Draw limbs
        for limb_type in range(17):
            A_type = limbSeq[limb_type][0]-1
            B_type = limbSeq[limb_type][1]-1
            A_x, A_y = self.getPart(A_type)
            B_x, B_y = self.getPart(B_type)
            if not np.isnan(A_x) and not np.isnan(B_x):
                cur_canvas = canvas.copy()
                cv.line(cur_canvas, (int(A_x), int(A_y)), (int(B_x), int(B_y)), colors[limb_type], thickness=4)
                canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        return canvas

class OpenPosePredictor(object):
    """
    Runs the OpenPose algorithm on an input image
    """
    
    def __init__(self, config_path=default_config):
        self.config_path = config_path

        ## Load the model parameters from config file
        self.param, self.model = config_reader(self.config_path)

        ## Set up Caffe model
        if self.param['use_gpu']: 
            caffe.set_mode_gpu()
            caffe.set_device(self.param['GPUdeviceNumber']) # set to your device!
        else:
            caffe.set_mode_cpu()

        self.net = caffe.Net(self.model['deployFile'], self.model['caffemodel'], caffe.TEST)

    def getPoseModels(self, input_path):
        ## Get the input image
        self.input_img = cv.imread(input_path)
        ## Multipliers for scale search
        multipliers = [x * self.model['boxsize'] / self.input_img.shape[0] for x in self.param['scale_search']]

        ## Get Body Part Confidence Maps and PAFs
        self.heatmap_agg, self.paf_agg = self.getSL(self.input_img, multipliers)

        ## Get body part locations
        num_keypoints, self.keypoints = self.getKeypoints(self.heatmap_agg)

        ## Create pose models using the detected keypoints
        self.pose_models = self.makeConnections(self.keypoints, self.paf_agg)
        return self.pose_models

    def getSL(self, img, multipliers):
        ## Average confidence map and PAF
        heatmap_avg = np.zeros((img.shape[0], img.shape[1], 19))
        paf_avg = np.zeros((img.shape[0], img.shape[1], 38))

        for m in range(len(multipliers)):
            scale = multipliers[m]
            imageToTest = cv.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, self.model['stride'], self.model['padValue'])

            self.net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
            self.net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
            start_time = time.time()
            output_blobs = self.net.forward()

            # extract outputs, resize, and remove padding
            heatmap = np.transpose(np.squeeze(self.net.blobs[output_blobs.keys()[1]].data), (1,2,0)) # output 1 is heatmaps
            heatmap = cv.resize(heatmap, (0,0), fx=self.model['stride'], fy=self.model['stride'], interpolation=cv.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
            heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv.INTER_CUBIC)
            
            paf = np.transpose(np.squeeze(self.net.blobs[output_blobs.keys()[0]].data), (1,2,0)) # output 0 is PAFs
            paf = cv.resize(paf, (0,0), fx=self.model['stride'], fy=self.model['stride'], interpolation=cv.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
            paf = cv.resize(paf, (img.shape[1], img.shape[0]), interpolation=cv.INTER_CUBIC)

            ## Add to aggregate
            heatmap_avg = heatmap_avg + heatmap / len(multipliers)
            paf_avg = paf_avg + paf / len(multipliers)

        return heatmap_avg, paf_avg

    def getKeypoints(self, heatmap):
        """
        Use non maximal surpression to extract body part locations from confidence maps.         Parameters
        ----------
        heatmap: matrix
            A (N x M x 17) matrix containing confidence values for each body part at each pixel location, where N x M is
            the size of the input image

        Output
        ------
        num_peaks: int
            The total number of peaks detected
        all_peaks: array of tuples
            all_peaks[i] gives list of (x, y, score, id) for body part type i
        """
        all_peaks = []
        peak_counter = 0

        for part in range(19-1):
            x_list = []
            y_list = []
            map_ori = heatmap[:,:,part]
            map_smooth = gaussian_filter(map_ori, sigma=3)
            
            map_left = np.zeros(map_smooth.shape)
            map_left[1:,:] = map_smooth[:-1,:]
            map_right = np.zeros(map_smooth.shape)
            map_right[:-1,:] = map_smooth[1:,:]
            map_up = np.zeros(map_smooth.shape)
            map_up[:,1:] = map_smooth[:,:-1]
            map_down = np.zeros(map_smooth.shape)
            map_down[:,:-1] = map_smooth[:,1:]
            
            peaks_binary = np.logical_and.reduce((map_smooth>=map_left, map_smooth>=map_right, map_smooth>=map_up, map_smooth>=map_down, map_smooth > self.param['thre1']))
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
            peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        return peak_counter, all_peaks

    def makeConnections(self, all_peaks, paf, mid_num=10):
        pose_model_of = dict()
        pose_models = []

        ## For each limb type
        for k in range(len(mapIdx)):
            ## Get the PAFs associated with the limb type
            score_mid = paf[:,:,[x-19 for x in mapIdx[k]]]
            
            ## Get all relevant detected body parts
            A_type = limbSeq[k][0]-1
            B_type = limbSeq[k][1]-1
            candA = all_peaks[A_type]
            candB = all_peaks[B_type]
            nA = len(candA)
            nB = len(candB)
            
            ## List of B that have already been matched
            if(nA != 0 and nB != 0):
                connection_candidates = []
                for i in range(nA):
                    for j in range(nB):
                        ## Determine connection score
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                        vec = np.divide(vec, norm)
                        
                        startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num))
                        
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                          for I in range(len(startend))])
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        
                        ## Check against thresholds
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*self.input_img.shape[0]/norm-1, 0)
                        if candA[i][3] in pose_model_of:
                            score_with_dist_prior = score_with_dist_prior * self.param['exist_boost']
                        ## 80% of the path must have agreeable PAF values
                        criterion1 = len(np.nonzero(score_midpts > self.param['thre2'])[0]) > 0.8 * len(score_midpts)
                        ## The average agreement must be above 0
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidates.append([i, j, score_with_dist_prior])
                
                ## Sort candidates by score
                connection_candidates = sorted(connection_candidates, key=lambda x: x[2], reverse=True)

                ## List of matched body parts
                matched_As = np.zeros(nA, dtype=bool)
                matched_Bs = np.zeros(nB, dtype=bool)
                
                num_matches = 0
                for i, j, score in connection_candidates:
                    if matched_As[i] or matched_Bs[j]:
                        continue
                    ## Part IDs of the parts being connected
                    A_id = candA[i][3]
                    B_id = candB[j][3]
                    ## Add B to the pose model associated with the "from" body part or create a new one
                    if A_id not in pose_model_of:
                        pose_model = PoseModel()
                        pose_models.append(pose_model)
                        pose_model_of[A_id] = pose_model
                        pose_model.setPart(A_type, candA[i][:2])
                    else:
                        pose_model = pose_model_of[A_id]

                    pose_model.setPart(B_type, candB[j][:2])
                    pose_model_of[B_id] = pose_model

                    num_matches = num_matches + 1
                    matched_As[i] = True
                    matched_Bs[j] = True

        return pose_models

    def drawPoseModels(self, modelnums=None):
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        canvas = self.input_img.copy()

        ## Select the models to
        if modelnums is None:
            queue = np.arange(len(self.pose_models))
        else:
            queue = np.array(modelnums)

        for index in queue:
            pose_model = self.pose_models[index]
            canvas = pose_model.overLay(canvas)

        return canvas
