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

        ## Limb Definitions
        self.limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                        [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
                        [1,16], [16,18], [3,17], [6,18]]
        self.mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
                       [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
                       [55,56], [37,38], [45,46]]

    def getPose(self, input_path):
        ## Get the input image
        input_img = cv.imread(input_path)
        ## Multipliers 
        multipliers = [x * self.model['boxsize'] / input_img.shape[0] for x in self.param['scale_search']]

        ## Confidence Map and PAF
        heatmap_agg, paf_agg = getSL(img, multipliers)

        ## Use NMS to get confidence map peaks
        num_peaks, peaks = nonMaximalSurpression(heatmap_agg)

    def getSL(img, multipliers):
        ## Average confidence map and PAF
        heatmap_avg = np.zeros((img.shape[0], img.shape[1], 19))
        paf_avg = np.zeros((img.shape[0], img.shape[1], 38))

        for m in range(len(multipliers)):
            scale = multipliers[m]
            imageToTest = cv.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])
            
            axarr[m].imshow(imageToTest_padded[:,:,[2,1,0]])
            axarr[m].set_title('Input image: scale %d' % m)

            net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
            net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
            start_time = time.time()
            output_blobs = net.forward()

            # extract outputs, resize, and remove padding
            heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1,2,0)) # output 1 is heatmaps
            heatmap = cv.resize(heatmap, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
            heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv.INTER_CUBIC)
            
            paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1,2,0)) # output 0 is PAFs
            paf = cv.resize(paf, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
            paf = cv.resize(paf, (img.shape[1], img.shape[0]), interpolation=cv.INTER_CUBIC)

            ## Add to aggregate
            heatmap_avg = heatmap_avg + heatmap / len(multipliers)
            paf_avg = paf_avg + paf / len(multipliers)

        return heatmap_avg, paf_avg

    def nonMaximalSurpression(heatmap)
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
