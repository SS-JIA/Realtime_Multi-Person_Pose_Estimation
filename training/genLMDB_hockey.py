import h5py
import sys, os
import os.path
import copy
import cv2
import lmdb
import caffe
import struct
import numpy as np

def writeLMDB(data_base_path, annot_relpath, lmdb_path, val=0):
    ## Setup
    annot = h5py.File(os.path.join(data_base_path, annot_relpath), 'r')
    env = lmdb.open(lmdb_path, map_size=int(1e12))
    txn = env.begin(write=True)

    ## Randomize order
    num_images = len(annot['imgname'])
    process_order = np.random.permutation(np.arange(183,num_images))
    
    ## Count validation and train
    total_writecount = num_images
    if val == 1:
        total_writecount = 0
        for i in process_order:
            if annot['istrain'][i]:
                total_writecount += 1

    write_count = 0
    ## Create input matrix for each image
    for idx in process_order:
        ## Skip non-val entries in validation mode
        if val == 1 and annot['istrain'][idx] == 1:
            continue

        ## Store the original image
        filename = ''.join(chr(int(i)) for i in annot['imgname'][idx])
        img = cv2.imread(os.path.join(data_base_path, filename))

        img_height = img.shape[0] 
        img_width = img.shape[1] 

        ## Filler masks
        mask_miss = np.zeros(shape=(img_height, img_width), dtype=np.uint8)
        mask_all = np.zeros(shape=(img_height, img_width), dtype=np.uint8)

        ## Metadata
        meta_data = np.zeros(shape=(img_height, img_width, 1), dtype=np.uint8)
        cur_row = 0
        cur_col = 0

        ### Dataset name
        dataname = 'hockey'
        if val:
            dataname = 'hockey_val'
        for i, char in enumerate(dataname):
            meta_data[cur_row][i] = ord(char)
        cur_row += 1
        cur_col = 0

        ### Image Dimensions
        height_binary = float2bytes(float(img_height))
        for i, val in enumerate(height_binary):
            meta_data[cur_row][cur_col] = ord(val)
            cur_col += 1

        width_binary= float2bytes(float(img_width))
        for i, val in enumerate(width_binary):
            meta_data[cur_row][cur_col] = ord(val)
            cur_col += 1

        cur_row += 1
        cur_col = 0
        
        ### Check if val
        meta_data[cur_row][cur_col] = float(not annot['istrain'][idx])
        cur_col += 1
        ### No other people
        meta_data[cur_row][cur_col] = float(0) # numOtherPeople
        cur_col += 1
        meta_data[cur_row][cur_col] = float(0) # people_index
        cur_col += 1
        ### Index in hdf5
        print("Index: {}".format(idx))
        anno_idx_bin = float2bytes(float(idx))
        for i, val in enumerate(anno_idx_bin):
            meta_data[cur_row][cur_col] = ord(val)
            cur_col += 1
        ### Write Count
        writecount_bin = float2bytes(float(write_count))
        for i, val in enumerate(writecount_bin):
            meta_data[cur_row][cur_col] = ord(val)
            cur_col += 1
        ### Total Write Count
        total_writecount_bin = float2bytes(float(total_writecount))
        for i, val in enumerate(total_writecount_bin):
            meta_data[cur_row][cur_col] = ord(val)
            cur_col += 1
        cur_row += 1
        cur_col = 0
        
        keypnt_coord = annot['part'][idx]
        ### Object Position
        objpos = np.mean(keypnt_coord, axis=0)
        for i, val in enumerate(float2bytes(objpos)):
            meta_data[cur_row][cur_col] = ord(val)
            cur_col += 1
        cur_row += 1
        cur_col = 0
        ### Scale provided
        scale_provided = float(annot['scale'][idx])
        for i, val in enumerate(float2bytes(scale_provided)):
            meta_data[cur_row][cur_col] = ord(val)
            cur_col += 1
        cur_row += 1
        cur_col = 0
        ### Keypoint Coordinates
        joints = np.concatenate((keypnt_coord, np.ones(18)[..., np.newaxis]), axis=1).T.tolist()
        print("Check: {}".format(joints[0]))
        for i, arr in enumerate(joints):
            arr_bin = float2bytes(arr)
            for j, val in enumerate(arr_bin):
                meta_data[cur_row][cur_col] = ord(val)
                cur_col += 1
            cur_row += 1
            cur_col = 0

        ## Store data
        input_mat = np.concatenate((img, meta_data, mask_miss[...,np.newaxis], mask_all[..., np.newaxis]), axis=2)
        input_mat = np.transpose(input_mat, (2,0,1))
        datum = caffe.io.array_to_datum(input_mat, label=0)
        key = '%07d' % write_count 
        txn.put(key, datum.SerializeToString())

        write_count += 1
    
    txn.commit()
    env.close()

def float2bytes(floats):
    if type(floats) is float:
        floats = [floats]
    return struct.pack('%sf' % len(floats), *floats)


if __name__ == '__main__':
    writeLMDB('/home/stephen/Datasets/harpe', 'annot.h5', '/home/stephen/Datasets/harpe/lmdb')
