% Performs two functions:
% 1. Filters annotations that don't contain keypoints
% 2. Package together annotations that belong to the same image
% Does the above for both validation and testing
%
% Output:
%   coco_kpt: array of packaged annotations
%     coco_kpt(i).image_id
%     coco_kpt(i).annorect - array of of annotations data for each person

dataType = '';
addpath('dataset/COCO/coco/MatlabAPI');

mkdir('dataset/COCO/mat')

annTypes = { 'instances', 'captions', 'person_keypoints' };
annType=annTypes{3}; % specify dataType/annType

for mode = 0:1
    %% Load keypoint annotations
    if mode == 0
        dataType= 'val2014';
        annFile=sprintf('dataset/COCO/annotations/%s_%s.json',annType,dataType);
    else
        dataType = 'train2014';
        annFile=sprintf('dataset/COCO/annotations/%s_%s.json',annType,dataType);
    end
    
    coco=CocoApi(annFile);
    
    my_anno = coco.data.annotations;
    
    %% for each annotation...
    prev_id = -1;
    p_cnt = 1;      % Number of people in current image
    cnt = 0;        % Total # of images
    coco_kpt = [];  % output structure
    
    for i = 1:1:size(my_anno,2)
        if my_anno(i).num_keypoints == 0
            continue;
        end
            
        curr_id = my_anno(i).image_id;
        if(curr_id == prev_id)
            p_cnt = p_cnt + 1;
        else
            p_cnt = 1;
            cnt = cnt + 1;
        end
        
        % store annotation data in heirarchy
        coco_kpt(cnt).image_id = curr_id;
        coco_kpt(cnt).annorect(p_cnt).bbox = my_anno(i).bbox;
        coco_kpt(cnt).annorect(p_cnt).segmentation = my_anno(i).segmentation;
        coco_kpt(cnt).annorect(p_cnt).area = my_anno(i).area;
        coco_kpt(cnt).annorect(p_cnt).id = my_anno(i).id;
        coco_kpt(cnt).annorect(p_cnt).iscrowd = my_anno(i).iscrowd;
        coco_kpt(cnt).annorect(p_cnt).keypoints = my_anno(i).keypoints;
        coco_kpt(cnt).annorect(p_cnt).num_keypoints = my_anno(i).num_keypoints;
        coco_kpt(cnt).annorect(p_cnt).img_width = coco.loadImgs(curr_id).width;
        coco_kpt(cnt).annorect(p_cnt).img_height = coco.loadImgs(curr_id).height;
        
        prev_id = curr_id;
        
        fprintf('%d/%d \n', i, size(my_anno, 2));
    end
    %%
    if mode == 0
        coco_val = coco_kpt;
        save('dataset/COCO/mat/coco_val.mat', 'coco_val');
    else
        save('dataset/COCO/mat/coco_kpt.mat', 'coco_kpt');
    end
    
end