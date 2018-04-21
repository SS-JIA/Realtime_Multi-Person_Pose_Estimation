## Getting Familiar

To try the model out and get familiar with the end-to-end algorithm, I recommend going
through the original python demo notebook which can be found at `testing/python/demo.ipynb`.
This notebook will allow you to get familiar with the inputs and outputs of the neural network,
as well as the matching algorithm used in post-processing to establish pose models.

Note that you can configure the behaviour of the algorithm with the configuration file found
at `testing/python/config`. The main parameters you'll want to look out for are

* **use_gpu**, toggles caffe GPU/CPU mode
* **caffemodel, deployFile**, change these to load in the desired model

The original project comes with two pre-trained models: one trained on the COCO dataset and one
trained on the MPII dataset. The COCO dataset uses different pose keypoints than MPII. Our hockey
keypoints extend the MPII pose keypoints, so the pretrained MPII model will be used as a starting
point for training later on.

## Pose Model Specification - MPII

The pose keypoints indices for MPII are defined as follows in the algorithm:

*  0 - head top
*  1 - upper neck
*  2 - r shoulder
*  3 - r elbow
*  4 - r wrist
*  5 - l shoulder
*  6 - l elbow
*  7 - l wrist)
*  8 - r hip
*  9 - r knee
* 10 - r ankle
* 11 - l hip
* 12 - l knee
* 13 - l ankle
* 14 - midpoint between pelvis and thorax

<p align="center">
<img src="mpii_keypoint_indices.png">
</p>

The red arrows represent limbs. So limb 2 would connect body part 2 (right shoulder) to
body part 3 (right elbow).

Also note that the original MPII pose keypoints consider the pelvis and thorax separately. So to make
MPII annotations compatible with a model trained with the above keypoint definitions, the pelvis and
thorax keypoints will have to be combined into a single keypoint.

## Algorithm Interface

For convenience, I've also wrapped the functionality described in `testing/python/demo.ipynb` into an
`OpenPosePredictor` class in `testing/python/predictor.py`. The class takes in an input image, passes into the
neural network, performs the graph matching on the network output, and returns a list of `PoseModel`
classes also described in `testing/python/predictor.py`. An usage example can be found at
`wrapper_test.ipynb`.

`OpenPosePredictor` and `PoseModel` can be configured with the same config file format as `testing/python/demo.ipynb`.
I've designed the two classes so that they can be configured to work with
any pose keypoint specification (i.e. can work with both COCO and MPII pose model outputs, and eventually
with the hockey pose models that has the two additional keypoints). To switch to a new pose model specification, simply
alter the following fields in the config file:

* **np**, the number of body parts in the pose model
* **limb_from**, the body part that each limb leaves from
* **limb_to**, the body part that each limb leaves to
* **limb_order**, the order in which to process limbs in the graph matching portion of the algorithm

The **limb_order** field is especially important to ensuring that the algorithm works correctly. Viewing the pose model
as a directed tree, I've streamlined the graph matching process under the assumption that the pose models are built in a
breadth first manner. 

So if you had a pose model that consisted of 4 body parts (indices 0-3) with the following limb definitions:

* limb 0 connects body part 0 to body part 1
* limb 1 connects body part 1 to body part 2
* limb 2 connects body part 2 to body part 3

When building the pose model, we would want to construct limb 0 first, then limb 1, then limb 2 to achieve a breadth
first traversal of the pose model tree. An example of an order that would not work would be 2, 1, 0.

The config file would define the following fields:

```
np = 4
limb_from = 0, 1, 1
limb_to = 1, 2, 3
limb_order = 0, 1, 2
```
## PCKh Calculation

[PCKh](http://human-pose.mpi-inf.mpg.de/#results) is the metric that we use to score the model. Essentially, it looks at
the *P*robability that the model detects a *C*orrect *K*eypoint. A detected keypoint is considered correct if it
deviates from the ground truth keypoint no more than 50% of the *h*ead segment length (calculated using ground truth
coordinates).

I've implemented a function `computePCKh` in `testing/python/predictor.py` which will determine which detected keypoints
are correct between a detected pose model and a ground truth pose model.

An example PCKh calculation can be found in the python notebook at `PCKh.ipynb`.

The script `test/python/computePCKh.py` runs through all the hockey images and records for each image:

* for each body part, 1 for correct, 0 for incorrect, and `np.nan` for not detected
* the total number of found body parts in the image
* the number of correct body parts found

Note that only the detected pose model with the highest number of detected body parts is used for comparison. The
results are stored in a pandas table and written to a csv at `testing/python/pckhrecord.csv` every 5 images. If the
file already exists, running the script will start from the last recorded image in `pckhrecord.csv`.

Once all the images can be processed, `pckhrecord.csv` can be loaded and the PCKh for each body part as well as the
overall PCKh can be calculated.

## Neural Network Training

Note that to train the network, you should install the author's custom caffe [caffe_train](https://github.com/CMU-Perceptual-Computing-Lab/caffe_train).
This project provides a special data augmentation layer that is necessary to generate the part affinity fields and heat
maps from the ground truth annotations. Building this project should be fairly straightforward (same as regular caffe),
but note that I ran into trouble building with cuDNN7, so I had to go back to cuDNN5. This
[project](https://github.com/dnzzcn/cuDNNv) came in handy managing multiple cuDNN versions.

First, I'll provide some context for how training works in caffe...

### Blobs

A [blob](http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html) is the format in which data flows through a caffe
network. It's basically just a matrix.

### Prototxts and Caffemodels

Caffe defines model architectures in the form of prototxts. These prototxts basically defines a list of layers. Each
layer outputs one or more blobs, called **tops**. Some types of layers can also receive blobs as input, called
**bottoms**. So the prototxt specifies for each layer

* name
* type
* bottom(s)
* top(s)

As well as other parameters for that layer.

The caffe framework reads the prototxt, which is essentially a blueprint for how it constructs the network internally.
The weights are provided by a `.caffemodel` file, which maps a layer name to the weights of that layer.

### Deploy and Train/Test Prototxts

Two prototxts for the model will be written. The first is a **train_test** prototxt which is used to tune the weights of the model.
This prototxt will have a data layer, which specifies an lmdb or hdf5 database containing your training data, as
well as loss layers. The **deploy** prototxt removes the loss layers and instead of having a data layer specifies the
data be input into the network.

`training/example_proto` provides an example train/test prototxt for training of COCO images. You can find my training
prototxt for MPII at `models/trained_harpe/pose_train_test.prototxt`. The main difference between these two are how the
input data is processed and the number of output layers.

### lmdb Input Database

For this project, the data layer specifies an lmdb that holds all the training data. For the hockey data, the lmdb is
generated by `training/genLMDB_hockey.py`. It's based off the original lmdb generator, `training/genLMDB.py`. Take
a look to see what the expected input format is. To give a brief summary, the lmdb should store a sequence of 6xHxW
matrices, one for each training image where H and W are the height and width of the images respectively.

* The first 3 channels contain the image
* The fourth channel contains metadata as well as ground truth keypoints
* The fifth and sixth channels store masks of the original image

Note that the grount truth keypoints should be stored using the original MPII indices, which are:

*  0 - r ankle
*  1 - r knee
*  2 - r hip
*  3 - l hip
*  4 - l knee
*  5 - l ankle
*  6 - pelvis
*  7 - thorax
*  8 - upper neck
*  9 - head top
* 10 - r wrist
* 11 - r elbow
* 12 - r shoulder
* 13 - l shoulder
* 14 - l elbow
* 15 - l wrist)

These will be reordered in the preprocessing stage into the format described in the picture above. The hockey
data annotations also use the above format, so no modifications were needed.

Also be aware that we have no segmentation annotations for our hockey images so we cannot create masks. For
now I just use a black image for the mask channels but I'm not sure how this affects training. From what I've read
of the source code of the data augmentation layer I don't think it should affect much.

### Generating Prototxts

When working with COCO input, you can just use the train/test prototxt at
`training/example_proto/pose_train_test.prototxt`; for MPII input, you can just use the train/test prototxt I've
generated at `models/trained_harpe/pose_train_test.prototxt`.

However, when you get around to adding the two extra hockey keypoints you can generate customized train/test and deploy
prototxts with `training/setLayers.py`. It will be necessary to generate new prototxts because the number of limbs/body
parts in the pose model you use affect the output format of the network. Specifically, you'll want to change the *np* parameter.
This value is to be set equal to `(number of body parts in pose model) + (number of joints in pose model * 2)`.

You'll also want to edit the Slice layer in the output train/test prototxt. This layer is defined right after the data
input layer in the generated train/test prototxt. Once the data augmentation layer generates
the part affinity fields all the part affinity field and heat map data is concatenated into a huge label matrix. This
label matrix will have **np*2** channels. The first half provides weightings, and the second half provides the actual
values for the pafs and heat maps.
The Slice layer is the layer that splits the huge label matrix into its individual components. You'll have to edit this layer according
to the number of heat maps and part affinity fields you expect. To give an example of the slice layer used for MPII input:

```
layer {
  name: "vec_weight"
  type: "Slice"
  bottom: "label"
  top: "vec_weight"
  top: "heat_weight"
  top: "vec_temp"
  top: "heat_temp"
  slice_param {
    slice_point: 28
    slice_point: 44
    slice_point: 72
    axis: 1
  }
}
```

The first slice point separates channels 0-27 (inclusive). These 28 channels correspond to the part affinity field weights; the pose
model for MPII input uses 14 limbs, and each limb's part affinity field is vector field with `x` and `y` magnitudes, 
giving a total of 28 weight channels. The next slice point separates out channels 28-44, the 16 channels that are associated
with the heat map weights. The pose model used for MPII has 15 body parts, which correspond to the first 15 channels of
this section. The last channel provides a heat map weights for all body parts detected. The last slice point separates
out channels 44-71, the 28 channels giving the vector field magnitudes for the pafs. Now only the last 16 channels are
left to provide the heat map magnitudes. You can also see that these sections are output as the `vec_weight`,
`heat_weight`, `vec_temp`, and `heat_temp` blobs respectively.

### cpm_data_layer and cpm_data_transformer

When looking through the generated prototxts notice that the data input layer is of type "CPMData". This layer's
functionality is defined by `cpm_data_layer.cpp` in the `caffe_train` project. Internally this layer uses the
`CPMDataTransformer` class defined in `cpm_data_transformer.cpp` to augment the data and generate the label data.

When adding the two extra hockey keypoints, you'll also have to patch in the functionality to handle the two extra
keypoints in `cpm_data_transformer.cpp`. As a guideline, you can search for `np == 56` and `np == 43` to see the
sections of the code where they handle the input data differently depending on whether it's COCO or MPII data. I think
this will be helpful in determining what changes you'll have to make in order to handle the two extra keypoints in the
hockey model.

### Training 

Once you have

* lmbd input database
* appropriate train_test prototxt
* solver prototxt

You can begin training by running

```
/path_to_caffe_train/caffe_train/build/tools/caffe train --solver=pose_solver.prototxt --gpu=$1 --weights=../_trained_MPI/pose_iter_146000.caffemodel 2>&1 | tee ./output.txt
```

The weights argument just provides a weight initialization to some or all of your layers (depending on which layers are
named in the caffemodel file you provide).

This command can can be run with `/model/_trained_harpe/train_pose.sh`. `setLayers` also generates a shell file for
training, but you'll have to edit it before you can use it.

### Debugging Training.

When running the above command, the output is sent to output.txt. If you want to get more information out of your logs,
you can add lines like

```
VLOG(2) << "Transform_nv: " << timer.MicroSeconds() / 1000.0  << " ms";
```

to `cpm_data_transformer` or `cpm_data_layer`. Caffe uses the GLOG library for logging so you can also take a look at
GLOG documentation to see what's possible. Do keep in mind that you'll have to set the `GLOG_v` environmental variable
appropriately to see `VLOG` output. For `VLOG(2)`, you'll need `GLOG_v = 2`.

Within `cpm_data_transformer` you can also output the generated part affinity field and heat map labels. You can do this
by setting the `visualize` parameter of the CPMData layer to true in the train/test prototxt. The visualization
functionality is found in the `visualize` method of `cpm_data_transformer`.

As a final note, you can also use the python interface to only run one forward pass of the training process. Take a look
at `model/test_input/test_input.py` to see how this is done.
