# Identifying Recurring Patterns with Deep Neural <br />  Networks for Natural Image Denoising

Copyright (C) 2018, Zhihao Xia and Ayan Chakrabarti

This distribution provides an implementation, along with trained models, for the method described in our paper:

Zhihao Xia and Ayan Chakrabarti, "**[Identifying Recurring Patterns with Deep Neural Networks for Natural Image Denoising](https://arxiv.org/abs/1806.05229)**", WACV 2020.


If you find the code useful for your research, we request that you cite the above paper. Please contact zhihao.xia@wustl.edu with any questions.

## Denoising with trained models

The repository contains pre-trained models in the `wts` directory for color denoising at different noise levels. To run the denoising code with trained models, run 
```
./denoise.py 50 -l listfile.txt
```
for denoising on noise level 50 on all images listed in the text file `listfile.txt`. Note that the denoising code loads clean images, adds AWGN at the provided noise level, and runs denoising with the corresponding model stored in the `wts` directory. 

You can also get results from denoising without the regression network, i.e., with only  averaging matched patterns, by running 
```
./denoise.py 50 -l listfile.txt -r 0
```
Look at `denoise/denoise.py` for a list of parameters you can specify or run 
```
./denoise.py -h
```

## Training

To train your own model, please create empty directories `wts/nstd_NN/train` and `wts/nstd_NN/regress`, where `NN` is the desired noise standard deviation (we will use noise level 50 in the following examples). You must also create the files `data/train.txt` and `data/dev.txt` with lists of image paths to be used for training and validation (since the training scripts periodically compute validation errors on the validation set, its best to keep this set reasonably small: we used 32 images in our experiments).

The whole training process includes two phases. In the first phase, we train a matching network for matching patches and outputting weights for sub-bands coefficients.  This phase also involves a pre-training step. In the second phase, we first need to generate triplets of clean image, noisy image and the clean estimate from matching network trained in the first phase. Then we use these triplets to train a regression network.


### Pre-training
In the root directory, run 
```
./pretrain.py 50
```
to pre-train the matching network for 100k iterations. The trained model will be saved in `wts/nstd_50/pretrain`.

### Training Matching Net
Run 
```
./train.py 50
```
to train the matching network which is initialized from the pre-trained model in the pre-training step above.

### Get population statistics
After you finish the training step, you need to get the population statistics for the batch-normalization layers in the model in order to test the model on test dataset. To do so, run
```
./getBN.py 50 train [-m wts/nstd_50/train/iter_600000.model.npz]
```

### Get denoised images to train the regression network
Call `getDnz.py` to get the triplets for training and validating the regression network. Specifically,
```
./getDnz.py 50 -l data/train.txt
./getDnz.py 50 -l data/dev.txt
```
Note that for every clean image in the dataset, five pairs of noisy input and initial estimate are generated. Generated results (clean image, noisy image and denoised image) are saved in `wts/nstd_50/initial_estimates` as uint16 files.

### Training Regression Network
Run
```
./regress.py 50
```
to train the regression network, which takes a noisy image and initial estimate pair and output final estimate of our overall approach. Similarly, after the training is done, you need to get the population statistics for the batch-norm layers in the model. Specifically, run
```
./getBN.py 50 regress [-m wts/nstd_50/regress/iter_600000.model.npz]
```
