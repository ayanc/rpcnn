#!/usr/bin/env python3

import os
import argparse
import numpy as np
from imageio import imread, imsave
import tensorflow as tf

from dnz import Denoiser
from net import net, rnet

#########################################################################
# Parse command line arguments

parser = argparse.ArgumentParser()
parser.add_argument('nstd',type=int,help='Noise level')
parser.add_argument('-tf',default=None,help='Path to saved training model.npz file.')
parser.add_argument('-rf',default=None,help='Path to saved regression model.npz file.')
parser.add_argument('-r',type=int,default=1,help='With final regression or not')
parser.add_argument('-l',default='data/dev.txt',help='Path to list of image names to test on.')
parser.add_argument('-o',default=None,help='(Optional) Directory to save noisy and denoised images in.')

opts = parser.parse_args()

tfile, rfile, lfile = opts.tf, opts.rf, opts.l
nstd = opts.nstd/255.
outdir = opts.o
ifregress = opts.r

if tfile is None:
    tfile = 'wts/nstd_%d/train/iter_600000.model.npz'%opts.nstd
if rfile is None:
    rfile = 'wts/nstd_%d/regress/iter_600000.model.npz'%opts.nstd
tbfile = tfile.replace('model', 'bnwts')
rbfile = rfile.replace('model', 'bnwts')

wsz = 31


#########################################################################
# Setup Graph for initial estimate
net.toTest()
tmodel = net.Net()

# Create variables for batch statistics
bnwts = {}
wts = np.load(tbfile)
for bnnm in wts.keys():
    bnwts[bnnm] = tf.Variable(tf.random_uniform(wts[bnnm].shape),trainable=False)
tmodel.bnwts = bnwts

denoiser = Denoiser(tmodel, net.psz, net.csz, wsz, net.stride, liter=1)


#########################################################################
if ifregress:
    # Setup Graph for regression
    rmodel = rnet.Net()

    # Create variables for batch statistics
    bnwts = {}
    wts = np.load(rbfile)
    for bnnm in wts.keys():
        bnwts[bnnm] = tf.Variable(tf.random_uniform(wts[bnnm].shape),trainable=False)
    rmodel.bnwts = bnwts

    rimnz = tf.placeholder(dtype=tf.float32,shape=[None,None,3])
    rimdz = tf.placeholder(dtype=tf.float32,shape=[None,None,3])
    rimmz = tf.concat([rimnz,rimdz],axis=2)
    residual = rmodel.refine(tf.stack([rimmz],axis=0))
    rimg = tf.squeeze(residual) + rimmz[:,:,3:]


#########################################################################
# Start TF session (respecting OMP_NUM_THREADS)
nthr = os.getenv('OMP_NUM_THREADS')
if nthr is None:
    sess = tf.Session()
else:
    sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=int(nthr)))
sess.run(tf.global_variables_initializer())

# Load model for denoising
wts = np.load(tfile)
ph = tf.placeholder(tf.float32)
for k in wts.keys():
    wvar = tmodel.weights[k]
    wk = wts[k].reshape(wvar.get_shape())
    sess.run(wvar.assign(ph),feed_dict={ph: wk})

wts = np.load(tbfile)
for k in wts.keys():
    wvar = tmodel.bnwts[k]
    wk = wts[k].reshape(wvar.get_shape())
    sess.run(wvar.assign(ph),feed_dict={ph: wk})

if ifregress:
    # Load model for regression
    wts = np.load(rfile)
    ph = tf.placeholder(tf.float32)
    for k in wts.keys():
        wvar = rmodel.weights[k]
        wk = wts[k].reshape(wvar.get_shape())
        sess.run(wvar.assign(ph),feed_dict={ph: wk})

    wts = np.load(rbfile)
    for k in wts.keys():
        wvar = rmodel.bnwts[k]
        wk = wts[k].reshape(wvar.get_shape())
        sess.run(wvar.assign(ph),feed_dict={ph: wk})


#########################################################################

files = [f.rstrip('\n') for f in open(lfile,'r').readlines()]
avpsnr = 0.
rand = np.random.RandomState()
for i in range(len(files)):
    img = np.float32(imread(files[i]))/255.

    # Add noise
    imnz = img + rand.standard_normal(img.shape)*nstd

    # Denoise
    imdz = denoiser.run(imnz, sess)

    if ifregress:
        # Regress
        imdz = sess.run(rimg, feed_dict={rimnz:imnz, rimdz:imdz})
        imdz = np.maximum(0.,np.minimum(1.,imdz))

    if outdir is not None:
        imsave(outdir+'/%d.png'%i,np.uint8(np.maximum(0.,np.minimum(1,img))*255))
        imsave(outdir+'/%d_nz.png'%i,np.uint8(np.maximum(0.,np.minimum(1,imnz))*255))
        imsave(outdir+'/%d_dnz.png'%i,np.uint8(np.maximum(0.,np.minimum(1,imdz))*255))

    mse = np.mean((imdz[:]-img[:])**2)
    psnr = -10*np.log10(mse)
    print("%d: MSE = %.2e, PSNR=%.2f" % (i,mse,psnr))
    avpsnr = avpsnr + psnr

print("Average PSNR = %.2f" % (avpsnr/len(files)))