#!/usr/bin/env python3

import os
import importlib, argparse
from glob import glob
import re
import numpy as np
from imageio import imread, imsave
import tensorflow as tf

from dnz import Denoiser
from net import net

parser = argparse.ArgumentParser()
parser.add_argument('nstd',type=int,help='noise level, from 0-255')
parser.add_argument('-l',default='data/dev.txt',help='List of images to generate initial estimate')
parser.add_argument('-w',type=int,default=31,help='Window size')
opts = parser.parse_args()

#########################################################################
# Params
NSTD = opts.nstd/255.   # Denoise for this noise level
wts = 'wts/nstd_%d/regress'%(opts.nstd)
if not os.path.exists(wts):
    os.mkdir(wts)

lfile = opts.l
ofile = wts + '/' + os.path.basename(lfile)
outdir = 'wts/nstd_%d/initial_estimates'%opts.nstd
if not os.path.exists(outdir):
    os.mkdir(outdir)

nstd = opts.nstd/255.0
wsz = opts.w

# Get the last model of training
wcard = "wts/nstd_%d/train/iter_*.model.npz"%(opts.nstd)
lst=[(l,int(re.match('.*/.*_(\d+)',l).group(1))) for l in glob(wcard)]
mfile = max(lst, key=lambda x: x[1])[0]
bfile = mfile.replace('model', 'bnwts')


#########################################################################
# Setup Graphs
net.toTest()
model = net.Net()

# create variables for batch statistics
bnwts = {}
wts = np.load(bfile)
for bnnm in wts.keys():
    bnwts[bnnm] = tf.Variable(tf.random_uniform(wts[bnnm].shape),trainable=False)
model.bnwts = bnwts

denoiser = Denoiser(model, net.psz, net.csz, wsz, net.stride, liter=1)

# Add nodes to save images
nm = tf.placeholder(dtype=tf.string)
im, nz, dz = tf.placeholder(dtype=tf.float32), tf.placeholder(dtype=tf.float32), tf.placeholder(dtype=tf.float32)
fnms = [tf.string_join([nm, '.png']), tf.string_join([nm, '_nz.png']), tf.string_join([nm, '_dnz.png'])]

imsp = tf.shape(im)
impng = tf.reshape(im, [imsp[0],imsp[1]*imsp[2],1])
impng = tf.cast(tf.clip_by_value(impng+0.5,0.0,2.0) / 2.0 * (2**16 -1),tf.uint16)
impng = tf.image.encode_png(impng)
nzpng = tf.reshape(nz, [imsp[0],imsp[1]*imsp[2],1])
nzpng = tf.cast(tf.clip_by_value(nzpng+0.5,0.0,2.0) / 2.0 * (2**16 -1),tf.uint16)
nzpng = tf.image.encode_png(nzpng)
dzpng = tf.reshape(dz, [imsp[0],imsp[1]*imsp[2],1])
dzpng = tf.cast(tf.clip_by_value(dzpng+0.5,0.0,2.0) / 2.0 * (2**16 -1),tf.uint16)
dzpng = tf.image.encode_png(dzpng)

encoded = [impng, nzpng, dzpng]
fwrites = [tf.write_file(fnms[i], encoded[i]) for i in range(len(fnms))]


#########################################################################
# Start TF session (respecting OMP_NUM_THREADS) & load model
nthr = os.getenv('OMP_NUM_THREADS')
if nthr is None:
    sess = tf.Session()
else:
    sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=int(nthr)))
sess.run(tf.global_variables_initializer())


# Load model
wts = np.load(mfile)
ph = tf.placeholder(tf.float32)
for k in wts.keys():
    wvar = model.weights[k]
    wk = wts[k].reshape(wvar.get_shape())
    sess.run(wvar.assign(ph),feed_dict={ph: wk})

wts = np.load(bfile)
for k in wts.keys():
    wvar = model.bnwts[k]
    wk = wts[k].reshape(wvar.get_shape())
    sess.run(wvar.assign(ph),feed_dict={ph: wk})


#########################################################################

files = [f.rstrip('\n') for f in open(lfile,'r').readlines()]
avpsnr = 0.

ofile = open(ofile, 'w')
for i in range(len(files)):
    name = os.path.splitext(os.path.basename(files[i]))[0]

    img = np.float32(imread(files[i]))/255.

    for k in range(5):
        impath = outdir + '/' + name + '_%d'%k
        ofile.write(impath + '\n')

        # Add noise
        imnz = img + np.random.standard_normal(img.shape)*nstd

        imdz = denoiser.run(imnz, sess)
        imdz = np.maximum(0.,np.minimum(1.,imdz))

        sess.run(fwrites, feed_dict={nm: impath, im: img, nz: imnz, dz:imdz})

        mse = np.mean((imdz[:]-img[:])**2)
        psnr = -10*np.log10(mse)
        print("%d: MSE = %.2e, PSNR=%.2f" % (i,mse,psnr))
        avpsnr = avpsnr + psnr

print("Average PSNR = %.2f" % (avpsnr/(len(files)*5.0)))
