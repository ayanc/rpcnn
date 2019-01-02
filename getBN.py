#!/usr/bin/env python3

import os
import importlib, argparse
import numpy as np
import tensorflow as tf

import utils as ut

from net import net, rnet, baseBn
from dataset import PreDataset, TrainDataset, ReDataset

parser = argparse.ArgumentParser()
parser.add_argument('nstd',type=int,help='noise level, from 0-255')
parser.add_argument('type',help='Get popultation stat for pretrain, train or refine')
parser.add_argument('-m',default=None,help='Path to saved model.npz file, default to the latest model')
parser.add_argument('-w',type=int,default=31,help='Window size')
opts = parser.parse_args()

#########################################################################

IMSZ = 304   # Sizes of original images to crop
niter = 100
mfile = opts.m

# Get the latest model if mfile is None
if mfile is None:
    msave = ut.ckpter('wts/nstd_%d/%s/iter_*.model.npz'%(opts.nstd, opts.type))
    mfile = msave.latest

if opts.type == 'train':
    tlist = 'data/train.txt'
    bsz = 512
    wsz = opts.w
    nstd = opts.nstd/255.
    net.toTest()
    net.base = baseBn
    model = net.Net()

    tset = TrainDataset(tlist,bsz,IMSZ,wsz,net.csz,net.psz,nstd,0)
    _ = model.encode(tset.pnzbatch)

elif opts.type == 'regress':
    tlist = os.path.dirname(mfile) + '/train.txt'
    bsz = 16
    rnet.base = baseBn
    model = rnet.Net()

    tset = ReDataset(tlist,bsz,IMSZ,0)
    _ = model.refine(tset.mzbatch)

elif opts.type == 'pretrain':
    tlist = 'data/train.txt'
    bsz = 16
    wsz = opts.w
    nstd = opts.nstd/255.
    net.toTest()
    net.base = baseBn
    model = net.Net()

    tset = PreDataset(tlist,bsz,IMSZ,0)
    img = tset.batch

    imnz = img + tf.random_normal(img.get_shape(),stddev=param.nstd)
    padding = (net.csz - net.psz)//2
    pimnz = tf.pad(imnz,[[0,0],[padding,padding],[padding,padding],[0,0]],'REFLECT')
    _ = model.encode(pimnz)


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
for k in model.weights.keys():
    wvar = model.weights[k]
    wk = wts[k].reshape(wvar.get_shape())
    sess.run(wvar.assign(ph),feed_dict={ph: wk})

# Get population stat
bnnms, bnwts = list(model.bnwts.keys()), list(model.bnwts.values())
bns = {k: [] for k in bnnms}
for i in range(niter):
    sess.run(tset.fetchOp,feed_dict=tset.fdict())
    tbns = sess.run(bnwts)
    for j in range(len(bnnms)):
        nm = bnnms[j]
        bns[nm].append(tbns[j])

# average over all batches
for nm in bnnms:
    if '_v' in nm:
        continue
    vnm = nm.replace('_mu', '_v')

    mean = np.mean(np.stack(bns[nm], axis=0), axis=0)
    mean_var = np.var(np.stack(bns[nm], axis=0), axis=0)
    var = np.mean(np.stack(bns[vnm], axis=0), axis=0)

    bns[nm] = mean
    bns[vnm] = mean_var + var

# Save population stat
ofile = mfile.replace('model','bnwts')
np.savez(ofile, **bns)