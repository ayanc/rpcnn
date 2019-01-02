#!/usr/bin/env python3

import os
import argparse
from glob import glob
import re
import numpy as np
import tensorflow as tf

import utils as ut

import net.net as net
import loss.train_loss as imp
from dataset import TrainDataset as Dataset

parser = argparse.ArgumentParser()
parser.add_argument('nstd',type=int,help='noise level, from 0-255')
parser.add_argument('-w',type=int,default=31,help='Window size')
opts = parser.parse_args()

#########################################################################

# Params

TLIST='data/train.txt'
VLIST='data/dev.txt'
BSZ = 256
WSZ = opts.w

IMSZ = 304   # Sizes of original images to crop
NSTD = opts.nstd/255.   # Train for this noise level

LR = 1e-3 # Learning rate, will be dropped at 400k and 500k by sqrt(10.) each time

VALFREQ = 1e3
SAVEFREQ = 5e3
MAXITER = 60e4

# Checkpoints and training log will be saved to
wts = 'wts/nstd_%d/train'%(opts.nstd)

if not os.path.exists(wts):
    os.makedirs(wts)

#########################################################################

# Check for saved weights & optimizer states
msave = ut.ckpter(wts + '/iter_*.model.npz')
ssave = ut.ckpter(wts + '/iter_*.state.npz')
ut.logopen(wts+'/train.log')
niter = msave.iter

#########################################################################

# Setup Graphs
net.toTest()
model = net.Net()

CSZ = net.csz
PSZ = net.psz
assert WSZ%2 == 1
assert CSZ%2 == 0 and PSZ%2 == 0

#### Image loading setup, add noise
tset = Dataset(TLIST,BSZ,IMSZ,WSZ,CSZ,PSZ,NSTD,niter)
vset = Dataset(VLIST,BSZ,IMSZ,WSZ,CSZ,PSZ,NSTD,0,True)

img, imnz, pimnz, swpT, swpV = tset.tvSwap(vset)

nps = WSZ*WSZ # Number of patches we extract

#### Encode to get features, slice to get ref and matching patches
featR, featM = model.encode(pimnz)

featR = tf.slice(featR, [0,(WSZ-1)//2,(WSZ-1)//2,0], [-1,1,1,-1])
imR = tf.slice(img, [0,(WSZ-1)//2,(WSZ-1)//2,0], [-1,PSZ,PSZ,-1])
nzR = tf.slice(imnz, [0,(WSZ-1)//2,(WSZ-1)//2,0], [-1,PSZ,PSZ,-1])

fsize = featM.get_shape().as_list()[-1]
featM = tf.reshape(featM, [BSZ,nps,1,fsize])

nch = imnz.get_shape().as_list()[-1]
nzM = tf.extract_image_patches(imnz, [1,PSZ,PSZ,1], [1,1,1,1], [1,1,1,1], 'VALID')
nzM = tf.reshape(nzM, [BSZ,nps,PSZ*PSZ*nch])
nzM = tf.reshape(nzM, [BSZ*nps,PSZ,PSZ,nch])

#### Get logits
logits = model.compare(featR, featM) # [bsz,nps,1,nCmp]

#### Define Losses
loss, lnms, lvals = imp.loss(logits, imR, nzR, nzM)
tnames,vnames = [nm+".t" for nm in lnms], [nm+".v" for nm in lnms]

#### Set up optimizer, only fine-tune compare-net
lr = tf.Variable(LR, trainable=False)
opt = tf.train.AdamOptimizer(lr)
tStep = opt.minimize(loss,var_list=list(model.weights.values()))

#########################################################################
# Start TF session (respecting OMP_NUM_THREADS)
nthr = os.getenv('OMP_NUM_THREADS')
if nthr is None:
    sess = tf.Session()
else:
    sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=int(nthr)))
sess.run(tf.global_variables_initializer())

########################################################################
# Load saved weights if any
if niter > 0:
    mfn = wts+"/iter_%06d.model.npz" % niter
    sfn = wts+"/iter_%06d.state.npz" % niter

    ut.mprint("Restoring model from " + mfn )
    ut.loadNet(mfn,model,sess)
    ut.mprint("Restoring state from " + sfn )
    ut.loadAdam(sfn,opt,model.weights,sess)
    ut.mprint("Done!")

else:
    # Load the last model in pretraining
    wcard = os.path.dirname(wts)+"/pretrain/iter_*.model.npz"
    lst=[(l,int(re.match('.*/.*_(\d+)',l).group(1))) for l in glob(wcard)]
    mfn = max(lst, key=lambda x: x[1])[0]
    ut.mprint("Start with pre-trained model " + mfn )
    ut.loadNet(mfn,model,sess)

#########################################################################
# Main Training loop

stop=False
ut.mprint("Starting from Iteration %d" % niter)
sess.run(tset.fetchOp,feed_dict=tset.fdict())

## Learning rate, if resume at an iteration after drop
if niter >= 4e5 and niter < 5e5:
    sess.run(tf.assign(lr, LR/np.sqrt(10.0)))
elif niter == 5e5:
    sess.run(tf.assign(lr, LR/10.0))

while niter < MAXITER and not ut.stop:

    ## Validate model every so often
    if niter % VALFREQ == 0:
        ut.mprint("Validating model")
        val_iter = vset.ndata // BSZ
        vloss, vset.niter = 0., 0
        sess.run(vset.fetchOp,feed_dict=vset.fdict())
        for its in range(val_iter):
            sess.run(swpV)
            outs = sess.run([lvals,vset.fetchOp],feed_dict=vset.fdict())
            vloss = vloss + np.float32(outs[0])
        vloss = list(vloss/np.float32(val_iter))
        ut.vprint(niter,vnames,vloss)
        
    ## Run training step and print losses
    sess.run(swpT)
    outs = sess.run([lvals,tStep,tset.fetchOp],feed_dict=tset.fdict())
    if niter % VALFREQ == 0:
        ut.vprint(niter,tnames,outs[0])
        ut.vprint(niter, ['1lr'], [LR])
        ut.vprint(niter, ['1nstd'], [NSTD])
    else:
        ut.vprint(niter,[tnames[0]],[outs[0][0]])
        
    niter=niter+1
                    
    ## Save model weights if needed
    if SAVEFREQ > 0 and niter % SAVEFREQ == 0:
        mfn = wts+"/iter_%06d.model.npz" % niter
        sfn = wts+"/iter_%06d.state.npz" % niter

        ut.mprint("Saving model to " + mfn )
        ut.saveNet(mfn,model,sess)
        ut.mprint("Saving state to " + sfn )
        ut.saveAdam(sfn,opt,model.weights,sess)
        ut.mprint("Done!")
        msave.clean(every=SAVEFREQ,last=1)
        ssave.clean(every=SAVEFREQ,last=1)

    ## Learning rate drop
    if niter == 4e5:
        sess.run(tf.assign(lr, LR/np.sqrt(10.0)))
    elif niter == 5e5:
        sess.run(tf.assign(lr, LR/10.0))

# Save last
if msave.iter < niter:
    mfn = wts+"/iter_%06d.model.npz" % niter
    sfn = wts+"/iter_%06d.state.npz" % niter

    ut.mprint("Saving model to " + mfn )
    ut.saveNet(mfn,model,sess)
    ut.mprint("Saving state to " + sfn )
    ut.saveAdam(sfn,opt,model.weights,sess)
    ut.mprint("Done!")
    msave.clean(every=SAVEFREQ,last=1)
    ssave.clean(every=SAVEFREQ,last=1)













