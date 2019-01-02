#!/usr/bin/env python3

import os
import importlib, argparse
import numpy as np
import tensorflow as tf

import utils as ut

import net.rnet as net
from dataset import ReDataset as Dataset

parser = argparse.ArgumentParser()
parser.add_argument('nstd',type=int,help='noise level, from 0-255')
opts = parser.parse_args()

#########################################################################

# Checkpoints and training log will be saved to
wts = 'wts/nstd_%d/regress'%(opts.nstd)

TLIST=wts + '/train.txt'
VLIST=wts + '/dev.txt'
BSZ = 16

IMSZ = 304   # Sizes of original images to crop

LR = 1e-4 # Learning rate, will be dropped at 400k and 500k by sqrt(10.) each time

VALFREQ = 1e3
SAVEFREQ = 1e4
MAXITER = 6e5

#########################################################################

# Check for saved weights & optimizer states
msave = ut.ckpter(wts + '/iter_*.model.npz')
ssave = ut.ckpter(wts + '/iter_*.state.npz')
ut.logopen(wts+'/refine.log')
niter = msave.iter

#########################################################################

# Setup Graphs
model = net.Net()

#### Images loading setup
tset = Dataset(TLIST,BSZ,IMSZ,niter)
vset = Dataset(VLIST,BSZ,IMSZ,0,True)

img, immz, swpT, swpV = tset.tvSwap(vset)

#### Get residual
res = model.refine(immz)

#### Define Losses
loss = tf.reduce_mean(tf.square(immz[:,:,:,3:]+res-img))

#### Set up optimizer
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

#########################################################################
# Load saved weights if any
if niter > 0:
    mfn = wts+"/iter_%06d.model.npz" % niter
    sfn = wts+"/iter_%06d.state.npz" % niter

    ut.mprint("Restoring model from " + mfn )
    ut.loadNet(mfn,model,sess)
    ut.mprint("Restoring state from " + sfn )
    ut.loadAdam(sfn,opt,model.weights,sess)
    ut.mprint("Done!")


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
            outs = sess.run([loss,vset.fetchOp],feed_dict=vset.fdict())
            vloss = vloss + np.float32(outs[0])
        vloss = vloss/np.float32(val_iter)
        ut.vprint(niter,['v'],[vloss])
        ut.vprint(niter, ['lr'], [LR])
        

    ## Run training step and print losses
    sess.run(swpT)
    outs = sess.run([loss,tStep,tset.fetchOp],feed_dict=tset.fdict())
    ut.vprint(niter,['t'],[outs[0]])

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
