#!/usr/bin/env python3

import os
import argparse
import numpy as np
import tensorflow as tf

import utils as ut

import net.net as net
import loss.pretrain_loss as imp
from dataset import PreDataset as Dataset


parser = argparse.ArgumentParser()
parser.add_argument('nstd',type=int,help='noise level, from 0-255')
opts = parser.parse_args()

#########################################################################

# Params

TLIST='data/train.txt'
VLIST='data/dev.txt'
BSZ = 16

IMSZ = 304   # Sizes of original images to crop
NSTD = opts.nstd/255.   # Train for this noise level

LR = 1e-3

VALFREQ = 1e2
SAVEFREQ = 1e4
MAXITER = 1e5

# Checkpoints and training log will be saved to
wts = 'wts/nstd_%d/pretrain'%(opts.nstd)

if not os.path.exists(wts):
    os.makedirs(wts)

#########################################################################

# Check for saved weights & optimizer states
msave = ut.ckpter(wts + '/iter_*.model.npz')
ssave = ut.ckpter(wts + '/iter_*.state.npz')
ut.logopen(wts+'/pretrain.log')
niter = msave.iter

#########################################################################

# Setup Graphs
model = net.Net()

#### Image loading setup, add noise
tset = Dataset(TLIST,BSZ,IMSZ,niter)
vset = Dataset(VLIST,BSZ,IMSZ,0,True)

img, swpT, swpV = tset.tvSwap(vset)

imnz = img + tf.random_normal(img.get_shape(),stddev=NSTD)

#### Pad, encode, crop, compare
padding = (net.csz - net.psz)//2
pimnz = tf.pad(imnz,[[0,0],[padding,padding],[padding,padding],[0,0]],'REFLECT')
featR,featM = model.encode(pimnz)
featR2,featM2,img2,imnz2 = imp.jShuffle([featR,featM,img,imnz],[1,1,net.psz,net.psz])

p12 = model.compare(featR,featM2)
p21 = model.compare(featR2,featM)

#### Define Losses
loss, lnms, lvals = imp.loss(p12,p21,img,imnz,img2,imnz2,net.psz)
tnames,vnames = [nm+".t" for nm in lnms], [nm+".v" for nm in lnms]

#### Set up optimizer
opt = tf.train.AdamOptimizer(LR)
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

while niter < MAXITER and not ut.stop:

    ## Validate model every so often
    if niter % VALFREQ == 0:
        ut.mprint("Validating model")
        val_iter = vset.ndata // BSZ
        vloss, oloss, vset.niter = 0., 0., 0
        sess.run(vset.fetchOp,feed_dict=vset.fdict())
        for its in range(val_iter):
            sess.run(swpV)
            outs = sess.run([lvals,vset.fetchOp],feed_dict=vset.fdict())
            vloss = vloss + np.float32(outs[0])
        vloss = list(vloss/np.float32(val_iter))
        ut.vprint(niter,vnames,vloss)
        
    ## Run training step and print losses
    sess.run(swpT)
    if niter % VALFREQ == 0:
        outs = sess.run([lvals,tStep,tset.fetchOp],feed_dict=tset.fdict())
        ut.vprint(niter,tnames,outs[0])
        ut.vprint(niter, ['1lr'], [LR])
        ut.vprint(niter, ['1nstd'], [NSTD])
    else:
        outs = sess.run([loss,tStep,tset.fetchOp],feed_dict=tset.fdict())
        ut.vprint(niter,[tnames[0]],[outs[0]])
        
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
