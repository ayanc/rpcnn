# Absolute loss

import tensorflow as tf
import numpy as np

###################################################################################

# Shuffle non-overlapping blocks of tensors jointly
def jShuffle(tensors, sizes):
    sz = tf.shape(tensors[0])
    sz = tf.cast(sz[1]*sz[2]//(sizes[0]**2),tf.int64)
    idx = tf.random_shuffle(tf.range(sz,dtype=tf.int64))
    
    crops = []
    for j in range(len(tensors)):
        crop = tensors[j]
        if sizes[j] > 1:
            crop = tf.space_to_depth(crop,sizes[j])
        cshp = tf.shape(crop)
        crop = tf.reshape(crop,tf.stack([cshp[0],-1,cshp[-1]]))
        crop = tf.gather(crop,idx,axis=1)
        crop = tf.reshape(crop,cshp)
        if sizes[j] > 1:
            crop = tf.depth_to_space(crop,sizes[j])
        crops.append(crop)
        
    return crops
    

###################################################################################


# Color Transform
r2y_= tf.constant(np.float32([
    0.57735002, -0.57735002, -0.57735002,
    0.57735002,  0.78867501, -0.211325,
    0.57735002, -0.211325,    0.78867501]),
    dtype=tf.float32,shape=[1,1,3,3])
                 
def r2y(inp):
    return tf.nn.conv2d(inp,r2y_,[1,1,1,1],'VALID')

# Harr Filters
L = np.zeros([2,2,3,1],dtype=np.float32)
for i in range(3):
    L[:,:,i,0] = [[1.,1.],[1.,1.]]
L = tf.constant(L/2.)

S = np.zeros([2,2,3,3],dtype=np.float32)
for i in range(3):
    S[:,:,i,0] = [[-1.,1.],[-1.,1.]]
    S[:,:,i,1] = [[-1.,-1.],[1.,1.]]
    S[:,:,i,2] = [[-1.,1.],[1.,-1.]]
S = tf.constant(S/2.)

# Name constants
lnms = ['yH','yV','yD','uH','uV','uD','vH','vV','vD']
dnms = ['yDC', 'uDC', 'vDC']

# Get tensor of matching errors between wavelet patches
# exy implies error of noisy patch x vs clean patch y.
# Returns tensors with num-feautres = 9 * log2(psz)
def get_errs(im1, nz1, im2, nz2, psz):
    e11,e12 = r2y(nz1-im1),r2y(nz2-im1)
    e21,e22 = r2y(nz1-im2),r2y(nz2-im2)
    
    e11s,e12s,e21s,e22s,nms = [],[],[],[],[]
    lev,dil = 1,1
    while psz > 1:
        psz = psz // 2

        e11d = tf.nn.depthwise_conv2d(e11,S,[1,2,2,1],'VALID')
        e11d = tf.nn.pool(tf.square(e11d),[psz,psz],'AVG','VALID',strides=[psz,psz])*np.float32(psz*psz)
        e11s.append(e11d)
    
        e12d = tf.nn.depthwise_conv2d(e12,S,[1,2,2,1],'VALID')
        e12d = tf.nn.pool(tf.square(e12d),[psz,psz],'AVG','VALID',strides=[psz,psz])*np.float32(psz*psz)
        e12s.append(e12d)

        e21d = tf.nn.depthwise_conv2d(e21,S,[1,2,2,1],'VALID')
        e21d = tf.nn.pool(tf.square(e21d),[psz,psz],'AVG','VALID',strides=[psz,psz])*np.float32(psz*psz)
        e21s.append(e21d)

        e22d = tf.nn.depthwise_conv2d(e22,S,[1,2,2,1],'VALID')
        e22d = tf.nn.pool(tf.square(e22d),[psz,psz],'AVG','VALID',strides=[psz,psz])*np.float32(psz*psz)
        e22s.append(e22d)

        lstr = "%d"%lev
        nms = nms + [lstr + nm for nm in lnms]
        
        e11 = tf.nn.depthwise_conv2d(e11,L,[1,2,2,1],'VALID')
        e12 = tf.nn.depthwise_conv2d(e12,L,[1,2,2,1],'VALID')
        e21 = tf.nn.depthwise_conv2d(e21,L,[1,2,2,1],'VALID')
        e22 = tf.nn.depthwise_conv2d(e22,L,[1,2,2,1],'VALID')
        lev = lev+1

    e11s.append(tf.square(e11))
    e12s.append(tf.square(e12))
    e21s.append(tf.square(e21))
    e22s.append(tf.square(e22))
        
    e11,e12 = tf.concat(e11s,-1),tf.concat(e12s,-1)
    e21,e22 = tf.concat(e21s,-1),tf.concat(e22s,-1)

    return e11,e12,e21,e22,nms+dnms


# Main loss function. Computes matching loss for individual
# wavelet components given appropriate logits
def loss(logits12, logits21, im1, nz1, im2, nz2, psz):
    e11,e12,e21,e22,nms = get_errs(im1, nz1, im2, nz2, psz)
    
    p12 = tf.sigmoid(logits12)
    p21 = tf.sigmoid(logits21)


    l12 = (e11+tf.square(p12)*e12) / tf.square(1.+p12)
    l12 = tf.reduce_mean(l12,[0,1,2])
    l21 = (e22+tf.square(p21)*e21) / tf.square(1.+p21)
    l21 = tf.reduce_mean(l21,[0,1,2])

    loss = 0.5*(l12+l21)
    total = tf.reduce_sum(loss,keep_dims=True)

    return total, ['0total']+nms, \
                  tf.concat([total,loss],0)
