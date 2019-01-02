import tensorflow as tf
import numpy as np

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


# Patch -> Coeffs
# Return a list of tensors for each coef, and the mean component
# For each level, the tensor was of shape [bsz, 1, (psz/2^lev)*(psz/2^lev), 9]
def decompose(patch):
    bsz, psz = patch.get_shape().as_list()[0:2]
    coef, mc, nms = [], r2y(patch), []
    
    lev = 1
    while psz > 1:
        psz = psz // 2
        dc = tf.nn.depthwise_conv2d(mc, S, [1,2,2,1], 'VALID')
        coef.append(tf.reshape(dc, [bsz,1,-1,9]))
        mc = tf.nn.depthwise_conv2d(mc, L, [1,2,2,1], 'VALID')

        lstr = "%d"%lev
        nms = nms + [lstr + nm for nm in lnms]
        lev = lev+1

    coef.append(tf.reshape(mc, [bsz,1,-1,3]))

    return coef, nms

# Main loss function. Computes matching loss for individual
# wavelet components given appropriate logits
def loss(logits, imR, nzR, nzM):
    """
    Call with
       logits: [bsz,nps,1,nCmp]
       imR: [bsz,psz,psz,nch]
       nzR: [bsz,psz,psz,nch]
       nzM: [bsz*nps,psz,psz,nch]
    """

    bsz, nps = logits.get_shape().as_list()[0:2]
    psz, nch = imR.get_shape().as_list()[2:]

    imcfR, nzcfR = decompose(imR)[0], decompose(nzR)[0]
    nzcfM, nms = decompose(nzM)

    ps = tf.sigmoid(logits)

    pos, loss = 0, []
    for i in range(len(imcfR)):
        ncf = imcfR[i].get_shape().as_list()[-1]

        dzcf = tf.reshape(nzcfM[i], [bsz,nps,-1,ncf])
        dzcf = dzcf * ps[:,:,:,pos:pos+ncf]
        dzcf = tf.reduce_sum(dzcf,axis=1, keepdims=True) + nzcfR[i]
        psum = tf.reduce_sum(ps[:,:,:,pos:pos+ncf], axis=1, keepdims=True) + 1.0

        dzcf = dzcf/psum
        err = tf.reduce_sum(tf.square(dzcf-imcfR[i]), axis=[1,2])
        err = tf.reduce_mean(err, axis=0)
        loss.append(err)

        pos = pos + ncf

    loss = tf.concat(loss, axis=0)
    total = tf.reduce_sum(loss, keepdims=True)

    return total, ['0total']+nms, \
              tf.concat([total/(psz*psz*nch),loss],0)










