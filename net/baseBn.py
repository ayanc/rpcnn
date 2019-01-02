import tensorflow as tf
import numpy as np

def init(self):
    self.weights = {}
    self.bnwts = {}


def maxpool(inp,ksz,pad=0,dil=1,stride=1):
    if pad > 0:
        inp = tf.pad(inp,[[0,0],[pad,pad],[pad,pad],[0,0]])
    return tf.nn.pool(inp,[ksz,ksz],'MAX','VALID',[dil,dil],[stride,stride])

    
def conv(self,name,inp,ksz,dil=1,stride=1,pad=0,relu=True,bias=True,ifbn=0):
    ksz = [ksz[0],ksz[0],ksz[1],ksz[2]]
    assert(dil == 1 or stride == 1)

    wnm = name + "_w"
    if wnm in self.weights.keys():
        w = self.weights[wnm]
    else:
        sq = np.sqrt(3.0 / np.float32(ksz[0]*ksz[1]*ksz[2]))
        w = tf.Variable(tf.random_uniform(ksz,minval=-sq,maxval=sq,dtype=tf.float32))
        self.weights[wnm] = w

    if pad not in ['SAME', 'VALID']:
        if pad != 0:
            inp = tf.pad(inp,[[0,0],[pad,pad],[pad,pad],[0,0]])
        pad = 'VALID'

    if dil > 1:
        out=tf.nn.atrous_conv2d(inp,w,dil,pad)
    else:
        out=tf.nn.conv2d(inp,w,[1,stride,stride,1],pad)
        
    if ifbn == 1: # instance norm
        mu, v = tf.nn.moments(out,[1, 2],keep_dims=True)
        out = (out-mu) / tf.sqrt(v + 1e-12)
    elif ifbn == 2: # batch norm
        mu, v = tf.nn.moments(out,[0, 1, 2],keep_dims=True)
        munm, vnm = name+'_mu', name+'_v'
        self.bnwts[munm], self.bnwts[vnm] = mu, v
        out = (out-mu) / tf.sqrt(v + 1e-12)

    if bias:
        bnm = name + "_b"
        if bnm in self.weights.keys():
            b = self.weights[bnm]
        else:
            b = tf.Variable(tf.constant(0,shape=[ksz[-1]],dtype=tf.float32))
            self.weights[bnm] = b
        out = out+b

    if relu:
        out = tf.nn.relu(out)

    return out
