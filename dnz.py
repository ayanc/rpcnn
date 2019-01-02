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

def y2r(inp):
    return tf.nn.conv2d(inp,tf.transpose(r2y_,[0,1,3,2]), \
                        [1,1,1,1],'VALID')

# Harr Filters
L = np.zeros([2,2,3,1],dtype=np.float32)
for i in range(3):
    L[:,:,i,0] = [[1.,1.],[1.,1.]]
L = tf.constant(L/2.)

S  = np.zeros([2,2,3,3],dtype=np.float32)
for i in range(3):
    S[:,:,i,0] = [[-1.,1.],[-1.,1.]]
    S[:,:,i,1] = [[-1.,-1.],[1.,1.]]
    S[:,:,i,2] = [[-1.,1.],[1.,-1.]]
S = tf.constant(S/2.)

# Image -> Coeffs
# Return a list of tensors for each coef, and the mean component
# For each coef, the tensor has shape [bsz, ny, nx, (psz/2^lev)^2]
# where ny*nx = npatches
def decompose(im, psz, stride=1):
    coef, mc, csizes = [], r2y(im), []

    mc = tf.extract_image_patches(mc,[1,psz,psz,1],[1,stride,stride,1],[1,1,1,1],'VALID')
    mcsz = tf.shape(mc)
    ny,nx = mcsz[1],mcsz[2]
    mc = tf.reshape(mc,[-1,psz,psz,3])

    while psz > 1:
        psz = psz // 2
        dc = tf.nn.depthwise_conv2d(mc, S, [1,2,2,1], 'VALID')
        oshp = tf.stack([1,ny,nx,psz*psz*9])
        coef.append(tf.reshape(dc,oshp))
        csizes.append([psz,9])
        mc = tf.nn.depthwise_conv2d(mc, L, [1,2,2,1], 'VALID')

    oshp = tf.stack([1,ny,nx,psz*psz*3])
    coef.append(tf.reshape(mc,oshp))
    csizes.append([1,3])

    return coef, csizes


# Coeffs -> Image
def accumulate(coef, psz, stride=1):
    mc, coef = coef[-1], coef[:-1]
    
    mcsz = tf.shape(mc)
    ny, nx = mcsz[1], mcsz[2]
    npatch, csz = ny*nx, 1

    mc = tf.reshape(mc,[-1,1,1,3])
    for i in range(len(coef)-1,-1,-1):
        co = tf.reshape(coef[i], [-1,csz,csz,9])
        csz = csz*2
        oshp = tf.stack([npatch,csz,csz,3])
        mc = tf.nn.depthwise_conv2d_native_backprop_input(oshp,S,co,[1,2,2,1], 'VALID') + \
             tf.nn.depthwise_conv2d_native_backprop_input(oshp,L,mc,[1,2,2,1], 'VALID')

    mc = tf.reshape(mc, tf.stack([1,ny,nx,-1]))
    eipF = tf.constant(np.float32(np.eye(psz*psz)),shape=[psz,psz,1,psz*psz],dtype=tf.float32)

    oshape = tf.stack([1,ny*stride+psz-1,nx*stride+psz-1,1])
    norm = tf.ones(shape=[1,ny,nx,psz*psz], dtype=tf.float32)
    norm = tf.nn.conv2d_transpose(norm,eipF,oshape,[1,stride,stride,1],'VALID')

    img = []
    for i in range(3):
        im = tf.nn.conv2d_transpose(mc[:,:,:,i::3],eipF,oshape,[1,stride,stride,1],'VALID')
        img.append(im/norm)

    return y2r(tf.concat(img,-1))

# Functions for broadcasting nch to psz*psz*nch
def broadmul(coef,ps,csz):
    out = tf.reshape(coef,[-1,csz[0]**2,csz[1]]) * tf.reshape(ps,[-1,1,csz[1]])
    return tf.reshape(out,tf.shape(coef))

def broaddiv(coef,ps,csz):
    out = tf.reshape(coef,[-1,csz[0]**2,csz[1]]) / tf.reshape(ps,[-1,1,csz[1]])
    return tf.reshape(out,tf.shape(coef))

# Window size padding
def windowPad(tensor, wsz):
    npad = (wsz-1)//2
    upad, lpad = tensor[:,npad+1:(2*npad+1),:,:], tensor[:,:,npad+1:(2*npad+1),:]
    dpad, rpad = tensor[:,-(2*npad+1):-(npad+1),:,:], tensor[:,:,-(2*npad+1):-(npad+1),:]
    
    up_lpad, do_lpad = lpad[:,npad+1:(2*npad+1),:,:], lpad[:,-(2*npad+1):-(npad+1),:,:]
    lpad = tf.concat([up_lpad,lpad,do_lpad],axis=1)
    up_rpad, do_rpad = rpad[:,npad+1:(2*npad+1),:,:], rpad[:,-(2*npad+1):-(npad+1),:,:]
    rpad = tf.concat([up_rpad,rpad,do_rpad],axis=1)
    
    padded = tf.concat([upad,tensor,dpad],axis=1)
    padded = tf.concat([lpad,padded,rpad],axis=2)
    return padded


MAXPIXELS=1024*1024
def saver(val,ops):
    nch = val.get_shape().as_list()[-1]
    var = tf.Variable(tf.zeros([MAXPIXELS*nch],dtype=tf.float32))
    svar = tf.Variable(tf.zeros([4],dtype=tf.int32)) # shape of effective var: HxWxC

    shape = tf.shape(val)
    prod = tf.reduce_prod(shape[1:])
    ops.append(tf.assign(var[:prod], tf.reshape(val,[-1])).op)
    ops.append(tf.assign(svar,shape).op)

    vprod = var[:tf.reduce_prod(svar[1:])]
    val2 = tf.reshape(vprod, svar)
    val2.adder = lambda x: tf.assign(vprod, vprod+tf.reshape(x,[-1])).op

    return val2

class Denoiser:
    def __init__(self,model,psz,csz,wsz,stride=1,liter=1):
        self.liter = liter
        yshifts, xshifts = np.meshgrid(np.arange(wsz), np.arange(wsz))
        self.yshifts, self.xshifts = yshifts.flatten(), xshifts.flatten()

        # Define placeholders
        self.imnz = tf.placeholder(dtype=tf.float32,shape=[None,None,3])
        imnz = tf.expand_dims(self.imnz,0)
        
        self.ys, self.xs = tf.placeholder(dtype=tf.int32,shape=[liter]), tf.placeholder(dtype=tf.int32,shape=[liter])

        # Get feats and coefs
        padding = (csz - psz)//2
        pimnz = tf.pad(imnz,[[0,0],[padding,padding],[padding,padding],[0,0]],'REFLECT')
        feat, featM = model.encode(pimnz)
        pFeatM = windowPad(featM, wsz)
        coef, csizes = decompose(imnz, psz, stride=stride)
        pCoef = [windowPad(c, wsz) for c in coef]

        # Define variables and placeholders
        initOps = []

        self.feat = saver(feat,initOps)
        self.pFeatM = saver(pFeatM,initOps)
        self.coef = [saver(c,initOps) for c in coef]
        self.pCoef = [saver(c,initOps) for c in pCoef]
        self.cout = [saver(c,initOps) for c in coef]

        cshape = tf.shape(feat)
        fh, fw = cshape[1],cshape[2]
        self.psum = [saver(tf.ones(tf.stack([1,fh,fw,csizes[i][1]])),initOps) for i in range(len(coef))]

        self.initOps = initOps

        cshape = tf.shape(self.feat)
        fh, fw = cshape[1],cshape[2]

        #### Denoise ops
        deOps = []
        cout_, psum_ = [[] for i in range(len(coef))],  [[] for i in range(len(coef))]
        for its in range(liter):
            y, x = self.ys[its], self.xs[its]
            logits = model.compare(self.feat,self.pFeatM[:, y:y+fh, x:x+fw, :])
            ps = tf.sigmoid(logits)

            pos = 0
            for i in range(len(coef)):
                nch = csizes[i][1]
                psi = ps[:,:,:,pos:pos+nch]
                pos = pos+nch

                cout_[i].append(broadmul(self.pCoef[i][:,y:y+fh,x:x+fw,:], psi, csizes[i]))
                psum_[i].append(psi)

        for i in range(len(coef)):
            deOps.append(self.cout[i].adder(tf.add_n(cout_[i])))
            deOps.append(self.psum[i].adder(tf.add_n(psum_[i])))
        self.deOps = deOps

        #### Define final denoised image
        dcoef = [broaddiv(self.cout[i],self.psum[i],csizes[i]) for i in range(len(coef))]
        self.dnz = tf.squeeze(accumulate(dcoef, psz, stride=stride),0)


    def run(self,imnz,sess):
        # Initialize feats and other variables
        sess.run(self.initOps, feed_dict={self.imnz: imnz})

        # Run denoise ops
        liter = self.liter
        niter = len(self.yshifts)//liter
        for i in range(niter):
            ys, xs = self.yshifts[i*liter:(i+1)*liter], self.xshifts[i*liter:(i+1)*liter]
            sess.run(self.deOps, feed_dict={self.ys:ys, self.xs:xs})

        return sess.run(self.dnz)