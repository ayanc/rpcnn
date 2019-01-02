import tensorflow as tf
import net.base as base

# Assumptions
#  1. first layer must have one even sized filter
#  2. product of estr's is psz
#  3. Have dilations only after you've sub-sampled
#  4. Before the first time you downsample, enough layers
#     to bring "pleft" down to 0.
#
# At test time, replace strides with dils for fcn.


nFeat  = [16, 24, 48, 96,  64, 32, 0, 64, 64,  0]
nLyrs  = [ 1,  1,  1,   1,  2,  2, 0,  3,  3,   0]
ksizes = [ 2,  3,  3,   3,  3,  3, 0,  1,  1,   0]

ifbn   = [ 0,  0,  0,   0,  2,  2, 0,  2,  0,   0] # 1 for instance norm
dils   = [ 1,  1,  1,   1,  1,  1, 0,  1,  1,   0]
estr   = [ 1,  1,  1,   1,  1,  8, 0,  1,  1,   0]

dcons = [ [0, 0, 0, 8, 8, 1],
          [0, 0, 0, 0, 0, 0, 1, 1, 1] ]
doffs = [ [0, 0, 0, 4, 2, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
# Values above hard-coded for these
psz = 8
ncmp = 30
csz = 16

cmpFeat = 256
nCLyr = [128, 64,32]

stride = 8

def toTest():
    global stride, estr, dcons
    stride = 1
    estr   = [ 1,  1,  1,   1,  1,  1, 0,  1,  1,   0]
    dcons = [ [0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 1] ]


class Net:
    def __init__(self):
        base.init(self)
        
    # Encode images to feature tensor
    def encode(self, img):

        out = 2*img-1.0
        nch, idx, didx = 3, 1, 0

        groups = []
        for g in range(len(nFeat)):
            
            if nLyrs[g] == 0:
                nch = 0
                out = []
                dense, off = dcons[didx], doffs[didx]
                didx = didx + 1
                for i in range(g):
                    if dense[i] > 0:
                        nch = nch + nFeat[i]
                        if off[i] == 0:
                            out.append(groups[i][:,::dense[i],::dense[i],:])
                        elif off[i] > 0:
                            out.append(groups[i][:,off[i]:-off[i]:dense[i],off[i]:-off[i]:dense[i],:])
                out = tf.concat(out,-1)
                groups.append(out)
                nFeat[g] = nch
                continue
            
            for lyr in range(nLyrs[g]):
                name = "enc%d" % idx
                idx = idx + 1

                ksz = [ksizes[g],nch,nFeat[g]]
                nch = ksz[-1]

                dil = dils[g]
                dsz = (ksz[0]-1)*dil+1

                if lyr == nLyrs[g]-1 and estr[g] > 1:
                    out = base.conv(self,name,out,ksz,stride=estr[g],ifbn=ifbn[g])
                else:
                    out = base.conv(self,name,out,ksz,dil,ifbn=ifbn[g])

            groups.append(out)
                    

        outR1 = base.conv(self,'outR1',out,[1,nch,cmpFeat],relu=False)
        outR2 = base.conv(self,'outR2',out,[1,nch,cmpFeat],bias=False,relu=False)

        return outR1,outR2

    # Compare two feature tensors
    # Interpret outputs as two logit tensors (to be sent to sigmoid)
    # Of ref1 matching to ref0 and vice-versa
    def compare(self, ref0, ref1):

        out = tf.nn.relu(ref0+ref1)

        nch = cmpFeat
        for i in range(len(nCLyr)):
            out = base.conv(self,'cmp%d'%i,out,[1,nch,nCLyr[i]])
            nch=nCLyr[i]

        out = base.conv(self,'cmpF',out,[1,nch,ncmp],relu=False)
        
        return out
