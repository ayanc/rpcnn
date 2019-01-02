import tensorflow as tf
import net.base as base

nFeat = [64, 64, 64, 64, 64, 64, 3]
nLyrs = [1, 1, 1, 1, 1, 1,1]
ksizes = [3, 3, 3, 3, 3, 3, 3]

ifbn = [0, 2, 2, 2, 2, 2, 0]
dils = [1, 2, 3, 4, 3, 2, 1]
estr = [1, 1, 1, 1, 1, 1, 1]

# Values above hard-coded for these
rsz = 33 # receptive field for padding
assert rsz%2 == 1

class Net:
    def __init__(self):
        base.init(self)

    def refine(self, immz):
        out = 2*immz - 1.0
        nch, idx = 6, 1

        # symmetric pad the image
        pad = (rsz-1)//2
        out = tf.pad(out,[[0,0],[pad,pad],[pad,pad],[0,0]],'reflect')

        for g in range(len(nFeat)):

            for lyr in range(nLyrs[g]):
                name = "conv%d" % idx
                idx = idx + 1

                ksz = [ksizes[g],nch,nFeat[g]]
                nch = ksz[-1]
                dil = dils[g]

                if g == len(nFeat)-1 and lyr == nLyrs[g]-1:
                    out = base.conv(self,name,out,ksz,dil=dil,stride=estr[g],relu=False)
                else:
                    out = base.conv(self,name,out,ksz,dil=dil,stride=estr[g],ifbn=ifbn[g])

        return out
