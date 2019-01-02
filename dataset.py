import numpy as np
import tensorflow as tf


################## Dataset for pre-training #######################
class PreDataset:
    def graph(self):
        self.names = []
        # Create placeholders
        for i in range(self.bsz):
            self.names.append(tf.placeholder(tf.string))

        batch = []
        for i in range(self.bsz):
            img = tf.read_file(self.names[i])
            imsp = tf.image.extract_jpeg_shape(img)
            yoff, xoff = tf.random_uniform([], 0, imsp[0]-self.csz, dtype=tf.int32), tf.random_uniform([], 0, imsp[1]-self.csz, dtype=tf.int32)
            img = tf.image.decode_and_crop_jpeg(img, [yoff,xoff,self.csz,self.csz], channels=3)

            batch.append(img)

        batch = tf.to_float(tf.stack(batch))/255.0

        # Fetch op
        self.batch = tf.Variable(tf.zeros([self.bsz,self.csz,self.csz,3],dtype=tf.float32),trainable=False)
        self.fetchOp = tf.assign(self.batch,batch).op

    def fdict(self):
        fd = {}
        for i in range(self.bsz):
            idx = self.idx[self.niter % self.ndata]
            self.niter = self.niter + 1
            if self.niter % self.ndata == 0 and self.isrand:
                self.idx = np.int32(self.rand.permutation(self.ndata))
            fd[self.names[i]] = self.files[idx]
        return fd
        
    def __init__(self,lfile,bsz,csz,niter=0,isval=False):
        """
        Call with
           lfile = Path of file with list of image file names
           bsz = Batch size you want this to generate
           csz = Output images will be size csz x csz
           niter = Resume at niterations
           isval = Running on train or val (random crops and shuffling for train)
        """

        self.bsz = bsz
        self.csz = csz
        self.isrand = not isval

        # Setup fetch graph
        self.graph()

        # Load file list
        self.files = [l.strip() for l in open(lfile).readlines()]
        self.ndata = len(self.files)
        self.niter = niter*bsz
        
        # Setup shuffling
        if self.isrand:
            self.rand = np.random.RandomState(0)
            idx = self.rand.permutation(self.ndata)
            for i in range(niter // self.ndata):
                idx = self.rand.permutation(self.ndata)
            self.idx = np.int32(idx)
        else:
            self.idx = np.int32(np.arange(self.ndata))
                
    # Sets up a common batch variable for train and val and ops
    # to swap in pre-fetched image data.
    def tvSwap(self,vset):
        batch = tf.Variable(tf.zeros(self.batch.shape,dtype=tf.float32),trainable=False)
        tSwap = tf.assign(batch,tf.identity(self.batch)).op
        vSwap = tf.assign(batch,tf.identity(vset.batch)).op
        
        return batch,tSwap,vSwap


################## Dataset for training #######################
class TrainDataset:
    def graph(self):
        self.names = []
        # Create placeholders
        for i in range(self.bsz):
            self.names.append(tf.placeholder(tf.string))

        imbatch, nzbatch, pnzbatch =[], [], []
        imsz, wsz, csz, psz = self.imsz, self.wsz, self.csz, self.psz
        padding = (csz- psz)//2
        for i in range(self.bsz):
            y, x = tf.random_uniform([], 0, imsz-wsz-psz+2, dtype=tf.int32), tf.random_uniform([], 0, imsz-wsz-psz+2, dtype=tf.int32)
            ylow, xlow = tf.maximum(y-padding,0), tf.maximum(x-padding,0)
            yhigh, xhigh = tf.minimum(y+wsz+psz-1+padding,imsz), tf.minimum(x+wsz+psz-1+padding,imsz)

            ptop, pdown = padding-(y-ylow), padding-(yhigh-(y+wsz+psz-1))
            pleft, pright = padding-(x-xlow), padding-(xhigh-(x+wsz+psz-1))
            
            img = tf.read_file(self.names[i])
            imsp = tf.image.extract_jpeg_shape(img)
            yoff, xoff = tf.random_uniform([], 0, imsp[0]-imsz, dtype=tf.int32), tf.random_uniform([], 0, imsp[1]-imsz, dtype=tf.int32)

            img = tf.image.decode_and_crop_jpeg(img,[yoff+ylow,xoff+xlow,yhigh-ylow,xhigh-xlow],channels=3)
            img = tf.to_float(img) / 255.0

            imnz = img + tf.random_normal(tf.shape(img),stddev=self.nstd)

            pimnz = tf.pad(imnz,[[ptop,pdown],[pleft,pright],[0,0]],'REFLECT')
            img = tf.slice(img, [y-ylow,x-xlow,0], [wsz+psz-1,wsz+psz-1,-1])
            imnz = tf.slice(imnz, [y-ylow,x-xlow,0], [wsz+psz-1,wsz+psz-1,-1])

            imbatch.append(img)
            nzbatch.append(imnz)
            pnzbatch.append(pimnz)

        # Fetch op
        self.imbatch = tf.Variable(tf.zeros([self.bsz,wsz+psz-1,wsz+psz-1,3],dtype=tf.float32),trainable=False)
        self.nzbatch = tf.Variable(tf.zeros([self.bsz,wsz+psz-1,wsz+psz-1,3],dtype=tf.float32),trainable=False)
        self.pnzbatch = tf.Variable(tf.zeros([self.bsz,wsz+csz-1,wsz+csz-1,3],dtype=tf.float32),trainable=False)
        self.fetchOp = [tf.assign(self.imbatch,imbatch).op, tf.assign(self.nzbatch,nzbatch).op, tf.assign(self.pnzbatch,pnzbatch).op]

    def fdict(self):
        fd = {}
        for i in range(self.bsz):
            idx = self.idx[self.niter % self.ndata]
            self.niter = self.niter + 1
            if self.niter % self.ndata == 0 and self.isrand:
                self.idx = np.int32(self.rand.permutation(self.ndata))
            fd[self.names[i]] = self.files[idx]
        return fd
        
    def __init__(self,lfile,bsz,imsz,wsz,csz,psz,nstd,niter=0,isval=False):
        """
        Call with
           lfile = Path of file with list of image file names
           bsz = Batch size you want this to generate
           imsz = Output images will be size imsz x imsz
           niter = Resume at niterations
           isval = Running on train or val (random crops and shuffling for train)
        """

        self.bsz, self.imsz = bsz, imsz
        self.wsz, self.csz, self.psz = wsz, csz, psz
        self.nstd = nstd
        self.isrand = not isval

        # Setup fetch graph
        self.graph()

        # Load file list
        self.files = [l.strip() for l in open(lfile).readlines()]
        if len(self.files) < bsz: # repeat file list if its size < bsz
            self.files = self.files * int(np.ceil(float(bsz)/len(self.files)))
        self.ndata = len(self.files)
        self.niter = niter*bsz
        
        # Setup shuffling
        if self.isrand:
            self.rand = np.random.RandomState(0)
            idx = self.rand.permutation(self.ndata)
            for i in range(niter // self.ndata):
                idx = self.rand.permutation(self.ndata)
            self.idx = np.int32(idx)
        else:
            self.idx = np.int32(np.arange(self.ndata))

    # Sets up a common batch variable for train and val and ops
    # to swap in pre-fetched image data.
    def tvSwap(self,vset):
        imbatch = tf.Variable(tf.zeros(self.imbatch.shape,dtype=tf.float32),trainable=False)
        nzbatch = tf.Variable(tf.zeros(self.nzbatch.shape,dtype=tf.float32),trainable=False)
        pnzbatch = tf.Variable(tf.zeros(self.pnzbatch.shape,dtype=tf.float32),trainable=False)
        tSwap = [tf.assign(imbatch,tf.identity(self.imbatch)).op, tf.assign(nzbatch,tf.identity(self.nzbatch)).op, tf.assign(pnzbatch,tf.identity(self.pnzbatch)).op]
        vSwap = [tf.assign(imbatch,tf.identity(vset.imbatch)).op, tf.assign(nzbatch,tf.identity(vset.nzbatch)).op, tf.assign(pnzbatch,tf.identity(vset.pnzbatch)).op]
        
        return imbatch,nzbatch,pnzbatch,tSwap,vSwap



################## Dataset for final regression #######################
class ReDataset:
    def graph(self):
        self.names = []
        # Create placeholders
        for i in range(self.bsz):
            self.names.append(tf.placeholder(tf.string))

        imbatch, mzbatch = [], []
        for i in range(self.bsz):
            img = tf.read_file(self.names[i]+'.png')
            imnz, imdz = tf.read_file(self.names[i]+'_nz.png'), tf.read_file(self.names[i]+'_dnz.png')
            img = tf.image.decode_png(img,channels=1,dtype=tf.uint16)
            imsp = tf.shape(img)
            img = tf.reshape(img, [imsp[0],-1,3])
            imnz, imdz = tf.image.decode_png(imnz,channels=1,dtype=tf.uint16), tf.image.decode_png(imdz,channels=1,dtype=tf.uint16)
            imnz = tf.reshape(imnz, [imsp[0],-1,3])
            imdz = tf.reshape(imdz, [imsp[0],-1,3])

            immz = tf.concat([img,imnz,imdz], axis=2)
            immz = tf.cast(immz,tf.float32) / (2**16-1)*2.0 - 0.5
            immz = tf.random_crop(immz,[self.csz,self.csz,9])
            imbatch.append(immz[:,:,:3])
            mzbatch.append(immz[:,:,3:])

        imbatch, mzbatch = tf.stack(imbatch), tf.stack(mzbatch)

        # Fetch op
        self.imbatch = tf.Variable(tf.zeros([self.bsz,self.csz,self.csz,3],dtype=tf.float32),trainable=False)
        self.mzbatch = tf.Variable(tf.zeros([self.bsz,self.csz,self.csz,6],dtype=tf.float32),trainable=False)
        self.fetchOp = [tf.assign(self.imbatch,imbatch).op, tf.assign(self.mzbatch,mzbatch).op]

    def fdict(self):
        fd = {}
        for i in range(self.bsz):
            idx = self.idx[self.niter % self.ndata]
            self.niter = self.niter + 1
            if self.niter % self.ndata == 0 and self.isrand:
                self.idx = np.int32(self.rand.permutation(self.ndata))
            fd[self.names[i]] = self.files[idx]
        return fd
        
    def __init__(self,lfile,bsz,csz,niter=0,isval=False):
        """
        Call with
           lfile = Path of file with list of image file names
           bsz = Batch size you want this to generate
           csz = Output images will be size csz x csz
           niter = Resume at niterations
           isval = Running on train or val (random crops and shuffling for train)
        """

        self.bsz = bsz
        self.csz = csz
        self.isrand = not isval

        # Setup fetch graph
        self.graph()

        # Load file list
        self.files = [l.strip() for l in open(lfile).readlines()]
        self.ndata = len(self.files)
        self.niter = niter*bsz
        
        # Setup shuffling
        if self.isrand:
            self.rand = np.random.RandomState(0)
            idx = self.rand.permutation(self.ndata)
            for i in range(niter // self.ndata):
                idx = self.rand.permutation(self.ndata)
            self.idx = np.int32(idx)
        else:
            self.idx = np.int32(np.arange(self.ndata))
                
    # Sets up a common batch variable for train and val and ops
    # to swap in pre-fetched image data.
    def tvSwap(self,vset):
        imbatch = tf.Variable(tf.zeros(self.imbatch.shape,dtype=tf.float32),trainable=False)
        mzbatch = tf.Variable(tf.zeros(self.mzbatch.shape,dtype=tf.float32),trainable=False)
        tSwap = [tf.assign(imbatch,tf.identity(self.imbatch)).op, tf.assign(mzbatch,tf.identity(self.mzbatch)).op]
        vSwap = [tf.assign(imbatch,tf.identity(vset.imbatch)).op, tf.assign(mzbatch,tf.identity(vset.mzbatch)).op]
        
        return imbatch,mzbatch,tSwap,vSwap