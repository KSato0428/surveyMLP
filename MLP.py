import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import time
class MLP(object):
    
    def __init__(self, eta = 0.01, l1=0, l2=0, M= []):
        np.random.seed(1)
        self.eta = eta
        self.l1 = l1
        self.l2 = l2
        
        self.M = M
        self.D = len(self.M)-1
        self.w, self.theta = self._initialize_weights()
        self.dw, self.dth = self._initialize_weights()
        _delta = []
        _o = []
        for i in range(self.D):
            _delta += [np.zeros(self.M[i+1], dtype = float )]
            _o += [np.zeros(self.M[i], dtype = float )]
        _o += [np.zeros(self.M[self.D], dtype = float)]
        self.delta = np.array(_delta)
        self.o = np.array(_o)
        
    def _initialize_weights(self):
        _w = []
        for i in range(self.D):
            _wi = np.random.uniform(-0.5,0.5, size = self.M[i]*self.M[i+1])
            _wi = _wi.reshape(self.M[i+1], self.M[i])
            _w += [_wi]
        _th = []
        for i in range(self.D):
            _th += [np.random.uniform(-0.5,0.5, size = self.M[i+1])]

        return np.array(_w), np.array(_th)
    
    def _sig(self, z): return expit(z)
    def _sign(self, z): return np.sign(z)
    
    def _predict(self,x):
        self.o[0] = 1.0 * x 
        for i in range(self.D):
            self.o[i+1] = expit(self.theta[i] + (self.w[i].dot(self.o[i].reshape(self.M[i],1))).reshape(self.M[i+1]))
            #for kn in range(self.M[i+1]):
            #    self.o[i+1][kn] = self._sig(self.theta[i][kn] + self.w[i][kn].T.dot(self.o[i]))
    
    def predict(self, x):
        self._predict(x)
        return self.o[self.D].copy()
    
    def gradient(self, x, y):
        self._predict(x)
        self.delta[self.D-1] = (self.o[self.D]-y) * self.o[self.D] * (1-self.o[self.D])
        for l in range(self.D-1):
            i = self.D-2-l
            self.delta[i] = self.delta[i+1].dot(self.w[i+1])
            self.delta[i] *= self.o[i+1] * (1.0-self.o[i+1])
        for i in range(self.D):
            self.dth[i] = self.delta[i]
            np.dot(self.delta[i].reshape(self.M[i+1],1),self.o[i].reshape(1,self.M[i]), self.dw[i])
            #for k in range(self.M[i+1]):
            #    self.dw[i][k] = self.delta[i][k] *self.o[i]
    
    def train(self, x, y):
        self.gradient(x,y)
        #self.theta -= self.eta * (self.dth + self.l1 * np.sign(self.theta) + self.l2 * self.theta)
        #self.w -= self.eta * (self.dw + self.l1 * np.sign(self.w) + self.l2 * self.w)
        for i in range(self.D):
            #self.theta[i] -= self.eta * (self.dth[i] + self.l1 * np.vectorize(self._sign)(self.theta[i]) +self.l2 * self.theta[i])
            #self.w[i] -= self.eta * (self.dw[i] + self.l1 * np.vectorize(self._sign)(self.w[i]) + self.l2 * self.w[i])
            self.theta[i] -= self.eta * (self.dth[i]  + self.l1 * np.sign(self.theta[i])+self.l2 * self.theta[i])
            self.w[i] -= self.eta * (self.dw[i] +  self.l1 * np.sign(self.w[i]) + self.l2 * self.w[i])
    

##progn program
nn =  MLP(M=[2,5,2]);
gen = lambda : np.random.uniform(-0.5,0.5)
within = lambda t,x,y : x*x+y*y < t*t
sampleSize = 1000
MaxIt = 1000
r = 30
pr = 10
sample = []
sampleLabel = []
for i in range(sampleSize):
    x = r * gen()
    y = r * gen()
    sample += [np.array([x,y])]
    if within(pr, x, y):
        sampleLabel += [np.array([0,1.0])]
    else :
        sampleLabel += [np.array([1.0,0])]
sample = np.array(sample)
sampleLabel = np.array(sampleLabel)

##training
for p in range(MaxIt):
    start = time.time()
    for i in range(sampleSize):
        nn.train(sample[i], sampleLabel[i])
    print(time.time() - start)
    if p%100==0:
        tcnt = 0
        for k in range(sampleSize):
            ny = nn.predict(sample[k])
            c1 = ny[0]*ny[0]+(1-ny[1])*(1-ny[1])
            c2 = (ny[0]-1)*(ny[0]-1)+ny[1]*ny[1]
            if sampleLabel[k][0]==0 and c1<c2 :
                tcnt+=1
            if sampleLabel[k][0]==1 and c2<c1 :
                tcnt+=1
        print(tcnt,"/",sampleSize)
    
    