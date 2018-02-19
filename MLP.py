import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import time
class MLP(object):
    
    def __init__(self, eta = 0.01, l1=0, l2=0, M= None):
        np.random.seed(1)
        self.eta = eta
        self.l1 = l1
        self.l2 = l2
        
        self.M = list(map(lambda x: x+1, M))
        self.D = len(self.M)-1
        self.w = self._initialize_weights()
        self.dw = self._initialize_weights()
        self.delta = np.zeros(self.D, dtype=object)
        self.o = np.zeros(self.D+1, dtype=object)
        for i in range(self.D):
            self.delta[i] = np.zeros(self.M[i+1], dtype=float).reshape(self.M[i+1],1)
            self.o[i] = np.zeros(self.M[i], dtype = float).reshape(1,self.M[i])
        self.o[self.D] = np.zeros(self.M[self.D], dtype = float).reshape(1,self.M[self.D])
        for i in range(self.D+1):
            self.o[i][0][0] = 1
        
    def _initialize_weights(self):
        _w = []
        for i in range(self.D):
            _wi = np.random.uniform(-0.5,0.5, size = self.M[i]*self.M[i+1])
            _wi = _wi.reshape(self.M[i+1], self.M[i])
            _w += [_wi]
        #waste code
        for i in range(self.D):
            _w[i][0] *= 0
        return np.array(_w)
    
    def _sig(self, z): return expit(z)
    def _sign(self, z): return np.sign(z)
    
    def _predict(self,x):
        np.copyto(self.o[0][0][1:], x)
        self.o[0][0][0] = 1
        for i in range(self.D):
            self.o[i+1] = (expit(self.w[i].dot(self.o[i].T))).T
            self.o[i+1][0][0] = 1
    
    def predict(self, x):
        self._predict(x)
        return (self.o[self.D].reshape(self.M[self.D]))[1:]
    
    def gradient(self, x, y):
        self._predict(x)
        self.delta[self.D-1][0][0] = 0
        self.delta[self.D-1][1:] = ((self.o[self.D][0][1:]-y) * self.o[self.D][0][1:] * (1-self.o[self.D][0][1:])).reshape(self.M[self.D]-1,1)
        for l in range(self.D-1):
            i = self.D-2-l
            self.delta[i] = self.delta[i+1].T.dot(self.w[i+1]).T
            self.delta[i] *= (self.o[i+1] * (1.0-self.o[i+1])).T
        for i in range(self.D):
            np.dot(self.delta[i],self.o[i], self.dw[i])
    
    def train(self, x, y):
        self.gradient(x,y)
        for i in range(self.D):
            self.w[i] -= self.eta * (self.dw[i] +  self.l1 * np.sign(self.w[i]) + self.l2 * self.w[i])
    

##progn program
nn =  MLP(M=[2,10,2]);
gen = lambda : np.random.uniform(-0.5,0.5)
within = lambda t,x,y : x*x+y*y < t*t
sampleSize = 3000
MaxIt = 3000
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
start = time.time()
for p in range(MaxIt):
    
    for i in range(sampleSize):
        nn.train(sample[i], sampleLabel[i])
    ##print(time.time() - start)
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
print(time.time() - start)
    