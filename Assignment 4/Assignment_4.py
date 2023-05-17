import numpy as np


class autoencoder:
    
    def __init__(self, d, n1, alpha=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        d: int, dimensionality of the input vectors
        n1: int, dimensionality of the hidden layer
        alpha: float, learning rate (default 0.001)
        beta1: float, decay rate for first moment estimate (default 0.9)
        beta2: float, decay rate for second moment estimate (default 0.999)
        epsilon: float, term added to denominator to avoid division by zero (default 1e-8)
        """
        self.d = d
        self.n1 = n1
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m  = None
        self.X = None
        self.Ybatch = None
        self.a1 = None
        self.a2= None
        self.o1 = None
        self.o2 = None

        
        # Glorot initialization
        limit = np.sqrt(6 / (d + n1))
        self.W1 = np.random.uniform(-limit, limit, size=(d, n1))
        self.b1 = np.zeros((1,n1))
        self.W2 = np.random.uniform(-limit, limit, size=(n1, d))
        self.b2 = np.zeros((1,d))
        
        # Adam algorithm parameters initialization
        self.t = 0
        self.nuW1 = np.zeros_like(self.W1)
        self.hW1 = np.zeros_like(self.W1)
        self.nub1 = np.zeros_like(self.b1)
        self.hb1 = np.zeros_like(self.b1)
        self.nuW2 = np.zeros_like(self.W2)
        self.hW2 = np.zeros_like(self.W2)
        self.nub2 = np.zeros_like(self.b2)
        self.hb2 = np.zeros_like(self.b2)

      
    ## Activation function for the hidden layer (ReLU)
    def h1(self, a):
        return np.maximum(0, a)


    ## Activation function for the output layer (Sigmoid)
    def h2(self, a):
        return 1 / (1 + np.exp(-a))
    
    
    ## Derivative of the activation h1
    def dh1(self, a):
        return (a > 0).astype(int)


    ##Derivative of the activation h2
    def dh2(self, a):
        return self.h2(a) * (1 - self.h2(a))
    
    
    def loss(self, Xbatch):
        Ybatch = self.forward(Xbatch)
        self.m = Xbatch.shape[0]
        loss = (-1 / (self.d * self.m)) * np.sum(Xbatch * np.log(Ybatch) + (1 - Xbatch) * np.log(1 - Ybatch))
        return loss
    
    def dloss(self, Ybatch):
        d = self.d
        m = Ybatch.shape[1]
        self.Ybatch = Ybatch
        return -(self.X / self.Ybatch) + ((1 - self.X) / (1 - self.Ybatch))


    def forward(self, Xbatch):
        self.X = Xbatch
        self.m = Xbatch.shape[1]

        # Hidden layer activation
        self.a1 = np.dot(Xbatch, self.W1) + self.b1
        self.o1 = self.h1(self.a1)

        # Output layer activation
        self.a2 = np.dot(self.a1,self.W2) + self.b2
        self.o2 = self.h2(self.a2)

        return self.o2


    def backward(self):
        
        # Compute derivatives of loss with respect to Ybatch
        dY = self.dloss(self.o2)

        # Compute derivatives of loss with respect to activations
        da2 = dY * self.dh2(self.a2)

        # Compute derivatives of loss with respect to weights and biases of the output layer
        self.dW2 = (self.a1.T @ da2) / self.m
        self.db2 = np.sum(da2, axis=0) / self.m

        # Compute derivatives of loss with respect to activations of the hidden layer
        da1 = da2 @ self.W2.T * self.dh1(self.a1)

        # Compute derivatives of loss with respect to weights and biases of the hidden layer
        self.dW1 = (self.X.T @ da1)/self.m
        self.db1 = np.sum(da1, axis=0)/self.m
        
        

    def adam_step(self, alpha=0.001, rho1=0.9, rho2=0.999, delta=1e-8):
        # Increment timestep
        self.t += 1

        # Update nuW1, nuW2, nub1, nub2
        self.nuW1 = rho1 * self.nuW1 + (1 - rho1) * self.dW1
        self.nuW2 = rho1 * self.nuW2 + (1 - rho1) * self.dW2
        self.nub1 = rho1 * self.nub1 + (1 - rho1) * self.db1
        self.nub2 = rho1 * self.nub2 + (1 - rho1) * self.db2

        # Update hW1, hW2, hb1, hb2
        self.hW1 = rho2 * self.hW1 + (1 - rho2) * self.dW1**2
        self.hW2 = rho2 * self.hW2 + (1 - rho2) * self.dW2**2
        self.hb1 = rho2 * self.hb1 + (1 - rho2) * self.db1**2
        self.hb2 = rho2 * self.hb2 + (1 - rho2) * self.db2**2

        # Compute nuPhat, hPhat
        nuW1_hat = self.nuW1 / (1 - rho1**self.t)
        nuW2_hat = self.nuW2 / (1 - rho1**self.t)
        nub1_hat = self.nub1 / (1 - rho1**self.t)
        nub2_hat = self.nub2 / (1 - rho1**self.t)

        hW1_hat = self.hW1 / (1 - rho2**self.t)
        hW2_hat = self.hW2 / (1 - rho2**self.t)
        hb1_hat = self.hb1 / (1 - rho2**self.t)
        hb2_hat = self.hb2 / (1 - rho2**self.t)

        # Update weights and biases
        self.W1 -= alpha * nuW1_hat / (np.sqrt(hW1_hat) + delta)
        self.W2 -= alpha * nuW2_hat / (np.sqrt(hW2_hat) + delta)
        self.b1 -= alpha * nub1_hat / (np.sqrt(hb1_hat) + delta)
        self.b2 -= alpha * nub2_hat / (np.sqrt(hb2_hat) + delta)





from tensorflow import keras

(Xtrain, _), (Xtest, _) = keras.datasets.fashion_mnist.load_data()

# normalize the pixel values to be between 0 and 1
Xtrain = Xtrain.astype('float32') / 255.
Xtest = Xtest.astype('float32') / 255.

# reshape the data to be of shape (n_samples, n_features)
Xtrain = Xtrain.reshape((len(Xtrain), np.prod(Xtrain.shape[1:])))
Xtest = Xtest.reshape((len(Xtest), np.prod(Xtest.shape[1:])))

d = Xtest.shape[1]
n1 = 100
numepochs = 20
numsplits = 10



ae = autoencoder(d,n1)
#ae.loss(Xtrain)
#Ybatch = ae.forward(Xtrain)
#ae.dloss(Ybatch)
#ae.backward()
#ae.adam_step()



for e in range(numepochs):
    print("Training loss [%d/%d]: %.2f" % (e+1, numepochs, ae.loss(Xtrain)))
    minibatches = np.split(np.random.permutation(Xtrain), numsplits)
    for Xbatch in minibatches: 
        #Xbatch = Xtrain[:, indices] 
        ae.forward(Xbatch) 
        ae.backward() 
        ae.adam_step()

     

    


