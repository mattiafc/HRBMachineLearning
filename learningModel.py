import numpy   as np
import pandas  as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt

class logistic_regression:
    
    train_size = 0.60
    valid_size = 0.0
    test_size = 0.40
    
    
    def __init__(self, X_input, y_input, dataset = 'Standard'):
        
        #if dataset == ' Standard'
        
        self.nFeatures, self.nSamples = X_input.shape
        
        self.X_mean = np.mean(X_input, axis = 1, keepdims = True)
        self.X_std  = np.std(X_input, axis = 1, keepdims = True, ddof = 1)+1e-10
        
        X = (X_input - self.X_mean)/self.X_std
        
        y_input[y_input>-1] = 1
        y_input[y_input<-1] = 0
        
        y = y_input
        
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = self.train_valid_test(X.T, y.T)
        
        print('=====================================================')
        print('Total number of samples is: ' + str(X.shape[1]))
        print('Training   set is ' + str(self.train_size) + ' which corresponds to ' +str(self.X_train.shape[1]) + ' samples')
        print('Validation set is ' + str(self.valid_size) + ' which corresponds to ' +str(self.X_valid.shape[1]) + ' samples')
        print('Test       set is ' + str(self.test_size)  + ' which corresponds to ' +str(self.X_test.shape[1])  + ' samples')
        print('=====================================================\n')
        
        params = self.initialize_parameters()
        
        for i in range(10000):
        
            AL, cache   = self.forward_propagation(params, self.X_train)
            grads, cost = self.backward_propagation(params, AL, self.X_train, self.y_train)
            params      = self.update_params(params, grads, 0.01)
            
        
        pred_train = self.predict(params, self.X_train)
        accuracy_train = np.sum(pred_train == self.y_train)/self.y_train.shape[1]
        
        pred_test = self.predict(params, self.X_test)
        accuracy_test = np.sum(pred_test == self.y_test)/self.y_test.shape[1]
        
        print('=====================================================')
        print('Train set accuracy is: ' + str(accuracy_train))
        print('Test  set accuracy is: ' + str(accuracy_test))
        print('Model parameters: ' + str(params["W1"]) + str(params["b1"]))
        print('=====================================================\n')
        
        

    
    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x)) 

    def train_valid_test(self, X, y):
        
        if abs(self.train_size + self.test_size + self.valid_size - 1.0) > 1e-6:
            raise Exception("Summation of dataset splits should be 1")
        
        X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=self.train_size, random_state=42)
        
        #X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=self.test_size/(self.valid_size + self.test_size), random_state=42)
        
        X_valid = np.zeros((2,2))
        y_valid = np.zeros((2,2))
        
        return X_train.T, X_valid.T, X_test.T, y_train.T, y_valid.T, y_test.T
    
    def initialize_parameters(self):
            
        W1 = np.ones((1,self.nFeatures))*0.01;
        b1 = 0.0 ;
        
        params = {"W1": W1,
                  "b1": b1}
        
        return params
    
    def forward_propagation(self, params, X):
        
        cache = 0

        W1 = params["W1"]
        b1 = params["b1"]
        
        Z1 = np.dot(W1,X) + b1
        
        A1 = self.sigmoid(Z1)
        AL = self.sigmoid(Z1)
        
        #cache.append = (A1,W1,b1)
        #cache.append = (W1)
        
        return AL, cache
    
    def backward_propagation(self, params, A, X, Y):
        
        cost = -np.sum(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1-A)))/X.shape[1]
        
        dW1 = (A-Y).dot(X.T)/X.shape[1]
        
        db1 = np.sum(A-Y)/X.shape[1]
        
        grads = {"dW1": dW1,
                 "db1": db1}
    
        return grads, cost
    
    def update_params(self, params, grads, learning_rate):
    
        W1 = params["W1"] - learning_rate*grads["dW1"]
        b1 = params["b1"] - learning_rate*grads["db1"]
        params = {"W1": W1,
                  "b1": b1}
        
        return params
    
    def predict(self, params, X):

        W1 = params["W1"]
        b1 = params["b1"]
        
        Z1 = np.dot(W1,X) + b1
        pred = self.sigmoid(Z1)
        pred[pred>=0.5] = 1
        pred[pred<0.5]  = 0
        
        return(pred)
        
    
        
def train_test_MF(angles_train, angles_test):
    
    data = []

    variables = ['CfMean','TKE','U','gradP','rmsCp','peakminCp','peakMaxCp','theta']
    labels = 'meanCp'
    
    for ang in angles_train:
        data.append(pd.read_csv(str('Features/Coarsest' + str(ang))))
    
    dataFrame = pd.concat(data, axis=0)   
    
    #test  = []
    
    #for ang in test_set:
        #test_set.append(pd.read_csv(str('Features/Coarsest' + str(ang))))
    

    X = dataFrame[variables].values.T
    y = np.asmatrix(dataFrame[labels].values)
    
    return X,y

X, y = train_test_MF([0,10,20,30,40,50,60,70,80,90], [0])

LR = logistic_regression(X, y)

print("Pull again")