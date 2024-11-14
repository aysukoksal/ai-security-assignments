import numpy as np


def unit_step_func(x):
    #if x>0 return 1 otherwise 0
    return np.where(x>0, 1, 0)


class Percep:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr=learning_rate
        self.n_iters= n_iters
        self.activation_func= unit_step_func
        self.weights= None
        self.bias= None
        self.ber_score=None
        self.accuracy_score=None
    
    def fit(self, X_train,y_train):
        n_samples, n_features= X_train.shape
        self.weights= np.random.rand(n_features)
        self.bias=0
        y_ = np.where(y_train>0, 1, 0)
        #learn weights

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X_train):
                linear_output=np.dot(x_i, self.weights) + self.bias
               
                y_predicted= self.activation_func(linear_output)

                #perceptr update rule
                update= self.lr * (y_[idx]-y_predicted)
                self.weights+= update*x_i
                self.bias+=update
        
    def BER(self, y_test, y_pred):

        is_negative_one= y_test==-1 #this creates a bool array where true when y_true is -1
        is_positive_one = y_test==1 #this creates a bool array where true when y_true is 1


        #y_true==y_pred returns a list of boolean values where true indicates elements are equal and false elements are not equal
        t_minus= np.sum((y_test==y_pred)& is_negative_one)  #when we sum an array true values are 1 and false 0. so this returns number of times where y_true==y_pred
        f_minus =np.sum((y_test!=y_pred)& is_negative_one) #~ is bitwise NOT and flip all the boolean values
        t_pos= np.sum((y_test==y_pred)& is_positive_one)
        f_pos= np.sum((y_test!=y_pred)& is_positive_one)
        ber_score= 0.5*(f_minus/(f_minus+t_minus) + f_pos/(f_pos+t_pos))
        self.ber_score=ber_score

    def accuracy(self, y_test, y_pred):
        accuracy_score = np.sum(y_test == y_pred) / len(y_test)
        self.accuracy_score=accuracy_score
    
    def predict(self, X):
        linear_output=np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def create_percep(self, X_train, y_train, X_test,y_test):
        self.fit(X_train, y_train)
        predictions = self.predict(X_test)
        self.accuracy(y_test, predictions)
        self.BER(y_test, predictions)
        return self