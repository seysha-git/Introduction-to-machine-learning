import numpy as np

class LinearRegression():
    def __init__(self, lr=0.001, epochs=1000):
        self.weight = None
        self.bias = None
        
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

    def update_parameters(self, grad_w, grad_b):
        self.weight = self.weight - self.lr*grad_w 
        self.bias = self.bias - self.lr*grad_b 
    def compute_gradients(self, x, y, y_pred):
        grad_b = np.sum(y_pred.to_numpy() - y.to_numpy())/(len(y.to_numpy()))
        grad_w = np.sum(x*(y_pred.to_numpy() - y.to_numpy()))/(len(y.to_numpy()))
        return grad_w, grad_b
    def compute_loss(self, y, y_pred):
        loss = (np.sum(y.to_numpy()-y_pred.to_numpy()))**2/(2*len(y.to_numpy()))
        return loss
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # ====================================
        self.bias = 0
        self.weight = 0 # X.shape[1] # datapunkter, feature 
        #print(self.weight)
        # Gradient descent 
        for _ in range(self.epochs):
            lin_model = self.weight* X + self.bias
            y_pred = lin_model
            #print(y_pred)
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

        # ====================================
        #raise NotImplementedError("LinearRegression.fit is not implemented yet.")
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # ====================================
        # YOUR CODE GOES HERE
        y_pred = self.weight*X + self.bias
        
        return y_pred
        

        # ====================================
        raise NotImplementedError("LinearRegression.predict is not implemented yet.")
    
class LogisticRegression():
    def __init__(self, lr=0.001, epochs=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.epochs = epochs
        
        self.loss_history = []
    
    def fit(self, X, y):
        # ====================================
        self.bias = 0
        self.weights = np.zeros(X.shape[1])
        #print(X.shape)
        for _ in range(self.epochs):
            lin_model = np.matmul(self.weights, X.transpose()) + self.bias
            y_pred = self.sigmoid(lin_model)
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            #print(self.weights)
            #print("Bias", self.bias)
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
        # ====================================
        #raise NotImplementedError("LogisticRegression.fit is not implemented yet.")
    def compute_loss(self, y, y_pred):
        loss = -y*np.log(y_pred) - (1-y)*np.log(1-y_pred)
        return np.average(loss)
    def update_parameters(self, grad_w, grad_b):
        #print(self.weights)
        self.weights = self.weights - self.lr*grad_w
        self.bias = self.bias - self.lr*grad_b 
    def compute_gradients(self, x, y, y_pred):
        grad_w = np.matmul(x.T, (y_pred-y))
        grad_b = (y_pred-y)
        return grad_w, grad_b 
    def predict_proba(self, X):
        # ====================================
        pass
        # ====================================
        raise NotImplementedError("LogisticRegression.predict_proba is not implemented yet.")
        
    def predict(self, X):
        # ====================================
        #print(X.transpose().shape)
        lin_model_ = np.matmul(self.weights, X.transpose()).to_numpy() + self.bias.to_numpy()
        y_pred = self.sigmoid(lin_model_)
        #print(y_pred)
        return [1 if _y > 0.5 else 0 for _y in y_pred]
        # ====================================
        raise NotImplementedError("LogisticRegression.predict is not implemented yet.")
    
    def sigmoid(self, z):
        # ====================================
        return(1/(1+np.exp(-z))) 
        # ====================================
        