#!/usr/bin/env python
# coding: utf-8

# In[457]:


import numpy as np


class Regression:
    """
    A class that makes basic common operations for linear models such as: fitting the model; choosing a loss function according to desired model;
    training model according to its loss function; optimizing model by a choosen algorythm (stochastic gradient descent, RMSProp, Nesterov ADA).
    Also able to predict y-values and score accuracy of a model.
    Parameters
    -------------------------
    lr: float, default = 0.01 
        Learning_rate, defines a step
        
    training_mode: str, default = 'GD'
        Choose a training algorythm. Now 3 kinds are realized: stochastic gradient descent, RMSPRop, Nesterov ADAM (NADAM).

    max_iter: int, default = 1000
        Parameter that defines number of maximum iterations of optimization.

    eps: float, default = 0.00001
        Minimal error value. When a difference between weights are less than 'eps', algorythm stops it's work. 

    yps: float, default = 0.9
        Parameter of 'remembering' of impulse. Only applied to impulse-kind of optimization algorythms.
    """
    def __init__(self, training_mode='GD', lr=0.01, max_iter=1000, eps=0.00001, yps=0.9):
        self.lr = lr
        self._n = max_iter
        self.eps = eps
        self.training_mode = training_mode
        self.theta = []
        self.yps = yps
        
    def fit(self, X, y):
        """
        Ez Peeze:
        Defines cost function. Fits the model. Saves coefficients. 
        
        Parameters
        -------------------------
        X: matrix, df slice or array-like
            Data-values. Works both for 1-D array or n-D array (inf > n > inf)
        y: array-like values
            Target value to predict
        """
        # defining a loss function
        self._define_cost()
        
        self.X = np.column_stack([X, np.ones((X.shape[0]))])
        self.y = y
        
        self.coef_ = np.random.normal(size=(self.X.shape[1]), scale=0.2)
        
        self._train()
        
    def _train(self):
        """
        1) Choose optimization algorythm
        2) Enacts choosen algorythm
        """
        _modes = {
            'GD' : self._gradient_descent,
            'RMSProp' : self._RMSProp,
            'Nesterov' : self._nesterov
        }
        
        self.coef_, self.intercept_ = _modes[self.training_mode](self.coef_)
        
    def predict(self, X):
        """
        Predicts the target value based on learned weights.

        Parameters
        -------------------------
        X: matrix, df slice or array-like
            Data-values. Works both for 1-D array or n-D array (inf > n > inf)

        """
        return self._predict_func(X)

    def score(self, X, y):
        """
        Scores the efficiency (accuracy) of the model based on adequate loss function
        Parameters
        -------------------------
        X: matrix, df slice or array-like
            Data-values. Works both for 1-D array or n-D array (inf > n > inf)   
        y: array-like values
            Target value to predict
        """
        return self._score_func(X, y)
    
    def _gradient_descent(self, coef):
        """
        Runs a stochastic gradient descent algorythm.
        
        Parameters
        -------------------------
        coef: 1-D array the same size as X
            Predicted weights
        """
        for i in range(self._n):
            old_coef = coef
            coef = old_coef - self.lr * self.cost_func(self.X, self.y, old_coef)

            if np.linalg.norm(old_coef - coef, ord=2) <= self.eps:
                break

        return (coef[:-1], coef[-1])
        
    def _nesterov(self, coef):
        """
        Runs a Nesterov ADAM (NADAM) algorythm.
        
        Parameters
        -------------------------
        coef: 1-D array the same size as X
            Predicted weights
        """
        Vt = np.zeros(self.X.shape[1])

        for i in range(self._n):
            old_coef = coef
            
            impulse = old_coef - (self.yps * Vt)
            Vt = (self.yps * Vt) + (self.lr * self.cost_func(self.X, self.y, old_coef))
            coef = old_coef - Vt

            if np.linalg.norm(old_coef - coef, ord=2) <= self.eps:
                break
        
        return (coef[:-1], coef[-1])
        
    def _RMSProp(self, coef):        
        """
        Runs an RMSProp algorythm.
        
        Parameters
        -------------------------
        coef: 1-D array the same size as X
            Predicted weights
        """
        Egt = 0

        for i in range(self._n):
            old_coef = coef
            
            g = self.cost_func(self.X, self.y, old_coef)
            Egt = (self.yps * Egt) + (1 - self.yps) * g**2
            coef = old_coef - (self.lr / np.sqrt(Egt + 1e-30)) * g
            
            if np.linalg.norm(old_coef - coef, ord=2) <= self.eps:
                break
                
        return (coef[:-1], coef[-1])

class LinearRegression(Regression):
    """
    Basic Linear Regression.
    """
    def _define_cost(self):
        
        def grad_mse(X, y, w):
            """ gradient of loss function for linear regression """
            yproba = X @ w
            return 2/len(X)*(y - yproba) @ (-X)

        self.cost_func = grad_mse
        
    def _predict_func(self, X):
        """ 
        Prediction function
        y = w0 + w1 * X
        """
        return X @ self.coef_ + self.intercept_
    
    def _score_func(self, X, y):
        yproba = X @ self.coef_
        return np.average((y - yproba) ** 2)
    
    
class LogisticRegression(Regression):    
    """
    Basic Logistic Regression.
    """
    def _define_cost(self):
        
        def grad_logloss(X, y, w):
            """ gradient of loss function for logistic regression """
            yproba = self.sigmoid(X @ w)
            return X.T @ (yproba - y)
        
        self.cost_func = grad_logloss
    
    def _predict_func(self, X):
        yproba = self.sigmoid(X @ self.coef_)
        yhat = np.where(yproba >= 0.5, 1, 0)
        return yhat
    
    def _score_func(self, X, y):
        yproba = self.sigmoid(X @ self.coef_)
        return -np.sum(y * np.log(yproba + 1e-30) + (1 - y) * np.log(1 - yproba + 1e-30))
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x.astype(float)))
        
