#!/usr/bin/env python
# coding: utf-8

# In[457]:


import numpy as np


class Regression:
    def __init__(self, training_mode='GD', lr=0.01, max_iter=1000, eps=0.00001, yps=0.9):
        self.lr = lr
        self._n = max_iter
        self.eps = eps
        self.training_mode = training_mode
        self.theta = []
        self.yps = yps
        
    def fit(self, X, y):
        # defining a loss function
        self.define_cost()
        
        self.X = np.column_stack([X, np.ones((X.shape[0]))])
        self.y = y
        
        self.coef_ = np.random.normal(size=(self.X.shape[1]), scale=0.2)
        
        self._train()
        
    def _train(self):
        """
        Здесь:
        1) Выбираем алгоритм оптимизации
        2) Запускаем алгоритм (без цикла, цикл внутри алгоритма)
        3) Сохраняем в селфовые переменные коеф, интерсепт
        """
        _modes = {
            'GD' : self._gradient_descent,
            'RMSProp' : self._RMSProp,
            'Nesterov' : self._nesterov
        }
        
        self.coef_, self.intercept_ = _modes[self.training_mode](self.coef_)
        
    def predict(self, X):
        return self.predict_func(X)
    
    def _gradient_descent(self, coef):
        """
        """
        
        for i in range(self._n):
            old_coef = coef
            coef = old_coef - self.lr * self.cost_func(self.X, self.y, old_coef)

            if np.linalg.norm(old_coef - coef, ord=2) <= self.eps:
                break

        return (coef[:-1], coef[-1])
    
    def score(self, X, y):
        return self.score_func(X, y)
        
    def _nesterov(self, coef):
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
    def define_cost(self):
        
        def grad_mse(X, y, w):
            yproba = X @ w
            return 2/len(X)*(y - yproba) @ (-X)

        self.cost_func = grad_mse
        
    def predict_func(self, X):
        return X @ self.coef_ + self.intercept_
    
    def score_func(self, X, y):
        yproba = X @ self.coef_
        return np.average((y - yproba) ** 2)
    
    
class LogisticRegression(Regression):
    def define_cost(self):
        
        def grad_logloss(X, y, w):
            yproba = self.sigmoid(X @ w)
            return X.T @ (yproba - y)
        
        self.cost_func = grad_logloss
    
    def predict_func(self, X):
        yproba = self.sigmoid(X @ self.coef_)
        yhat = np.where(yproba >= 0.5, 1, 0)
        return yhat
    
    def score_func(self, X, y):
        yproba = self.sigmoid(X @ self.coef_)
        return -np.sum(y * np.log(yproba + 1e-30) + (1 - y) * np.log(1 - yproba + 1e-30))
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x.astype(float)))

