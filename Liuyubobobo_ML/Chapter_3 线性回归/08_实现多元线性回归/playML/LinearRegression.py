# 实现线性回归

import numpy as np
from .metric import r2_score

class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None  # [θ_1,θ_2,...,θ_n]
        self.intercept_ = None  # 截距θ_0
        self._theta = None  # θ,列向量

    # 利用正规方程解来求解θ
    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        
        # X_b，X_train的基础上拼接一列全为1（θ_0）的矩阵
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        # 对应正规方程解
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0] # θ_0
        self.coef_ = self._theta[1:]    # [θ_1,θ_2,...,θ_n]

        return self
    
    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)
    
    def __repr__(self):
        return "LinearRegression()"

