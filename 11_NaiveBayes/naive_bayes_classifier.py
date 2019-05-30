''' 自定义的贝叶斯文本分类器，支持二分类 '''

import numpy as np
from collections import defaultdict
import re

class BinaryNaiveBayesClassifier:
    """使用朴素贝叶斯的二分类算法"""
    def __init__(self):
        self.alpha = 1.0
        self.LAMBDA = 0
        self.p_positive = 0.0                       # Positive文本所占比例
        self.p_negtive = 0.0                        # Negtive文本所占比例
        self.p_log_elem_in_positive = []            # 各单词在Positive元素总数中的概率
        self.p_log_elem_in_negtive = []             # 各单词在Negtive元素总数中的概率

    def train(self, training_set, classify_set):
        """trainning_set：m_samples X n_features的矩阵，矩阵中每个元素是0或1
           classify_set：m_samples X 1的矩阵或数组对应training_set中每行数据的分类结果，0或1表示
        """
        num_total = len(training_set)
        num_positive = np.sum(classify_set)         # 因为每个元素都是0或1，因此求和就是positive的个数
        num_negtive = num_total - num_positive
        self.p_positive = num_positive / num_total
        self.p_negtive = 1 - self.p_positive
        self.LAMBDA = training_set.shape[1]         # 设置为单词表的容量

        positive_set = [x for x, y in zip(training_set, classify_set) if y==1]
        negtive_set = [x for x, y in zip(training_set, classify_set) if y==0]
        # 对每个单词，分别计算其占positive/negtive中元素总数的比例
        # 应用拉普拉斯平滑，并且取对数
        self.p_log_elem_in_positive = np.log((np.sum(positive_set, axis=0) + self.alpha) / (np.sum(positive_set) + self.alpha * self.LAMBDA))
        self.p_log_elem_in_negtive = np.log((np.sum(negtive_set, axis=0) + self.alpha) / (np.sum(negtive_set) + self.alpha * self.LAMBDA))

    def classify(self, predicting_vector):
        """predicting_vector：n_features的向量，矩阵中每个元素是0或1"""
        # 贝叶斯公式分子部分。
        # 与predicting_vector作对应元素乘积，相当于不计入predicting_vector中为0的元素
        log_numerator = np.sum(self.p_log_elem_in_positive * predicting_vector)
        numerator = self.p_positive * np.exp(log_numerator)
        # 贝叶斯公式分母部分
        log_denominator_negtive = np.sum(self.p_log_elem_in_negtive * predicting_vector)
        denominator = numerator + np.exp(log_denominator_negtive) * self.p_negtive
        return numerator / denominator
        
