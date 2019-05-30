''' 自定义ID3决策树 '''

import numpy as np
import collections as col
from functools import partial

def entropy(classified_probabilities):
    return np.sum(-p * np.log(p) for p in classified_probabilities if p)

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count for count in col.Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_entropy(subsets):
    total_count = sum(len(subset) for subset in subsets)
    return sum( data_entropy(subset) * len(subset) / total_count for subset in subsets )

def partition_by(inputs, attribute):
    groups = col.defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)
    return groups

def partition_entropy_by(inputs, attribute):
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

def build_tree_id3(inputs, split_candidates=None):
    """以ID3算法构建决策树。最终决策树形如：
        ('level',
            {
                'Junior': ('phd', {'no': True, 'yes': False}),
                'Mid': True,
                'Senior': ('tweets', {'no': False, 'yes': True})
            }
        )
    """
    # 最初构建决策树时，要把所有特征都纳入条件熵计算
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()    # 本样例中，生成：['level', 'lang', 'tweets', 'phd']

    # 统计出所有样例中分类为True和False的数目
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues
    if num_trues == 0: return False      # 样例分类全为False，则直接构造仅有一个False叶子节点的树并返回
    if num_falses == 0: return True      # 样例分类全为True，则直接构造仅有一个True叶子节点的树并返回

    # 如果已经没有下级需要计算条件熵的特征，则完成树的构建，并且叶子节点取决于True和False的数量对比
    if not split_candidates:
        return num_trues >= num_falses

    # 如果还有需要计算条件熵的特征，则计算每个特征的条件熵，取最小的那个作为下级节点
    # 借助偏函数partial来调用partition_entropy_by函数。
    # inputs作为第一个参数，split_candidates中的元素作为第二个参数传入
    best_attribute = min(split_candidates, key=partial(partition_entropy_by, inputs))

    # 将原始样本集合，根据选中的节点的特征取值分成若干个子分区(partition)，从而为各子分区形成工作数据集
    partitions = partition_by(inputs, best_attribute)

    # 后续计算不再计入父节点特征
    new_candidates = [a for a in split_candidates if a != best_attribute]
    
    # 递归调用，生成本节点的下级树结构
    subtrees = { attribute_value : build_tree_id3(subset, new_candidates) 
        for attribute_value, subset in partitions.items() }
    
    # 为每个节点添加一个None分支，以统一处理未来待预测的数据中出现样本中不存在的特征或特征值的情况
    subtrees[None] = num_trues > num_falses

    return (best_attribute, subtrees)


def classify(tree, input):
    """根据决策树tree对输入数据input进行预测"""
    # 如果当前检索的节点已经是叶子节点，则直接返回该叶子节点
    if tree in [True, False]:
        return tree
    
    # 如果当前检索的节点不是叶子节点，则准备获取分支并选择下级节点进行比较
    # 获取当前节点的特征名称，及该节点下的分支树结构
    attribute, subtree_dict = tree              
    # 如果特征名称在input中不存在(待预测数据可能缺失某些特征)，则get方法返回None(NoneType)；
    # 而input[attribute]将会报错
    subtree_key = input.get(attribute)    
    # 如果item中的特征取值在训练样本数据中不存在，则视为None
    if subtree_key not in subtree_dict:
        subtree_key = None
    # 根据subtree_key来选择下级节点路径
    subtree = subtree_dict[subtree_key]
    # 递归调用，直到找到了最终叶子节点
    return classify(subtree, input)