''' 批量梯度下降算法 '''
import numpy as np

def batch_gradient_descent(target_fn, gradient_fn, init_W, X, Y, learning_rate=0.001, tolerance=1e-7):
    """支持多变量的批量梯度下降法"""
    # 假设函数为：y = wn * xn + w(n-1) * x(n-1) +... + w2 * x2 + w1 * x1 + w0 * x0 其中，x0为1
    # X中：第一列为xn,第二列为x(n-1)，最后一列为x0(全为1)，依次类推
    # W向量顺序是：wn,w(n-1),...w1,w0，要确保与X中各列顺序一致
    W = init_W
    target_value = target_fn(W, X, Y) 
    iter_count = 0
    while iter_count < 50000:                      # 如果50000次循环仍未收敛，则认为无法收敛
        gradient = gradient_fn(W, X, Y)
        next_W = W - gradient * learning_rate 
        next_target_value = target_fn(next_W, X, Y)
        if abs(target_value - next_target_value) < tolerance:
            print("循环", iter_count, "次后收敛")
            return W
        else:                                             
            W, target_value = next_W, next_target_value
            iter_count += 1
    
    print("50000次循环后，计算仍未收敛")
    return W