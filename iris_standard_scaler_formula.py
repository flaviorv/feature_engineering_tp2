from sklearn.datasets import load_iris
import math
import numpy as np

iris = load_iris()

def mean(nums):
    sum = 0
    for n in nums:
        sum += n
    return sum/len(nums)

#Population standard deviation, not sample
def standard_deviation(nums):
    _mean = mean(nums)
    sum = 0
    for n in nums:
        sum += (n - _mean) ** 2
    return math.sqrt((sum / (len(nums))))

def standard_scaler(feature):
    std = standard_deviation(feature)
    _mean = mean(feature)
    scaled = []
    for value in feature:
        new_value = (value - _mean) / std
        scaled.append(new_value)
    return scaled

def iris_std_scaling():
    for index, feature in enumerate(iris['feature_names']):
        scaled = standard_scaler(iris['data'][:, index])
        print(feature)
        print('standard deviation', round(standard_deviation(scaled), 2))
        print('mean', round(mean(scaled), 2))
        print(np.array2string(np.array(scaled), suppress_small=True, precision=8), '\n')

if __name__ == "__main__":
    scaled = iris_std_scaling()  
