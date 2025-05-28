from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
import numpy as np


def normalizing(data):
    normalized = normalize(data, norm='l1')
    return normalized

#The some of all feature values in each sample must be 1
def validating(data):
    result = []
    sum = 0
    for sample in data:
        for value in sample:
            sum += value
        result.append(sum)
        sum = 0
    return result

if __name__ == "__main__":
    iris = load_iris()

    normalized = normalizing(iris['data'])
    print('Before normalization:')
    print(iris['data'][:5])
    print('After normalization via L1:')
    print(normalized[:5])

    result = np.array(validating(normalized))
    print('Validating:')
    print(np.array2string(result, suppress_small=True, precision=2))    