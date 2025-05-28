from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize

iris = load_iris()
normalized = normalize(iris['data'], norm='l2')
print('Before normalization:')
print(iris['data'][:5])
print('After normalization via L2:')
print(normalized[:5])