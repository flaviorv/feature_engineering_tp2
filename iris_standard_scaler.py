from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

#Showing the original data
iris = load_iris()
print(iris['feature_names'])
print(iris['data'], '\n')
print(iris['target_names'])
print(iris['target'])

#Applying the normalization
scaler = StandardScaler()
scaled= scaler.fit_transform(iris['data'])
print('scaled:\n', scaled)

for index, feature in enumerate(iris['feature_names']):
    print('\n', feature)
    #Standard Mean always must be 0
    print('mean', round(scaled[:, index].mean()))
    #Standard Deviation always must be 1
    print('standard deviation', scaled[:, 0].std())