from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score

def wrapping(n_features, features, target, n_estimators=100, cv=5):
    sfs = SequentialFeatureSelector(
        estimator=RandomForestClassifier(n_estimators=n_estimators),
        n_features_to_select=n_features,
        direction='forward',
        scoring='accuracy',
        cv=cv
    )

    sfs.fit(features, target)
    indexes = sfs.get_support(indices=True)
    selected = sfs.transform(features)
    model = RandomForestClassifier(n_estimators=n_estimators)
    scores = cross_val_score(model, selected, target, cv=5)
    print("Mean accuracy:", scores.mean())
    return indexes

if __name__ == '__main__':
    data = load_breast_cancer()
    features = data.data
    target = data.target
    indexes = wrapping(18, features, target)
    print('Selected features:\n', data.feature_names[indexes])