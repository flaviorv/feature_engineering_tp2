from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

def filtering(method, features, target, k=5, cv=5):
    selector = SelectKBest(score_func=method, k=k)
    selected = selector.fit_transform(features, target)
    mask = selector.get_support(indices = True)
    k_folds(selected, target, cv, mask)

def k_folds(features, target, cv, mask=[]):
    model = RandomForestClassifier()
    scores = cross_validate(model, features, target, cv=cv, return_train_score=True, scoring='accuracy')
    if len(mask) > 0:
        print('Selected features:', data.feature_names[mask])
    train_mean = scores['train_score'].mean()
    test_mean = scores['test_score'].mean()
    print('Train mean', train_mean)
    print('Test mean', test_mean)
    print('Gap', train_mean - test_mean, '\n')

if __name__ == '__main__':
    data = load_breast_cancer()
    features, target = data.data, data.target
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    print('Features:', len(data.feature_names))    
    print('Without filter:')
    k_folds(scaled, target, 5)

    k = 18
    print(f'With ANOVA, CHI-SQUARE, MUTUAL INFORMATION (Best {k})')
    filtering(f_classif, features, target, k)
    filtering(chi2, features, target, k)
    filtering(mutual_info_classif, features, target, k)
    