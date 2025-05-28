from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def filtering(method, features, target, k=5, cv=5):
    selector = SelectKBest(score_func=method, k=k)
    selected = selector.fit_transform(features, target)
    mask = selector.get_support(indices = True)
    k_folds(selected, target, cv, mask)

def k_folds(features, target, cv, mask=[]):
    model = RandomForestClassifier()
    scores = cross_val_score(model, features, target, cv=cv)
    if len(mask) > 0:
        print('Selected features:', data.feature_names[mask])
    print('Accuracy mean:', round(scores.mean(), 2), '\n')

if __name__ == '__main__':
    data = load_breast_cancer()
    features, target = data.data, data.target

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    print('Features:', len(data.feature_names))    
    print('Without filter:')
    k_folds(scaled, target, 5)

    k = 18
    print(f'With ANOVA, CHI2, MUTUAL INFORMATION (Best {k})')
    filtering(f_classif, features, target, k)
    filtering(chi2, features, target, k)
    filtering(mutual_info_classif, features, target, k)