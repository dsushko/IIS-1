import numpy as np
import pandas as pd

import sklearn.tree
from sklearn.tree import DecisionTreeClassifier

np.random.seed(0)

class OneHotEncoder:

    def __init__(self):
        pass

    def fit(self, X):
        res = pd.get_dummies(X)
        self.columns = res.columns
        return self

    def transform(self, X):
        return pd.DataFrame(pd.get_dummies(X), columns=self.columns).fillna(0)


def get_curr_string(s):
    return s.split('\n')[0]

def get_stepbystep_output(repr, encoded_input):
    output = []
    sample = encoded_input
    repr = repr + '\n'
    while True:
        curr_q = get_curr_string(repr).split(' ')[1]
        if curr_q != 'class:':
            curr_ans = sample[get_curr_string(repr).split(' ')[1]].iloc[0]
            current_node = f'|--- {curr_q} {curr_ans}\n'
            repr = repr.split(current_node)[1].replace('\n|---', '\n---').split('\n---')[0].replace('|   |---', '|---')
            output.append({curr_q: curr_ans})
        else:
            break
    output.append({'ответ': get_curr_string(repr).split('class: ')[1]})
    return output

def fit_predict(x):
    df = pd.read_excel('instruments.xlsx')

    X, y = df.iloc[:,:-1], df.iloc[:,-1]

    ohe = OneHotEncoder()
    ohe.fit(X)

    tree = DecisionTreeClassifier(random_state=3)
    tree.fit(ohe.transform(X), y)

    repr = sklearn.tree.export_text(tree, feature_names=list(ohe.columns)) \
            .replace('<= 0.50', 'нет') \
            .replace('>  0.50', 'да')

    return get_stepbystep_output(repr, ohe.transform(x).replace({0.0: 'нет', 1: 'да'}))


