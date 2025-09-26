import torch
import sys
sys.path.append("../")
# from aif360.datasets.meps_dataset_panel21_fy2016 import MEPSDataset21
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder,KBinsDiscretizer
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


def load_adult_ac1(sen='sex'):
    # data = pd.read_csv("adult.csv")
    train_path = 'data/adult/adult.data'
    test_path = 'data/adult/adult.test'

    column_names = ['age', 'workclass', 'fnlwgt', 'education',
                    'education-num', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'income-per-year']
    na_values = ['?']

    train = pd.read_csv(train_path, header=None, names=column_names,
                        skipinitialspace=True, na_values=na_values)
    test = pd.read_csv(test_path, header=0, names=column_names,
                       skipinitialspace=True, na_values=na_values)

    df = pd.concat([test, train], ignore_index=True)

    del_cols = ['fnlwgt']  # 'education-num'
    df.drop(labels=del_cols, axis=1, inplace=True)

    ##### Drop na values
    dropped = df.dropna()
    count = df.shape[0] - dropped.shape[0]
    print("Missing Data: {} rows removed.".format(count))
    df = dropped

    cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
    ## Implement label encoder instead of one-hot encoder
    for feature in cat_feat:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    #    df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

    ## Implement label encoder instead of one-hot encoder
    cat_feat = ['race']
    for feature in cat_feat:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    bin_cols = ['capital-gain', 'capital-loss']
    for feature in bin_cols:
        bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
        df[feature] = bins.fit_transform(df[[feature]])

    #    df = df[columns]
    label_name = 'income-per-year'

    favorable_label = 1
    unfavorable_label = 0
    favorable_classes = ['>50K', '>50K.']

    pos = np.logical_or.reduce(np.equal.outer(favorable_classes, df[label_name].to_numpy()))
    df.loc[pos, label_name] = favorable_label
    df.loc[~pos, label_name] = unfavorable_label

    X = df.drop(labels=[label_name], axis=1, inplace=False)
    y = df[label_name]

    seed = 42  # randrange(100)
    #    train, test  = train_test_split(df, test_size = 0.15, random_state = seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    from .data_config import adult
    sen_ind = adult.sensitive_feature[sen]
    if sen in adult.sensitive_feature.keys():
        A_train = X_train[:, sen_ind]
        A_test = X_test[:, sen_ind]
    else:
        A_train = None
        A_test = None
        print('error, sen should be \'sex\', \'race\' or \'age\'')
    return (
    df, X_train, y_train.to_numpy().astype('int'), A_train, X_test, y_test.to_numpy().astype('int'), A_test)

def load_compas(sen='race'):
    filepath = 'data/compas/propublica_data_for_fairml.csv'
    column_names = ['Two_yr_Recidivism','Number_of_Priors','score_factor',
                    'Age_Above_FourtyFive','Age_Below_TwentyFive',
                    'African_American','Asian','Hispanic','Native_American','Other',
                    'Female','Misdemeanor']
    df = pd.read_csv(filepath, header=0, names=column_names)

    age_column = [0 for _ in range(len(df))]
    race_column = [0 for _ in range(len(df))]

    for index, row in df.iterrows():
        # age: {<25 = 0, >= 25 = 1}
        if row['Age_Below_TwentyFive'] == 1:
            age_column[index] = 0
        elif row['Age_Above_FourtyFive'] == 1:
            age_column[index] = 1 
        else:
            age_column[index] = 1

        # race: {White = 0, Non-White = 1}
        if row['African_American'] == 1:
            race_column[index] = 1
        elif row['Asian'] == 1:
            race_column[index] = 1
        elif row['Hispanic'] == 1:
            race_column[index] = 1
        elif row['Native_American'] == 1:
            race_column[index] = 1
        elif row['Other'] == 1:
            race_column[index] = 1
        else: # White
            race_column[index] = 0

    # add the new columns
    df.insert(loc = 3, column = 'Age', value = age_column)
    df.insert(loc = 4, column = 'Race', value = race_column)

    # drop the originial columns
    feat_to_drop = ['Age_Above_FourtyFive','Age_Below_TwentyFive',
                    'African_American','Asian','Hispanic','Native_American','Other']
    df = df.drop(feat_to_drop, axis=1)

    # done preprocessing df, now create X and y
    label_name = 'score_factor' # 'Two_yr_Recidivism' is a feature
    X = df.drop(labels = [label_name], axis = 1, inplace = False)
    y = df[label_name]
    # print(X)
    # print(y)

    from sklearn.model_selection import train_test_split
    seed = 2025
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    from .data_config import compas
    sen_ind = compas.sensitive_feature[sen]
    A_train = X_train[:, sen_ind]
    A_test = X_test[:, sen_ind]
    return (
    df, X_train, y_train.to_numpy().astype('int'), A_train, X_test, y_test.to_numpy().astype('int'), A_test)



def german_custom_preprocessing(df):
    def group_credit_hist(x):
        if x in ['A30', 'A31', 'A32']:
            return 'None/Paid'
        elif x == 'A33':
            return 'Delay'
        elif x == 'A34':
            return 'Other'
        else:
            return 'NA'

    def group_employ(x):
        if x == 'A71':
            return 'Unemployed'
        elif x in ['A72', 'A73']:
            return '1-4 years'
        elif x in ['A74', 'A75']:
            return '4+ years'
        else:
            return 'NA'

    def group_savings(x):
        if x in ['A61', 'A62']:
            return '<500'
        elif x in ['A63', 'A64']:
            return '500+'
        elif x == 'A65':
            return 'Unknown/None'
        else:
            return 'NA'

    def group_status(x):
        if x in ['A11', 'A12']:
            return '<200'
        elif x in ['A13']:
            return '200+'
        elif x == 'A14':
            return 'None'
        else:
            return 'NA'

    # status_map = {'A91': 1.0, 'A93': 1.0, 'A94': 1.0,
    #               'A92': 0.0, 'A95': 0.0}  # A91: male
    # df['sex'] = df['personal_status'].replace(status_map)
    status_map = {'A91': 1, 'A93': 1, 'A94': 1, 'A92': 0, 'A95': 0}  # 1: 'male'
    df['sex'] = df['personal_status'].replace(status_map)

    # group credit history, savings, and employment
    df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
    df['savings'] = df['savings'].apply(lambda x: group_savings(x))
    df['employment'] = df['employment'].apply(lambda x: group_employ(x))
    # df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
    df['status'] = df['status'].apply(lambda x: group_status(x))

    df.credit.replace([1, 2], [1, 0], inplace=True)

    return df




def load_german(sen="age"):
    filepath = 'data/german/german.data'
    column_names = ['status', 'month', 'credit_history',
                'purpose', 'credit_amount', 'savings', 'employment',
                'investment_as_income_percentage', 'personal_status',
                'other_debtors', 'residence_since', 'property', 'age',
                'installment_plans', 'housing', 'number_of_credits',
                'skill_level', 'people_liable_for', 'telephone',
                'foreign_worker', 'credit']
    na_values=[]
    df = pd.read_csv(filepath, sep=' ', header=None, names=column_names,na_values=na_values)
    df['age'] = df['age'].apply(lambda x: np.float64(x >= 26))
    df = german_custom_preprocessing(df)
    feat_to_drop = ['personal_status']
    df = df.drop(feat_to_drop, axis=1)
    
    cat_feat = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'installment_plans', 
                'housing', 'skill_level', 'telephone', 'foreign_worker']
    
    
    for f in cat_feat:
        label = LabelEncoder()
        df[f] = label.fit_transform(df[f])      
    
#    bin_cols = ['capital-gain', 'capital-loss']
#    for feature in bin_cols:
#        bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
#        df[feature] = bins.fit_transform(df[[feature]])
    
#    df = df[columns]
    label_name = 'credit'
#    
    favorable_label = 1
    unfavorable_label = 0
    #favorable_classes=['>50K', '>50K.']
    
    
    #pos = np.logical_or.reduce(np.equal.outer(favorable_classes, df[label_name].to_numpy()))
    #df.loc[pos, label_name] = favorable_label
    #df.loc[~pos, label_name] = unfavorable_label
#    
    X = df.drop(labels = [label_name], axis = 1, inplace = False)
    y = df[label_name]
#    
#    
    seed = 42 # randrange(100)
#    train, test  = train_test_split(df, test_size = 0.15, random_state = seed)
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.3, random_state = seed)     
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    from .data_config import german
    sen_ind = german.sensitive_feature[sen]
    A_train = X_train[:, sen_ind]
    A_test = X_test[:, sen_ind]
    return (
    df, X_train, y_train.to_numpy().astype('int'), A_train, X_test, y_test.to_numpy().astype('int'), A_test)


def load_bank(sen='age'):
    file_path = 'data/bank/bank-additional-full.csv'

    column_names = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 
                    'month', 'day_of_week', 'duration', 'emp.var.rate',  
                    'campaign', 'pdays', 'previous', 'poutcome', 'y']
    na_values=['unknown']
    
    df = pd.read_csv(file_path, sep=';', na_values=na_values)
    
    ### Drop na values
    dropped = df.dropna()
    count = df.shape[0] - dropped.shape[0]
    print("Missing Data: {} rows removed.".format(count))
    df = dropped
    columns = ['education=Assoc-acdm', 'education=Assoc-voc', 'education=Bachelors',]
    
    df['age'] = df['age'].apply(lambda x: np.float64(x >= 25))
    
    ## Feature selection
    # features_to_keep = []
    # df = df[features_to_keep]
    
    # Create a one-hot encoding of the categorical variables.
    cat_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    #df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
    
    
    for f in cat_feat:
        label = LabelEncoder()
        df[f] = label.fit_transform(df[f])      
    
#    bin_cols = ['capital-gain', 'capital-loss']
#    for feature in bin_cols:
#        bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
#        df[feature] = bins.fit_transform(df[[feature]])
    #print(df.columns)
    
    df = df[column_names]
    label_name='y'
    favorable_label = 1
    unfavorable_label = 0
    favorable_classes=['yes']
    
    pos = np.logical_or.reduce(np.equal.outer(favorable_classes, df[label_name].to_numpy()))
    df.loc[pos, label_name] = favorable_label
    df.loc[~pos, label_name] = unfavorable_label
    df = df.round(0).astype(int)
    
#    
    X = df.drop(labels = [label_name], axis = 1, inplace = False)
    y = df[label_name]
#    
#    
    seed = 42 # randrange(100)
#    train, test  = train_test_split(df, test_size = 0.15, random_state = seed)
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.15, random_state = seed)        
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    from .data_config import bank
    sen_ind = bank.sensitive_feature[sen]
    if sen in bank.sensitive_feature.keys():
        A_train = X_train[:, sen_ind]
        A_test = X_test[:, sen_ind]
    else:
        A_train = None
        A_test = None
        print('error, sen should be \'sex\', \'race\' or \'age\'')
    return (
    df, X_train, y_train.to_numpy().astype('int'), A_train, X_test, y_test.to_numpy().astype('int'), A_test)


def preprocess_bank_data(data_path='/data/home/mjnn/majianan/fairness-repair/RULER-main/Ruler/data/PGD_dataset/original_data/bank.csv'):
    df = pd.read_csv(data_path, sep=";", encoding='latin-1')

    df.replace('unknown', np.nan, inplace=True)
    for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    data = df.values
    list_index_cat = [1, 2, 3, 4, 6, 7, 8, 10, 15, 16]
    for i in list_index_cat:
        vocab = np.unique(data[:, i])
        mapping = {label: idx for idx, label in enumerate(vocab)}
        data[:, i] = np.array([mapping[str(item)] for item in data[:, i]], dtype=np.int64)
    data = data.astype(np.int32)
    # print(data[:, 0], np.max(data[:, 0]), np.min(data[:, 0]))
    # bins_age = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bins_age = [15, 25, 45, 65, 120]
    bins_balance = [-1e4] + [np.percentile(data[:, 5], percent, axis=0) for percent in [25, 50, 75]] + [2e5]
    bins_day = [0, 10, 20, 31]
    bins_month = [-1, 2, 5, 8, 11]
    bins_duration = [-1.0] + [np.percentile(data[:, 11], percent, axis=0) for percent in [25, 50, 75]] + [6e3]
    bins_campaign = [0.0] + [np.percentile(data[:, 12], percent, axis=0) for percent in [25, 50, 75]] + [1e2]
    bins_pdays = [-10.0] + [np.percentile(data[:, 13], percent, axis=0) for percent in [25, 50, 75]] + [1e3]
    bins_previous = [-1.0] + [np.percentile(data[:, 14], percent, axis=0) for percent in [25, 50, 75]] + [3e2]
    list_index_num = [0, 5, 9, 10, 11, 12, 13, 14]
    list_bins = [bins_age, bins_balance, bins_day, bins_month, bins_duration, bins_campaign, bins_pdays, bins_previous]
    for index, bins in zip(list_index_num, list_bins):
        data[:, index] = np.digitize(data[:, index], bins, right=True)

    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T
    # print(constraint)

    # 0 is sens att
    protected_attribs = [0]
    A_train = X_train[:, 0]
    A_test = X_test[:, 0]
    return (df, X_train, y_train, A_train, X_test, y_test, A_test)

if __name__ == '__main__':
    print(load_compas())