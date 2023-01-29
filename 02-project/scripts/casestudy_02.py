import argparse
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score


def map_now():
    listname = [('infections', 139),
                ('neoplasms', (239 - 139)),
                ('endocrine', (279 - 239)),
                ('blood', (289 - 279)),
                ('mental', (319 - 289)),
                ('nervous', (359 - 319)),
                ('sense', (389 - 359)),
                ('circulatory', (459-389)),
                ('respiratory', (519-459)),
                ('digestive', (579 - 519)),
                ('genitourinary', (629 - 579)),
                ('pregnancy', (679 - 629)),
                ('skin', (709 - 679)),
                ('musculoskeletal', (739 - 709)),
                ('congenital', (759 - 739)),
                ('perinatal', (779 - 759)),
                ('ill-defined', (799 - 779)),
                ('injury', (999 - 799))]
    dictcout = {}
    count = 1
    for name, num in listname:
        for i in range(num):
            dictcout.update({str(count): name})
            count += 1
    return dictcout


def codemap(indf, codes):
    namecol = indf.columns.tolist()
    for col in namecol:
        temp = []
        for num in indf[col]:
            if (num is None):
                temp.append('NoDiagnosis')
            elif (num in ['NoDiagnosis', '?']):
                temp.append('NoDiagnosis')
            elif (pd.isnull(num)):
                temp.append('NoDiagnosis')
            elif num.upper()[0] == 'V':
                temp.append('supplemental')
            elif num.upper()[0] == 'E':
                temp.append('injury')
            else:
                lkup = num.split('.')[0]
                temp.append(codes[lkup])
        indf.loc[:, col] = temp
    return indf


def preprocess_dataframe(indf):
    indf.replace('?', np.NaN, inplace=True)
    to_drop = ['weight',
               'medical_specialty',
               'payer_code',
               'encounter_id',
               'patient_nbr',
               'acetohexamide',
               'tolbutamide',
               'miglitol',
               'troglitazone',
               'tolazamide',
               'examide',
               'citoglipton',
               'glipizide-metformin',
               'glimepiride-pioglitazone',
               'metformin-rosiglitazone',
               'metformin-pioglitazone',
               'chlorpropamide']
    indf.drop(to_drop, axis=1, inplace=True)
    cols_to_replace = ['diag_1', 'diag_2', 'diag_3']
    indf[cols_to_replace] = indf[cols_to_replace].fillna('NoDiagnosis')
    indf['readmitted'] = indf['readmitted'].replace('NO', 'No')
    indf['readmitted'] = indf['readmitted'].replace('>30', 'Yes')
    indf['readmitted'] = indf['readmitted'].replace('<30', 'Yes')
    indf['race'].fillna(value='Other', inplace=True)
    indf.drop(indf.index[indf["gender"] == "Unknown/Invalid"], inplace=True)
    listcol = ['diag_1', 'diag_2', 'diag_3']
    codes = map_now()
    indf[listcol] = codemap(indf[listcol], codes)

    return indf


def main(input):
    df = pd.read_csv(input)
    # process the dataframe
    df = preprocess_dataframe(df)
    # Split the data into features and target
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
    # make numeric dataframe
    num_cols = df.filter(items=df.select_dtypes(include='int64').columns)
    # scale numeric dataframe
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(num_cols)
    X_scale = pd.DataFrame(X_scale, columns=num_cols.columns)

    # make cat column dataframe
    cat_cols = X.select_dtypes(include='object').columns

    # one hot encoding
    df_dum = pd.get_dummies(X, columns=cat_cols)
    df_dum = df_dum.astype('int64')
    df_dum.reset_index(drop=True, inplace=True)
    X_scale.reset_index(drop=True, inplace=True)

    # create scaled and onehot encoded dataframe
    encdf = pd.concat([df_dum, X_scale], axis=1)

    # Initialize the model
    model = LogisticRegression()

    # Create basic LR model for coef analysis
    _ = model.fit(encdf, y)
    y_pred = cross_val_predict(model, encdf, y, cv=5)

    # Compute evaluation metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, pos_label='Yes')
    recall = recall_score(y, y_pred, pos_label='Yes')
    f1 = f1_score(y, y_pred, pos_label='Yes')

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, encdf, y, cv=5)

    # Print the evaluation metrics
    print('Evaluation Metrics: ')
    print("\tAccuracy: {:.2f}".format(acc))
    print("\tPrecision: {:.2f}".format(prec))
    print("\tRecall: {:.2f}".format(recall))
    print("\tF1-score: {:.2f}".format(f1))
    print("\tMean cross-validation score: {:.2f}".format(cv_scores.mean()))
    print("\tStdDev cross-validation scores: {:.2f}".format(cv_scores.std()))
    print('')

    importance = model.coef_

    feature_names = encdf.columns
    # Print the feature importances
    coef_dict = {key: val for key, val in zip(feature_names, importance[0])}
    s_dict = dict(sorted(
                    coef_dict.items(),
                    reverse=True,
                    key=lambda item: abs(item[1])))
    keys = list(s_dict.keys())[:10]
    print('Coefficient Importance: ')
    for k in keys:
        print('\t', k, s_dict[k])
    return


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, help='input csv location')
    args = ap.parse_args()
    warnings.filterwarnings('ignore')
    main(args.input)
