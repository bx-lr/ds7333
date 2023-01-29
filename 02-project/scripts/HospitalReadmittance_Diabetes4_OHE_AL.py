# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 16:13:31 2023

@author: erinm
"""

#Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score

#Read in csv file
data = pd.read_csv("C:/Users/erinm/Documents/7333_QuantifyingTheWorld/CaseStudy2/diabetic_data.csv")
print(data)

#Set python so it displays all column information
pd.options.display.max_columns = None

#Check the size of the data, quick exploration
print(data.shape)
data.info()
data.head()

#Look at basic statistics
print(data.describe())

#Swap the '?' entries for NAN
df = data.replace('?', np.NaN)
na_entries = df.isnull().sum()
na_entries

#Remove columns with >= 50% missing values, i.e., 'weight' & 'medical_specialty'
#"payer_code", 'encounter_id', 'patient_nbr' don't seem to be important. Dropping that too.

df = df.drop(['weight', 'medical_specialty', 'payer_code','encounter_id', 'patient_nbr'], axis = 1)
print(df.shape)

#For those columns missing negligible entries (n=21-1423) we remove the entries
#(alternatively, replace these with something else, like "NA")
cols_to_replace = ['diag_1', 'diag_2', 'diag_3']

# Replace missing values in the diagnosis columns with 'NoDiagnosis'
df[cols_to_replace] = df[cols_to_replace].fillna('NoDiagnosis')

na_rows = df.isnull().sum()
na_rows

#Count unique levels in each column.
unique = df.nunique().sort_values()
print(unique)

#Check for columns with single class features
for col in df.columns:
    print(f'Value counts for column {col}:')
    print(df[col].value_counts())
    print()

#dropping unnecessary columns
df = df.drop(['acetohexamide','tolbutamide', 'miglitol', 'troglitazone', 'tolazamide','examide', 'citoglipton', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone','metformin-pioglitazone', 'chlorpropamide' ], axis = 1)

for col in df.columns:
    print(f'Value counts for column {col}:')
    print(df[col].value_counts())
    print()

#Remove unimportant variables from discharge_disposition_id
#(these are people who died or are in hospice etc)
df = df.loc[~df.discharge_disposition_id.isin([11,13,14,19,20,21])]

#Plot counts of some obvious categorical data
fig, ax = plt.subplots(figsize=(15,10), ncols=2, nrows=3)

sns.countplot(x="readmitted", data=df, ax=ax[0][0])
sns.countplot(x="race", data=df, ax=ax[0][1])
sns.countplot(x="gender", data=df, ax=ax[1][0])
sns.countplot(x="age", data=df, ax=ax[1][1])
sns.countplot(x="readmitted", data=df, ax=ax[2][1])
sns.countplot(x="time_in_hospital", data=df, ax=ax[2][0])

#As seen above, we have too many levels in some categories to one-hot-encode
df.nunique()

#make a copy of df
df2 = df.copy()
df.shape
df2.shape

#Put the readmission groups into a single level
df2['readmitted'] = df2['readmitted'].replace('NO', 'No')
df2['readmitted'] = df2['readmitted'].replace('>30', 'Yes')
df2['readmitted'] = df2['readmitted'].replace('<30', 'Yes')


#imputing missing values into 'other' level
df2['race'].fillna(value='Other', inplace=True)

#gender treatment. Remove the 1 instance of Unknown/Invalid

# Create a mask for rows where the "gender" column is "Unknown/Invalid"
mask = df2["gender"] == "Unknown/Invalid"

# Select only the rows where the mask is False (i.e., where the "gender" column is not "Unknown/Invalid")
df2 = df2[~mask]

# Remove the rows where the "gender" column is "Unknown/Invalid"
df2 = df2.drop(df2.index[df2["gender"] == "Unknown/Invalid"])

# Print the unique values in the "gender" column
print(df2['gender'].value_counts())

#double check for na entries
na_rows2 = df2.isnull().sum()
na_rows2

#Print categorical column levels. These all seem fine for ordinal or onehot
cat_cols = df2.select_dtypes(include='object').columns
for column in cat_cols:
    if df2[column].nunique()<10:
        print(f"Unique values for {column}: {df2[column].unique()}")

#Looking into these bitches
#Diagnosis groupings
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
  

def codemap(df2, codes):
    import pandas as pd
    namecol = df2.columns.tolist()
    for col in namecol:
        temp = [] 
        for num in df2[col]:           
            if ((num is None) | (num in ['NoDiagnosis', '?']) | (pd.isnull(num))): temp.append('NoDiagnosis')
            elif(num.upper()[0] == 'V'): temp.append('supplemental')
            elif(num.upper()[0] == 'E'): temp.append('injury')
            else: 
                lkup = num.split('.')[0]
                temp.append(codes[lkup])           
        df2.loc[:, col] = temp               
    return df2


listcol = ['diag_1', 'diag_2', 'diag_3']
codes = map_now()
df2[listcol] = codemap(df2[listcol], codes)


df2['diag_1'].unique()
df2['diag_2'].unique()
df2['diag_3'].unique()

# Split the data into features and target
X = df2.drop('readmitted', axis=1)
y = df2['readmitted']

df2.info()

#make cat column dataframe
cat_cols = X.select_dtypes(include='object').columns
for column in cat_cols:
    if df2[column].nunique()<50:
        print(f"Unique values for {column}: {df2[column].unique()}")

print(y)
print(cat_cols)

#numeric columns
#num_cols = df2.select_dtypes(include='int64').columns
num_cols = df2.filter(items=df2.select_dtypes(include='int64').columns)

# scale numeric features
# Normalize the numeric columns not one hot encoded

scaler = StandardScaler()
X_scale = scaler.fit_transform(num_cols)
X_scale = pd.DataFrame(X_scale, columns=num_cols.columns)

# X_scale has all numeric non-OHE features normalized
X_scale.info()
X_scale.describe()


#One-hot-encode categorical columns
df_dum = pd.get_dummies(X, columns=cat_cols)
df_dum.info()

# Convert df_dum to int64
df_dum = df_dum.astype('int64')


#merge cat and num columns together
print(type(df_dum))
print(type(X_scale))
df_dum.shape
X_scale.shape

df_dum.reset_index(drop=True, inplace=True)
X_scale.reset_index(drop=True, inplace=True)

df3 = pd.concat([df_dum, X_scale], axis=1)

X_scale.isna().sum().sum()
df_dum.isna().sum().sum()
df3.isna().sum().sum()

df4 = df3.copy()
df4.head()

#check for multicolinearity
X_c = df4.copy()
corr_matrix = X_c.corr()

#Create threshold for high correlation
threshold = 1

# Print the variables that are highly correlated with each other
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            print(f"{corr_matrix.columns[i]} is highly correlated with {corr_matrix.columns[j]} with correlation coefficient {corr_matrix.iloc[i, j]:.2f}")

# Normalize the numeric columns not one hot encoded
# new_num_col = df4.select_dtypes(include='int64').columns
# scaler = StandardScaler()
# X_scale = scaler.fit_transform(df4[new_num_col])
# X_scale = pd.DataFrame(X_scale, columns=df4[new_num_col].columns)

# Initialize the model
model = LogisticRegression()

#Create basic LR model for coef analysis

clf= model.fit(df4, y)
coef = model.coef_
print(coef)

# Perform 5-fold cross-validation and get predictions
y_pred = cross_val_predict(model, df4, y, cv=5)

# Compute evaluation metrics
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, pos_label='Yes')
recall = recall_score(y, y_pred, pos_label='Yes')
f1 = f1_score(y, y_pred, pos_label='Yes')
# roc_auc = roc_auc_score(y, y_pred)

# Print the evaluation metrics
print("Accuracy: {:.2f}".format(acc))
print("Precision: {:.2f}".format(prec))
print("Recall: {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1))
# print("ROC-AUC: {:.2f}".format(roc_auc))

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, df4, y, cv=5)

# Print the mean and standard deviation of the cross-validation scores
print("Mean cross-validation score: {:.2f}".format(cv_scores.mean()))
print("Standard deviation of cross-validation scores: {:.2f}".format(cv_scores.std()))

# Extract the feature importances
importance = model.coef_

feature_names = df4.columns
# Print the feature importances
for feature, importance in zip(feature_names, importance[0]):
    print(feature, ':', importance)


