# import machine learning tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

# read datasets from given .csv files
df_train = pd.read_csv("af2_dataset_training_labeled.csv")
df_test = pd.read_csv("af2_dataset_testset_unlabeled.csv")

# categorize data into numerical and categorical data
df_num = df_train[['feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT',
                   'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11',
                   'feat_DSSP_12', 'feat_DSSP_13', 'entry_index']]
df_cat = df_train[['feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K',
                   'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V',
                   'feat_W', 'feat_Y', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I',
                   'feat_DSSP_T', 'feat_DSSP_S', 'y_Ligand']]

# calculate correlation between categorical data
df_cat.corr()
cat_and_y = df_cat.merge(df_train['y_Ligand'], left_index=True, right_index=True)
print(cat_and_y.corr())

# graph numerical data to visualize patterns
for col in df_num.columns:
    plt.hist(df_num[col])
    plt.title(col)
    plt.show()

# calculate correlation between numerical data
print(df_num.corr())

# clean_df(df) 'cleans' data by dropping unnecessary data and replacing null values with median values
def clean_df(df: pd.DataFrame) -> None:
    # remove irrelevant data
    df.drop(['Unnamed: 0', 'coord_X', 'coord_Y', 'coord_Z', 'annotation_atomrec', 'annotation_sequence', 'entry'],
            axis=1, inplace=True)
    cols = ['feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT',
            'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11',
            'feat_DSSP_12', 'feat_DSSP_13']
    
    # replace 0 (null) values with medians
    for col in cols:
        if (df[col] == 0).all():
            df[col] = None
        df[col].fillna(df[col].median(), inplace=True)

# assign 'Unnamed: 0' column as testing ID
test_ids = df_test['Unnamed: 0']

# clean datasets for training
clean_df(df_train)
clean_df(df_test)

# drop result from training data
X = df_train.drop('y_Ligand', axis=1)
y = df_train['y_Ligand']

# undersample training data
undersampler = RandomUnderSampler()
X_under, y_under = undersampler.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.2)

# uses logistic regression to train the model with X and y training sets
clf = LogisticRegression(max_iter=10000)
clf = clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(accuracy_score(y_test, preds))

# calculate and print F1 score
f1 = f1_score(y_test, preds)
recall = recall_score(y_test, preds)
precision = precision_score(y_test, preds)
roc_auc = accuracy_score(y_test, preds)
print('F1 score:', f1)
print('recall: ', recall)
print('precision: ', precision)
print("ROC AUC score:", roc_auc)

# run testing set through the trained model to calculate predictions
submission_preds = clf.predict(df_test)

result = pd.DataFrame({
    'id': test_ids,
    'Predicted': submission_preds
})

# write results to .csv file
result.to_csv('submission.csv', index=False)