# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 12:34:39 2019

@author: luis-
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from info_gain import info_gain
import numpy as np
from scipy.stats import pearsonr
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

df = pd.read_csv('DatasetML.csv')

#==============================================================================
# PREPROCESSING
#==============================================================================

#CHECKING FOR MISSING VALUES
#No missing values found
print("NaNs per column:")
print(df.isna().sum())

#NORMALIZE NUMERICAL FEATURES & PLOT DISTRIBUTIONS
#Most features are either stable or bell-shapped with a right tail.
#No significant outliers seem to be present
for column in df[df.columns[pd.Series(df.columns).str.startswith('N')]]:
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    img_name=("Viz/distribution_"+column+"_normalized.png")
    sn.distplot(df[column]).get_figure().savefig(img_name)
    plt.clf()

#ANALYZE CARDINALITY
#Cardinality is relatively low which enables one-hot encoding.
print("\nUnique categorical values:")
for column in df[df.columns[pd.Series(df.columns).str.startswith('C')]]:
    print("Column ",column," unique values: ",df[column].unique())

#==============================================================================
# FEATURE IMPORTANCE
#==============================================================================

#CORRELATION ANALYSIS (numerical)
#N1 and N2 appear to have the highest correlation to the output but also seem to suffer from multicolinearity.
#A somewhat significant negative correlation also appears between N2 and N3
#A somewhat significant positive correlation also appears between N4 and N5
plt.figure(figsize=(10, 7))
plot = sn.heatmap(df.corr(), square=True, annot=True).get_figure().savefig("Viz/correlation_matrix.png")
plt.clf()

#INFORMATION GAIN RATIO ANALYSIS (categorical)
#Mostly low values. C1 and C2 are the highest
print("\nInformation Gain Ratio:")
for column in df[df.columns[pd.Series(df.columns).str.startswith('C')]]:
    print("Info gain ratio for column ", column, ": ", info_gain.info_gain_ratio(df[column], df['LABEL']))

#plot = sn.scatterplot(x="N3", y="N2", data=df[['N3','N2']]).get_figure().savefig("N1_N2_scatter.png")
#plt.clf()
    
#==============================================================================
# FEATURE ENGINEERING
#==============================================================================

#CATEGORICAL VARIABLES 1-HOT ENCODING
df = pd.get_dummies(df)

#Plot new correlation matrix
plt.figure(figsize=(50, 47))
plot = sn.heatmap((df[df.columns[pd.Series(df.columns).str.contains('_')]].join(df['LABEL'])).corr(), square=True, annot=True).get_figure().savefig("Viz/correlation_matrix_1hot.png")
plt.clf()

#PCA
#Project feature pairs (N1,N2), (N4,N5) into a new axis using PCA.
pca = PCA(n_components=1)
principalComponents = pca.fit_transform(df[['N1','N2']])
df['N1N2_pcomponent'] = principalComponents

pca = PCA(n_components=1)
principalComponents = pca.fit_transform(df[['N4','N5']])
df['N4N5_pcomponent'] = principalComponents

#Plot new correlation matrix
plt.figure(figsize=(10, 7))
plot = sn.heatmap(df[['N1','N2','N4','N5','N1N2_pcomponent','N4N5_pcomponent', 'LABEL']].corr(), square=True, annot=True).get_figure().savefig("Viz/correlation_matrix_pca.png")
plt.clf()

#==============================================================================
# FEATURE SELECTION
#==============================================================================

#Select all features that show a Pearson's Correlation above 0.1 or below -0.1
selected_features = []
for column in df.drop(['N1','N2','N4','N5','LABEL'], axis=1):
    p_corr, _ = pearsonr(df[column], df['LABEL'])
    if (p_corr>0.1) or (p_corr<-0.1):
        selected_features.append(column)
selected_features.append('LABEL')

final_df = df[selected_features]

#Plot final correlation matrix
plt.figure(figsize=(20, 17))
plot = sn.heatmap(final_df.corr(), square=True, annot=True).get_figure().savefig("Viz/correlation_matrix_final.png")
plt.clf()

#==============================================================================
# MODELING
#==============================================================================

#Initial split of 70/30 for train/test data. Seed fixed at 42 for reproducibility.
df_train, df_test = train_test_split(final_df, test_size=0.3, random_state=42)

#LOGISTIC REGRESSION
#The low ammount of samples made me choose to use a cross validation setup.
#Only 30% of the samples belong to class 1. F1 score is used to obtain an unbiased evaluation of the model.

d_tree = tree.DecisionTreeClassifier()
scores = cross_val_score(d_tree, df_train.drop(['LABEL'], axis=1), df_train['LABEL'], cv=10, scoring='f1')
print("\nDecision Tree:\nF1 cross-val score: %0.2f +/- %0.2f" % (scores.mean(), scores.std() * 2))

log_reg = LogisticRegression(penalty='l2', C=1, solver='lbfgs', max_iter=100)
scores = cross_val_score(log_reg, df_train.drop(['LABEL'], axis=1), df_train['LABEL'], cv=10, scoring='f1')
print("\nLogistic Regression:\nF1 cross-val score: %0.2f +/- %0.2f" % (scores.mean(), scores.std() * 2))

r_forest = RandomForestClassifier(max_depth=3)
scores = cross_val_score(r_forest, df_train.drop(['LABEL'], axis=1), df_train['LABEL'], cv=10, scoring='f1')
print("\nRandom Forest:\nF1 cross-val score: %0.2f +/- %0.2f" % (scores.mean(), scores.std() * 2))

g_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3)
scores = cross_val_score(g_boost, df_train.drop(['LABEL'], axis=1), df_train['LABEL'], cv=10, scoring='f1')
print("\nRandom Forest:\nF1 cross-val score: %0.2f +/- %0.2f" % (scores.mean(), scores.std() * 2))

#==============================================================================
# MODEL SELECTION
#==============================================================================
final_model = r_forest.fit(final_df.drop(['LABEL'], axis=1), final_df['LABEL'])
pickle.dump(final_model, open("final_model.sav", 'wb'))

#a small sample of data to be used to test the REST API
final_df.head().drop(['LABEL'], axis=1).to_csv('request_sample.csv', index=False)
