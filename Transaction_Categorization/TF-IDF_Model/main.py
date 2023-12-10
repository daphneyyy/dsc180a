# import required packages
from data_processing import *
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC 
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def main():
    # Define the feature dataframe "X" and prediction column "y".
    X = dff[["memo",'amount', 'month', 'day','weekday']]
    y = dff['category']

    # Doing the train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    # standardize these new_added features
    std_pl = Pipeline(steps=[('std',StandardScaler())])
    # one-hot-encode nominal features
    encode_pl = Pipeline(steps=[('encode',OneHotEncoder(handle_unknown='ignore'))])

    # create columntransformer
    ## Using tf-idf to encode the memo column with ngram within maximum of three. 
    ## Using standardize pipline to normalize the amount.
    ## Using one-hot encode for date value.
    preproc1 = ColumnTransformer( transformers = [
        ('tfidf', TfidfVectorizer(ngram_range = (1,3)), 'memo'),
        ('quant',std_pl,['amount']),
        ('cat', encode_pl, ['month', 'day', 'weekday'])
    ] )


    # create the final pipline
    # using Linear Support Vector Classification model
    pl = Pipeline([
    ('preprocessor', preproc1),
    ('clf', LinearSVC())
    ])

    # build a model
    mod = pl.fit(X_train,y_train)

    # calcualte the accuracy score for our model
    mod.score(X_test,y_test)

    # getting the metric report to evaluate our model performance
    pred_test = pl.predict(X_test)
    print('Classification report: \n', classification_report(y_test, pred_test))
    conf_mat = confusion_matrix(y_test, pred_test)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=dff['category'].unique(), yticklabels=dff['category'].unique())
    
if __name__ == "__main__":
    main()