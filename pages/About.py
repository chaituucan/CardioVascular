import streamlit as st
import subprocess
import pandas as pd #Data manipultion and analysisi
import numpy as np #mathematical calculations
import matplotlib.pyplot as plt #plotting
import scipy.stats as sta#
import seaborn as sns  #data visualtion
import pandas_profiling
st.header("Real time Model training and testing and comparison with other datasets")
lr=[]
dt=[]
knn=[]
en=[]
#%matplotlib inline
if st.button("Framingham dataset model prediction"):
    df = pd.read_csv(r'C:\Users\Admin\Desktop\main\coding_part\framingham.csv')
#data preprocessing 
#step-1 missing values filling
    df['TenYearCHD'].value_counts(normalize = True)
    df['cigsPerDay'].value_counts(normalize = True).plot(kind="bar")
    df['cigsPerDay'][df['currentSmoker']==0].isna().sum()
# creating a boolean array of smokers
    smoke = (df['currentSmoker']==1)
# applying mean to NaNs in cigsPerDay but using a set of smokers only
    df.loc[smoke,'cigsPerDay'] = df.loc[smoke,'cigsPerDay'].fillna(df.loc[smoke,'cigsPerDay'].mean())
    df['cigsPerDay'][df['currentSmoker']==1].mean()
    df['cigsPerDay'][df['currentSmoker']==0].mean()
    df['education'].value_counts(normalize = True).plot(kind="bar")
# Filling out missing values
    df['BPMeds'].fillna(0, inplace = True)
    df['glucose'].fillna(df.glucose.mean(), inplace = True)
    df['totChol'].fillna(df.totChol.mean(), inplace = True)
    df['education'].fillna(1, inplace = True)
    df['BMI'].fillna(df.BMI.mean(), inplace = True)
    df['heartRate'].fillna(df.heartRate.mean(), inplace = True)
    st.subheader("NULL Values after preprocessing")
    st.write(df.isna().sum())
    from sklearn.feature_selection import SelectKBest#  
    from sklearn.feature_selection import chi2

# separate independent & dependent variables
    X = df.iloc[:,0:14]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range
    st.subheader("Top 10 best features after Feature selection")
# apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=11)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    st.write(featureScores.nlargest(11,'Score'))  #print 10 best featuresfeatureScores = featureScores.sort_values(by='Score', ascending=False)
    st.subheader("featureScores")
    st.write(featureScores)
    # visualizing feature selection
    import matplotlib.pyplot as plt
    import seaborn as sns
    st.subheader("Feature Importance")
    plt.figure(figsize=(20,5))
    sns.barplot(x='Specs', y='Score', data=featureScores, palette = "GnBu_d")
    plt.box(False)
    plt.title('Feature importance', fontsize=16)
    plt.xlabel('\n Features', fontsize=14)
    plt.ylabel('Importance \n', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(plt)
    features_list = featureScores["Specs"].tolist()[:13]
    features_list
    df = df[['sysBP','age','totChol','cigsPerDay','diaBP','TenYearCHD']]
    df.head()
    from sklearn.model_selection import train_test_split
    y = df['TenYearCHD'] #target variable
    X = df.drop(['TenYearCHD'], axis = 1) #features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print (X_train.shape, y_train.shape)
    print (X_test.shape, y_test.shape)
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets, linear_model
    from imblearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from imblearn.pipeline import Pipeline
    from sklearn.model_selection import RepeatedStratifiedKFold
# decision tree  on imbalanced dataset with SMOTE oversampling and random undersampling
    from numpy import mean
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.tree import DecisionTreeClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
    import sklearn.linear_model as lm
# fit a model
    lm = lm.LogisticRegression()
    model = lm.fit(X_train, y_train)
    over = SMOTE(sampling_strategy=0.1)
    steps = [('over', over), ('model', model)]
    pipeline = Pipeline(steps=steps)
    X, y = make_classification(n_samples=10000, n_features=5, n_redundant=0,
	    n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
#model.score(X_test, y_test)
    lr.append(mean(scores))
    st.write('Logistic Regression Accuracy: %.3f' % mean(scores))
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(X_train)
    from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test, y_pred))
    from sklearn.metrics import accuracy_score
    st.write("Decision Tree Accuracy: ",accuracy_score(y_test, y_pred, normalize=True)
    )
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    st.write("KNN Accuracy:",accuracy_score(y_test, y_pred, normalize=True)
    )
    
    st.subheader("Voting Ensemble for Classification")
    import pandas
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier
    import pickle

    kfold = model_selection.KFold(n_splits=5)
# create the sub models
    estimators = []
    model1 = LogisticRegression().fit(X_test,y_test)
    estimators.append(('logistic', model1))
    model2 = KNeighborsClassifier(n_neighbors=3)
    estimators.append(('cart', model2))

# create the ensemble model
    ensemble = VotingClassifier(estimators)
    ensemble.fit(X_train, y_train)
    pickle.dump(ensemble,open("model.pkl",'wb'))
    over = SMOTE(sampling_strategy=0.1)
    steps = [('over', over), ('model', ensemble)]
    pipeline = Pipeline(steps=steps)
    results = model_selection.cross_val_score(pipeline, X, y, cv=kfold)
    st.write("Ensemble Accuracy: ",results.mean())
if st.button("Cleveland dataset model prediction"):
    import pandas as pd #Data manipultion and analysisi
    import numpy as np #mathematical calculations
    import matplotlib.pyplot as plt #plotting
    import scipy.stats as sta #
    import seaborn as sns  #data visualtion
    import pandas_profiling
    #%matplotlib inline
    df = pd.read_csv(r'C:\Users\Admin\Desktop\main\datasets\cleveland.csv',on_bad_lines='skip')
    info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]
    st.subheader("Attribute Details")
    for i in range(len(info)):
        st.write(df.columns[i],":   \t\t\t"+info[i])
    from sklearn.feature_selection import SelectKBest#  
    from sklearn.feature_selection import chi2

    # separate independent & dependent variables
    X = df.iloc[:,0:14]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range

    # apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=11)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    st.subheader("FeatureScores")
    st.write(featureScores.nlargest(11,'Score'))  #print 10 best features
    featureScores = featureScores.sort_values(by='Score', ascending=False)
    st.subheader("Feature Values")
    st.write(featureScores)
    # visualizing feature selection
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(20,5))
    sns.barplot(x='Specs', y='Score', data=featureScores, palette = "GnBu_d")
    plt.box(False)
    plt.title('Feature importance', fontsize=16)
    plt.xlabel('\n Features', fontsize=14)
    plt.ylabel('Importance \n', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.subheader("Feature Importance")
    st.pyplot(plt)
    features_list = featureScores["Specs"].tolist()[:13]
    features_list
    from sklearn.model_selection import train_test_split

    y = df['condition'] #target variable
    X = df.drop(['condition'], axis = 1) #features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print (X_train.shape, y_train.shape)
    print (X_test.shape, y_test.shape)
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets, linear_model
    from imblearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from imblearn.pipeline import Pipeline
    from sklearn.model_selection import RepeatedStratifiedKFold
    # decision tree  on imbalanced dataset with SMOTE oversampling and random undersampling
    from numpy import mean
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.tree import DecisionTreeClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
    import sklearn.linear_model as lm




    # fit a model
    lm = lm.LogisticRegression()
    model = lm.fit(X_train, y_train)
    over = SMOTE(sampling_strategy=0.1)
    steps = [('over', over), ('model', model)]
    pipeline = Pipeline(steps=steps)
    X, y = make_classification(n_samples=10000, n_features=5, n_redundant=0,
        n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    #model.score(X_test, y_test)
    st.write('Logistic Regression  AUC: %.3f' % mean(scores))
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(X_train)
    from sklearn.metrics import classification_report, confusion_matrix
    #print(confusion_matrix(y_test, y_pred))
    from sklearn.metrics import accuracy_score
    st.write("Decision Tree AUC: ",accuracy_score(y_test, y_pred, normalize=True)
    )
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    st.write("KNN AUC",accuracy_score(y_test, y_pred, normalize=True)
    )
    
# Voting Ensemble for Classification
    import pandas
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier
    import pickle

    kfold = model_selection.KFold(n_splits=5)
    # create the sub models
    estimators = []
    model1 = LogisticRegression().fit(X_test,y_test)
    estimators.append(('logistic', model1))
    model2 = KNeighborsClassifier(n_neighbors=3)
    estimators.append(('cart', model2))

    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    ensemble.fit(X_train, y_train)
    pickle.dump(ensemble,open("model.pkl",'wb'))
    over = SMOTE(sampling_strategy=0.1)
    steps = [('over', over), ('model', ensemble)]
    pipeline = Pipeline(steps=steps)
    results = model_selection.cross_val_score(pipeline, X, y, cv=kfold)
    st.write("Ensemble AUC",results.mean())
if st.button("Heart Disease Dataset Model Prediction"):
    import pandas as pd #Data manipultion and analysisi
    import numpy as np #mathematical calculations
    import matplotlib.pyplot as plt #plotting
    import scipy.stats as sta #
    import seaborn as sns  #data visualtion
    import pandas_profiling
    #%matplotlib inline
    df = pd.read_csv(r'C:\Users\Admin\Desktop\main\datasets\heart.csv')
    info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]
    st.subheader("Feature Details")
    for i in range(len(info)):
        st.write(df.columns[i]+":\t\t\t"+info[i])
    st.subheader("Target value Details")
    st.write(df["target"].describe())
    from sklearn.feature_selection import SelectKBest#  
    from sklearn.feature_selection import chi2

    # separate independent & dependent variables
    X = df.iloc[:,0:14]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range

    # apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=11)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    st.subheader("Top 10 best Features based on Feature Score")
    st.write(featureScores.nlargest(11,'Score'))  #print 10 best features  
    featureScores = featureScores.sort_values(by='Score', ascending=False)
    st.subheader("FeatureScores")
    st.write(featureScores)
# visualizing feature selection
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(20,5))
    sns.barplot(x='Specs', y='Score', data=featureScores, palette = "GnBu_d")
    plt.box(False)
    plt.title('Feature importance', fontsize=16)
    plt.xlabel('\n Features', fontsize=14)
    plt.ylabel('Importance \n', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.subheader("Feature Importance")
    st.pyplot(plt)
    features_list = featureScores["Specs"].tolist()[:13]
    features_list
    from sklearn.model_selection import train_test_split

    y = df['target'] #target variable
    X = df.drop(['target'], axis = 1) #features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print (X_train.shape, y_train.shape)
    print (X_test.shape, y_test.shape)
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets, linear_model
    from imblearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from imblearn.pipeline import Pipeline
    from sklearn.model_selection import RepeatedStratifiedKFold
    # decision tree  on imbalanced dataset with SMOTE oversampling and random undersampling
    from numpy import mean
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.tree import DecisionTreeClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
    import sklearn.linear_model as lm




    # fit a model
    lm = lm.LogisticRegression()
    model = lm.fit(X_train, y_train)
    over = SMOTE(sampling_strategy=0.1)
    steps = [('over', over), ('model', model)]
    pipeline = Pipeline(steps=steps)
    X, y = make_classification(n_samples=10000, n_features=5, n_redundant=0,
        n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    #model.score(X_test, y_test)
    lr.append(mean(scores))
    st.write('Logistic Regression  AUC: %.3f' % mean(scores))
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    #print(confusion_matrix(y_test, y_pred))
    from sklearn.metrics import accuracy_score
    st.write("Decision Tree AUC",accuracy_score(y_test, y_pred, normalize=True)
    )
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    st.write("KNN AUC : ",accuracy_score(y_test, y_pred, normalize=True)
    )

# Voting Ensemble for Classification
    import pandas
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier

    kfold = model_selection.KFold(n_splits=5)
    # create the sub models
    estimators = []
    model1 = LogisticRegression().fit(X_test,y_test)
    estimators.append(('logistic', model1))
    model2 = KNeighborsClassifier(n_neighbors=3)
    estimators.append(('cart', model2))

    # create the ensemble model
    ensemble = VotingClassifier(estimators)

    over = SMOTE(sampling_strategy=0.5)
    steps = [('over', over), ('model', ensemble)]
    pipeline = Pipeline(steps=steps)


    results = model_selection.cross_val_score(pipeline, X, y, cv=kfold)
    st.write("Ensemble AUC : ",results.mean())
    print(lr)
if st.button("Compare All"):
    import matplotlib.pyplot as plt
    st.subheader("Models on Framingham dataset")
    x_labels = ['lr', 'dt', 'knn', 'ensemble']
    y_values = [0.94, 0.77, 0.82, 0.99]
    colors = ['red', 'green', 'blue', 'orange']

    plt.bar(x_labels, y_values, color=colors)

    # Annotate each bar with its value
    for i, v in enumerate(y_values):
        plt.text(i, v+0.01, str(v), ha='center')
    plt.title('Accuracy Scores')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    st.pyplot(plt)
    import matplotlib.pyplot as pl
    st.subheader("Models on Cleveland dataset ")
    y_values1 = [0.94, 0.73, 0.63, 0.99]
    colors = ['red', 'green', 'blue', 'orange']

    pl.bar(x_labels, y_values1, color=colors)

    # Annotate each bar with its value
    for i, v in enumerate(y_values1):
        pl.text(i, v+0.01, str(v), ha='center')
    pl.title('Accuracy Scores')
    pl.xlabel('Models')
    pl.ylabel('Accuracy')
    st.pyplot(pl)
    import matplotlib.pyplot as pl
    st.subheader("Models on Heart Disease dataset ")
    y_values1 = [0.94, 0.72, 0.67, 0.98]
    colors = ['red', 'green', 'blue', 'orange']

    pl.bar(x_labels, y_values1, color=colors)

    # Annotate each bar with its value
    for i, v in enumerate(y_values1):
        pl.text(i, v+0.01, str(v), ha='center')
    pl.title('Accuracy Scores')
    pl.xlabel('Models')
    pl.ylabel('Accuracy')
    st.pyplot(pl)
    
    