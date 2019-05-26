import numpy as np
import itertools

#Transform text into numeric features: CuntVectorizers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

#Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.svm import SVC

#Score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def classifiers(count_train,count_test,tfidf_train,tfidf_test,y_train,y_test):
    '''This function takes a TF and TfIdf and shows the results for several classifiers.
    :input: TfCountVectorizer for training features
    :input: TfCountVectorizer for test features
    :input: TfidfCountVectorizer for trainning features
    :input: TfidfCountVectorizer for test features
    :input: Series object with labels 'FAKE','REAL' for trainning
    :input: Series object with labels 'FAKE','REAL' for testing    
    '''
    
    #Classifiers definition
    naive_clf= MultinomialNB()
    passive_clf=PassiveAggressiveClassifier(n_iter=50)
    max_clf=LogisticRegression()
    svc_clf = SVC()
    
    #Models and scores dictionary
    model_score_dict={}
    
    
    #NAIVE BAYES CLASSIFIER
    naive_clf.fit(tfidf_train, y_train)
    pred1 = naive_clf.predict(tfidf_test)
    score1 = accuracy_score(y_test, pred1)
    print('\n\nNAIVE BAYES CLASSIFIER','-------------------------------------',sep='\n')    
    print(f"NB with TfIdf accuracy:   {score1:.3f}")
    cm1 = confusion_matrix(y_test, pred1, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm1, classes=['FAKE', 'REAL'])
    model_score_dict['naive_clf_itidf']=score1
    
    naive_clf.fit(count_train, y_train)
    pred2 = naive_clf.predict(count_test)
    score2 = accuracy_score(y_test, pred2)
    print(f"NB with Tf accuracy:   {score2:.3f}")
    cm2 = confusion_matrix(y_test, pred2, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm2, classes=['FAKE', 'REAL'])
    model_score_dict['naive_clf_tfidf']=score2
    
    #PASSIVE AGGRESIVE CLASSIFIER
    passive_clf.fit(tfidf_train, y_train)
    pred3 = passive_clf.predict(tfidf_test)
    score3 = accuracy_score(y_test, pred3)
    print('\n\nPASSIVE AGGRESIVE CLASSIFIER','-------------------------------------',sep='\n')
    print(f"Passive Aggresive with TfIdf accuracy:   {score3:.3f}")
    cm3 = confusion_matrix(y_test, pred3, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm3, classes=['FAKE', 'REAL'])
    model_score_dict['passive_clf_tfidf']=score3
    
    passive_clf.fit(count_train, y_train)
    pred4 = passive_clf.predict(count_test)
    score4 = accuracy_score(y_test, pred4)
    print(f"Passive Aggresive with Tf accuracy:   {score4:.3f}")
    cm4 = confusion_matrix(y_test, pred4, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm4, classes=['FAKE', 'REAL'])
    model_score_dict['passive_clf_tf']=score4
    
    #MAXIMUM ENTROPY CLASSIFIER
    max_clf.fit(tfidf_train, y_train)
    pred5 = max_clf.predict(tfidf_test)
    score5 = accuracy_score(y_test, pred5)
    print('\n\nMAXIMUM ENTROPY CLASSIFIER','-------------------------------------',sep='\n')
    print(f"Maximum Entropy with TfIdf accuracy:   {score5:.3f}")
    cm5 = confusion_matrix(y_test, pred5, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm5, classes=['FAKE', 'REAL'])
    model_score_dict['max_clf_tfidf']=score5
    
    max_clf.fit(count_train, y_train)
    pred6 = max_clf.predict(count_test)
    score6 = accuracy_score(y_test, pred6)
    print(f"Maximum Entropy with Tf accuracy:   {score6:.3f}")
    cm6 = confusion_matrix(y_test, pred6, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm6, classes=['FAKE', 'REAL'])
    model_score_dict['max_clf_tf']=score6
    
    #SVM CLASSIFIER
    svc_clf.fit(tfidf_train, y_train)
    pred7 = svc_clf.predict(tfidf_test)
    score7 = accuracy_score(y_test, pred7)
    print('\n\nSVM CLASSIFIER','-------------------------------------',sep='\n')
    print(f"SVM with TfIdf accuracy:   {score7:.3f}")
    cm7 = confusion_matrix(y_test, pred7, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm7, classes=['FAKE', 'REAL'])
    model_score_dict['svc_clf_tfidf']=score7
    
    svc_clf.fit(count_train, y_train)
    pred8 = svc_clf.predict(count_test)
    score8 = accuracy_score(y_test, pred8)
    print(f"SVM with Tf accuracy:   {score8:.3f}")
    cm8 = confusion_matrix(y_test, pred8, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm8, classes=['FAKE', 'REAL'])
    model_score_dict['svc_clf_tf']=score8
    
    #What is the model with the highest score?
    highest_score = max(model_score_dict, key=model_score_dict.get)  # Just use 'min' instead of 'max' for minimum.
    print(f'\n\nThe best model is {highest_score} with an accuracy of: {model_score_dict[highest_score]:.3f}')
    
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        ...
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')