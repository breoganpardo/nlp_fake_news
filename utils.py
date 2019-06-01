#This python script file contains several functions that are used in the
#notebook nlp_fake_notebook.ipynb. The purpose of this file is to reduce
#the length of the notebook, facilitating the understanding of the ideas
#discussed on it.

#Basic imports
import pandas as pd
import numpy as np
import itertools
import nltk

#NLTK imports
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import DefaultTagger,UnigramTagger,NgramTagger
from nltk.corpus import brown

#Download brown corpus to be used in train_for_n_grams()
#nltk.download('brown')


##############################################################################
####################           NLP FUNCTIONS              ####################
##############################################################################

def wordnet_lemmatizer(df, label=[1, 2]):
    '''This function iterates over all rows of the dataframe and only the columns
    in stated in the input parameter label. In each iteration takes the text of
    the cell and lemmatize it using the nltk_tagger own defined function.

    Parameters
    ----------
    df: pandas dataframe
    label: column number of the columns that needs to be lemmatized

    Returns
    -------
    df: lemmatized dataframe'''

    for column in label:
        # Every loop in i is a new row in the dataframe
        for i in range(df.shape[0]):
            # tokenize each row
            text_list = word_tokenize(str(df.iloc[i, column]), language='english')


            if column == label[0]:
                stamp = 'title_wordnet_lemmatized'
            elif column == label[1]:
                stamp = 'text_wordnet_lemmatized'

            df.at[i, stamp] = nltk_tagger(text_list)
            text_list = []

    return df


def nltk_tagger(words):
    '''This function takes a list of words and returns the same list transformed intro the lemmatized string.
    Parameters
    ----------
    words: list of words

    Returns
    -------
    lemmatized_string: string with lemmatized words
    '''

    #Change words to lower case
    words=convert_list_to_lower(words)
    wnlt = WordNetLemmatizer()

    #POS tagging the words based on WordNet
    #This point can be improved with N-grams. Next iteration will improve tag.
    tagged_words = nltk.pos_tag(words)
    #tagged_words: [('word1','tag1'),('word2','tag2'),...,('wordn','tagn')]

    #Create a string with all the words lemmatized
    lemmatized_string=''
    for (w,t) in tagged_words:
        lemmatized_string+=' '+wnlt.lemmatize(w,  pos=get_wordnet_pos(t))
    return lemmatized_string


def get_wordnet_pos(treebank_tag):
    '''This function translates the nltk.pos_tag tags into the wordnet tag
    so that WordNetLemmatizer works.
    Parameters
    ----------
    treebank_tag: POS tag

    Returns
    -------
    :wordnet tags
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN #by default is noun



def ngram_lemmatizer(df, tagger, label=[1, 2]):
    '''This function iterates over all rows of the dataframe and only the columns
    stated in the input parameter label. In each iteration takes the text of
    the cell and lemmatize it using the n_gram_POS_tagger own defined function.

    Parameters
    ----------
    df: input dataframe
    tagger: the tagger function used
    label= number of columns to iterate over

    Returns
    -------
    df: dataframe lemmatized
    '''

    for column in label:
        # Every loop in i is a new row in the dataframe
        for i in range(df.shape[0]):
            # tokenize each row
            text_list = word_tokenize(str(df.iloc[i, column]), language='english')


            if column == label[0]:
                stamp = 'title_ngram_lemmatized'
            elif column == label[1]:
                stamp = 'text_ngram_lemmatized'

            df.at[i, stamp] = n_gram_POS_tagger(text_list,tagger)
            text_list = []

    return df


def n_gram_POS_tagger(words,t4):
    '''This function performs POS tagging based on concatenated n-grams + DefaultTagger
     This function takes a list of words and returns a lemmatized string of words.

    Parameters
    ----------
    words: list of words
    t4: ngram tagger

    Returns
    -------
    lemmatized_string: lemmatized string of the list of words
    '''

    wnlt = WordNetLemmatizer()
    #Change words to lower case
    words=convert_list_to_lower(words)

    #Deploy the POS tagging chain
    tagged_words=t4.tag(words)

    #Create lemmatize string based on ngrams
    lemmatized_string=''
    for (w,t) in tagged_words:
        lemmatized_string+=' '+wnlt.lemmatize(w,pos=get_wordnet_pos(t))
    return lemmatized_string


def ngram_pos_tagger():
    '''This function trains ngram tagger based on the brown corpus and returns it trainned

    Parameters
    ----------


    Returns
    -------
    t4:pos tagger based on up to 4-grams and using DefaultTagger for not determined words
    using method t4.tag(list_words) with determine the POS-tag of the words in the list.
    '''

    default_tagger=DefaultTagger('NOUN')
    t1=UnigramTagger(train=train_for_n_grams(),backoff=default_tagger)
    t2=NgramTagger(n=2,train=train_for_n_grams(),backoff=t1)
    t3=NgramTagger(n=3,train=train_for_n_grams(),backoff=t2)
    t4=NgramTagger(n=4,train=train_for_n_grams(),backoff=t3)

    return t4


def train_for_n_grams():
    '''This function just returns the whole brown corpus with all words and
    tags to serve as the trainning for the n-grams
    Parameters
    ----------


    Returns
    -------
    :brown corpus tagged ready to be used for trainning n-grams'''

    train_for_n_grams = brown.tagged_sents(tagset='universal')
    return train_for_n_grams


def del_special_char(df, label=[1, 2], add_char = False):
    '''This function returns a dataframe with two new columns for "text" and "label"
    with special characters eliminated so it's performance is evaluated later.
    Parameters
    ----------
    df:pandas dataframe
    label:number of the column that needs to be corrected
    add_char:True replaces special characters with the word SPECIAL_CHAR. False
            eliminates the special characters

    Returns
    -------
    df:dataframe with two new columns for label and text without special characters'''

    for column in label:
        # Every loop in i is a new row in the dataframe
        for i in range(df.shape[0]):
            # tokenize each row
            text_list = word_tokenize(str(df.iloc[i, column]), language='english')

            if column == label[0]:
                stamp = 'title_no_special_char'
            elif column == label[1]:
                stamp = 'text_no_special_char'

            df.at[i, stamp] = str_special_char(text_list, add_char)
            text_list = []

    return df


def str_special_char(words, add_char):
    '''This function removes all special characters from an input list if they are not letters
    or numbers.

    Parameters
    ----------
    words: list of tokenized words
    add_char: boolean parameter from del_special_char

    Returns
    -------
    new_string: string with only words without special characters
    '''

    from string import punctuation
    import re

    new_string = ''

    for word in words:

        if (word in set(punctuation)) and add_char:
            new_string+=' '+'SPECIAL_CHAR'

        reg_exp = re.sub('[^A-Za-z0-9]+', '', word)
        if len(reg_exp) != 0:
            new_string += ' ' + reg_exp
        else:
            new_string += reg_exp

    return new_string



def eliminate_over_30(df,label=[4,5]):
    '''This function iterates over all rows of the dataframe and only the columns
    stated in the input parameter label. In each iteration takes the text of
    the cell and eliminate those words with a length greater than 30 characters.

    Parameters
    ----------
    df: pandas dataframe
    label: number of the column to iterate through

    Returns
    -------
    df: pandas dataframe without words longer than 30 characters
    '''

    for column in label:
        # Every loop in i is a new row in the dataframe
        for i in range(df.shape[0]):
            # tokenize each row
            text_list = word_tokenize(str(df.iloc[i, column]), language='english')


            if column == label[0]:
                stamp = 'title_eliminate_over30'
            elif column == label[1]:
                stamp = 'text_eliminate_over30'

            df.at[i, stamp] =str_eliminate_over_30(text_list)
            text_list = []

    return df

def str_eliminate_over_30(words):

    '''This function takes a list of words and remove those with length greater than 30
    Parameters
    ----------
    words: list of words

    Returns
    -------
    new_string: string without 30 characters length words
    '''

    new_string = ''

    for word in words:

        if len(word)<30:
            new_string+=' '+word

    return new_string

##############################################################################
####################             CLASSIFIERS              ####################
##############################################################################

#Transform text into numeric features: CuntVectorizers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.svm import SVC

#Score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def classifiers(count_train,count_test,tfidf_train,tfidf_test,y_train,y_test):
    '''This function takes a TF and TfIdf and shows the score results for
    several classifiers.

        Parameters
    ----------
    count_train: trainned TfCountVectorizer for training features
    count_test: TfCountVectorizer applied over the test set
    tfidf_train: trainned TfidfCountVectorizer for trainning features
    tfidf_test: TfIdfCountVectorizer applied over the test set
    y_train: Series object with labels 'FAKE','REAL' from train set
    y_test:  Series object with labels 'FAKE','REAL' from test set

    Returns
    -------
    :score results for each model.
     Last line provides the best model with its accuracy
    '''

    #Classifiers definition
    naive_clf= MultinomialNB()
    passive_clf=PassiveAggressiveClassifier(n_iter=50,random_state=198374)
    max_clf=LogisticRegression(random_state=198374)
    svc_clf = SVC(random_state=198374)

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

##############################################################################
####################         CONFUSION MATRIX PLOT        ####################
##############################################################################


#Plotting libraries
import matplotlib.pyplot as plt
#import seaborn as sns


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
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
    plt.show()
    plt.close()


##############################################################################
####################        PYTHON SIMPLE FUNCTIONS       ####################
##############################################################################
def missing_values_table(df):
    '''
    This function shows the columns that have missing values and the percentage that those
    missing values represent within the column

    Parameters
    ----------
    df: pandas dataframe

    Returns
    -------
    :mis_val_table_ren_columns: summary table
    '''
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
	columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
		mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
		'% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
		"There are " + str(mis_val_table_ren_columns.shape[0]) +
			" columns that have missing values.")
    return mis_val_table_ren_columns


def convert_list_to_lower(list_words):
    '''Function that turns all the strings of a list into lower case
    Parameters
    ----------
    list_words: a list of words

    Returns
    -------
    :same list but transformed into lower case'''

    return [word.lower() for word in list_words]


def calculate_target_proportion(df, column='label'):
    '''This function takes column of a dataframe and shows all different values
    in that column, the number of instances and the proportion
    Parameters
    ----------
    df: pandas dataframe
    column: name of the column to be checked

    Returns
    -------
    :summary message
    '''
    classes=list(set(df.loc[:,column]))
    n_classes=len(classes)
    total=df[column].shape[0]

    for (element,i) in zip(classes,range(n_classes)):
        amount=df.loc[:,column].value_counts()[i]
        proportion=(amount/total)*100

        print(f'{element}:  {amount:d} instances    {proportion:.1f}%')


def print_csv(df,col,path='./Sebas&Breo_Predictions.csv'):
    '''This function takes a dataframe and adds an array and generates a CSV file
    in the assigned location.

    Parameters
    ----------
    df: pandas dataframe
    col: array to add as a column to df
    path: location where the csv file is created

    Returns
    -------
    result: the concatenation of df and col as a pandas dataframe
    :csv file containing result
    '''

    result=df.copy(deep=True)
    result['prediction']=col

    result.to_csv(path_or_buf=path)
    return result
