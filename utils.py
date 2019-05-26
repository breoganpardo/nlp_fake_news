import pandas as pd
import numpy as np

#NLTK imports
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import DefaultTagger,UnigramTagger,NgramTagger 
from nltk.corpus import brown
nltk.download('brown')
    
def missing_values_table(df):
    '''
    function which shows the features that have missing values and the percentage
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
    '''Function that turns all the strings of a list into lower case'''
    return [word.lower() for word in list_words]

	
def calculate_target_proportion(df, column='label'):
    classes=list(set(df.loc[:,column]))
    n_classes=len(classes)
    total=df[column].shape[0]
    
    for (element,i) in zip(classes,range(n_classes)):
        amount=df.loc[:,column].value_counts()[i]
        proportion=(amount/total)*100
        
        print(f'{element}:  {amount:d} instances    {proportion:.1f}%')


def get_wordnet_pos(treebank_tag):

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
		
def nltk_tagger(words):
    '''This function takes a list of words and returns the same list transformed intro string and lemmatized'''
    #Change words to lower case
    words=convert_list_to_lower(words)
    wnlt = WordNetLemmatizer()
    #POS tagging the words based on WordNet
    #This point can be improved with N-grams. Next iteration will improve tag.
    tagged_words = nltk.pos_tag(words)
    
    #Create a string with all the words lemmatized
    lemmatized_string=''
    for (w,t) in tagged_words:
        lemmatized_string+=' '+wnlt.lemmatize(w,  pos=get_wordnet_pos(t))
    return lemmatized_string

	
def wordnet_lemmatizer(df, label=[1, 2]):
    '''This function returns a text after being lemmatized to be afterwards inserted in the CountVectorizer functions.
    :input: panda.dataframe
    :input: number of the column that needs to be lemmatized
    :output: dataframe lemmatized'''
    
    for column in label:
        # Every loop in i is a new row in the dataframe
        for i in range(df.shape[0]):
            # tokenize each row
            text_list = word_tokenize(str(df.iloc[i, column]), language='english')
                   

            if column == 1:
                stamp = 'title_wordnet_lemmatized'
            elif column == 2:
                stamp = 'text_wordnet_lemmatized'

            df.at[i, stamp] = nltk_tagger(text_list)
            text_list = []

    return df
	
def train_for_n_grams():
    '''This function just returns the whole brown corpus with all words and tags to serve as the trainning for the n-grams'''


    train_for_n_grams = brown.tagged_sents(tagset='universal')
    return train_for_n_grams
	

def n_gram_POS_tagger(words):
    '''This function performs POS tagging based on concatenated n-grams + DefaultTagger
       This function must take a list of words and return a list of the format: [('word1','tag1'), ('word2','tag2'),...]
    '''
    
    wnlt = WordNetLemmatizer()
    #Change words to lower case
    words=convert_list_to_lower(words)
    
    #Trainning taggers
    default_tagger=DefaultTagger('NOUN')
    t1=UnigramTagger(train=train_for_n_grams(),backoff=default_tagger)
    t2=NgramTagger(n=2,train=train_for_n_grams(),backoff=t1)
    t3=NgramTagger(n=3,train=train_for_n_grams(),backoff=t2)
    t4=NgramTagger(n=4,train=train_for_n_grams(),backoff=t3)
    
    #Deploy the POS tagging chain
    tagged_words=t4.tag(words)
    
    #Create lemmatize string based on ngrams
    for (w,t) in tagged_words:
        lemmatized_string=''
        lemmatized_string+=' '+wnlt.lemmatize(w,pos=get_wordnet_pos(t))
    
    return lemmatized_string


def ngram_lemmatizer(df, label=[1, 2]):
    '''This function returns a text after being lemmatized to be afterwards inserted in the CountVectorizer functions.
    :input: panda.dataframe
    :input: number of the column that needs to be lemmatized
    :output: dataframe lemmatized'''

    
    for column in label:
        # Every loop in i is a new row in the dataframe
        for i in range(df.shape[0]):
            # tokenize each row
            text_list = word_tokenize(str(df.iloc[i, column]), language='english')
                   

            if column == 1:
                stamp = 'title_ngram_lemmatized'
            elif column == 2:
                stamp = 'text_ngram_lemmatized'

            df.at[i, stamp] = n_gram_POS_tagger(text_list)
            text_list = []

    return df