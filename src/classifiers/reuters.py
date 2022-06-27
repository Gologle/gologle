
"""

"""

#IA-SRI
from nltk.corpus import reuters, stopwords, wordnet2021
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_selection import SelectFpr, chi2

#Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Metrics
from sklearn.metrics import classification_report

#Others
import numpy

n_classes = 90
labels = reuters.categories()
size_limit = 1000

        
def load_data(config={}):
    """
    Load the Reuters dataset.
    Returns
    -------
    data : dict
        with keys 'x_train', 'x_test', 'y_train', 'y_test', 'labels', 'vocabulary'
    """
    stop_words = stopwords.words("english")
    vectorizer = CountVectorizer(stop_words=stop_words, binary = False)
    mlb = MultiLabelBinarizer()

    documents = reuters.fileids()
    test = [d for d in documents if d.startswith('test/')]
    train = [d for d in documents if d.startswith('training/')]

    # LIMIT #
    test = test[:size_limit]
    train = train[:size_limit] 
    #### LIMIT ####

    docs = {}
    docs['train'] = [reuters.raw(doc_id) for doc_id in train]
    docs['test'] = [reuters.raw(doc_id) for doc_id in test]
    xs = {'train': [], 'test': []}
    xs['train'] = vectorizer.fit_transform(docs['train']).toarray()
    xs['test'] = vectorizer.transform(docs['test']).toarray()
    ys = {'train': [], 'test': []}
    ys['train'] = mlb.fit_transform([reuters.categories(doc_id)
                                     for doc_id in train])
    ys['test'] = mlb.transform([reuters.categories(doc_id)
                                for doc_id in test])
    data = {'x_train': xs['train'], 'y_train': ys['train'],
            'x_test': xs['test'], 'y_test': ys['test'],
            'labels': globals()["labels"] ,'vocabulary': vectorizer.get_feature_names_out()}
    
    return data



def MultiClassPredictor (training_docs, target_docs, training_classes, method = MultinomialNB):
    
    Classifier = method(max_iter = 1000) if method == LogisticRegression else method()
    classesTotal = len(training_classes[0])
    predicted_classification = [[] for _ in range (len(target_docs))]
    
    for i in range (classesTotal):
        classes = [] 
        for c in training_classes:
            classes.append(c[i])
        Classifier.fit(training_docs, classes)
        p_f = Classifier.predict(target_docs)
        for j in range (len(p_f)):
            predicted_classification[j].append(p_f[j])
    return predicted_classification



if __name__ == '__main__':
    config = {}
    data = load_data(config)

    train_docs = data['x_train']
    train_classes = data['y_train']
    test_docs = data['x_test']
    test_classes = data['y_test']
    vocabulary = data['vocabulary']
    labels = data['labels']

    selector = SelectFpr(chi2, alpha=0.001)
    filtered_train_docs = selector.fit_transform(train_docs, train_classes) #this reduces the number of features from 26147 to 9722
    filtered_test_docs = selector.transform(test_docs)

    training_docs = filtered_train_docs
    target_docs = filtered_test_docs
    training_classes = train_classes
    

    #Multinomial Naive Bayes
    predicted_classes = MultiClassPredictor(training_docs, target_docs, training_classes, MultinomialNB)
    all_classes = numpy.append(train_classes, predicted_classes, axis = 0)
    classReport = classification_report(test_classes,predicted_classes,output_dict=True, zero_division=1) 
    print("\n")
    print("Multinomial Naive Bayes")
    print("(avg : f1-score)")
    print("micro avg: " + str(classReport['micro avg']['f1-score']))  
    print("macro avg: " + str(classReport['macro avg']['f1-score']))  
    print("weighted avg: " + str(classReport['weighted avg']['f1-score']))  
    print("samples avg: " + str(classReport['samples avg']['f1-score']))  


    #Logistic Regression
    predicted_classes = MultiClassPredictor(training_docs, target_docs, training_classes, LogisticRegression)
    all_classes = numpy.append(train_classes, predicted_classes, axis = 0)
    classReport = classification_report(test_classes, predicted_classes, output_dict=True, zero_division=1) 
    print("\n")
    print("Logistic Regression")
    print("(avg : f1-score)")
    print("micro avg: " + str(classReport['micro avg']['f1-score']))  
    print("macro avg: " + str(classReport['macro avg']['f1-score']))  
    print("weighted avg: " + str(classReport['weighted avg']['f1-score']))  
    print("samples avg: " + str(classReport['samples avg']['f1-score']))  


    #K Neighbors Classifier (with k = 5)
    predicted_classes = MultiClassPredictor(training_docs, target_docs, training_classes, KNeighborsClassifier)
    all_classes = numpy.append(train_classes, predicted_classes, axis = 0)
    classReport = classification_report(test_classes, predicted_classes, output_dict=True, zero_division=1) 
    print("\n")
    print("K Neighbors Classifier")
    print("(avg : f1-score)")
    print("micro avg: " + str(classReport['micro avg']['f1-score']))  
    print("macro avg: " + str(classReport['macro avg']['f1-score']))  
    print("weighted avg: " + str(classReport['weighted avg']['f1-score']))  
    print("samples avg: " + str(classReport['samples avg']['f1-score']))  


    # DecisionTree Classifier
    predicted_classes = MultiClassPredictor(training_docs, target_docs, training_classes, DecisionTreeClassifier)
    all_classes = numpy.append(train_classes, predicted_classes, axis = 0)
    classReport = classification_report(test_classes, predicted_classes, output_dict=True, zero_division=1)
    print("\n")
    print("Decision Tree Classifier")
    print("(avg : f1-score)")
    print("micro avg: " + str(classReport['micro avg']['f1-score']))  
    print("macro avg: " + str(classReport['macro avg']['f1-score']))  
    print("weighted avg: " + str(classReport['weighted avg']['f1-score']))  
    print("samples avg: " + str(classReport['samples avg']['f1-score']))  


    # Random Forest Classifier
    predicted_classes = MultiClassPredictor(training_docs, target_docs, training_classes, RandomForestClassifier)
    all_classes = numpy.append(train_classes, predicted_classes, axis = 0)
    classReport = classification_report(test_classes, predicted_classes, output_dict=True, zero_division=1)
    print("\n")
    print("Random Forest Classifier")
    print("(avg : f1-score)")
    print("micro avg: " + str(classReport['micro avg']['f1-score']))  
    print("macro avg: " + str(classReport['macro avg']['f1-score']))  
    print("weighted avg: " + str(classReport['weighted avg']['f1-score']))  
    print("samples avg: " + str(classReport['samples avg']['f1-score']))  
    print("\n")

    print(numpy.array(training_classes).shape)
    print(numpy.array(predicted_classes).shape)