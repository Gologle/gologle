
from tokenize import group
from typing import Iterator
from pathlib import Path
import numpy

from base import DatasetEntry, DatasetParser

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFpr, chi2

#Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Metrics
from sklearn.metrics import classification_report



class NewsgroupsEntry(DatasetEntry):

    def __init__(self, group_id: int, entry_path: Path):
        try:
            text = entry_path.read_text(errors="ignore")
        except UnicodeDecodeError as e:
            print(entry_path)
            raise e
        end_of_line1 = text.find("\n")
        end_of_line2 = text.find("\n", end_of_line1 + 1)
        line1 = text[:end_of_line1]
        line2 = text[end_of_line1 + 1: end_of_line2]

        super(NewsgroupsEntry, self).__init__(f"{group_id}_{entry_path.name}")

        self.path = entry_path
        self.group = entry_path.parent.name

        if line1.startswith("From: "):
            self.from_ = line1[6:]
            assert line2.startswith("Subject: ")
            self.subject = line2[9:]
        elif line1.startswith("Subject: "):
            self.subject = line1[9:]
            assert line2.startswith("From: ")
            self.from_ = line2[6:]
        else:
            assert False, f"From/Subject not found in {entry_path}"

        self.text = text[end_of_line2:].strip()

    @property
    def raw_text(self):
        return self.path.read_text(errors="ignore")


class NewsgroupsParser(DatasetParser):
    """Parser for the 20 Newsgroups dataset"""

    def __init__(self):
        super(NewsgroupsParser, self).__init__(
            data=self.root / "20newsgroups-18828",
            count_vzer=CountVectorizer(
                input="filename",
                decode_error="ignore",
                stop_words="english"
            ),
            total=18828
        )

        self.entries: list[NewsgroupsEntry] = []

        for group_id, folder in enumerate(self.data.iterdir()):
            for file in folder.iterdir():
                self.entries.append(NewsgroupsEntry(group_id, file))

        assert len(self.entries) == self.total

    def __iter__(self) -> Iterator[NewsgroupsEntry]:
        return iter(self.entries)\

    def fit_transform(self):
        return self.count_vzer.fit_transform(
            tuple(str(entry.path) for entry in self)
        )



def load_data(config={}):
    """
    Load the Newsgroup dataset.
    Returns
    -------
    data : dict
        with keys 'x_train', 'x_test', 'y_train', 'y_test', 'labels', 'vocabulary'
    """

    parser = NewsgroupsParser()
    vectorizer = parser.count_vzer
    docs_paths = [str(entry.path) for entry in parser.__iter__()]
    docs_groups= [str(entry.group) for entry in parser.__iter__()]
    docsTotal = len(docs_groups)
    labels = {}
    labelCount = 0
    for i in range(docsTotal):
        vals = labels.keys()
        group = docs_groups[i] 
        if group not in vals:
            labels[group] = labelCount
            labelCount += 1 
  
    for i in range (docsTotal):
        group = docs_groups[i]
        new_group = [0] * labelCount
        index = labels[group]
        new_group[index] = 1
        docs_groups[i] = new_group 
    
    train_set = {}
    test_set = {}

    for i in range(docsTotal):
        id = int(str(docs_paths[i]).split(sep = "\\")[-1])
        #if id % 4 == 0: test_set[i] = docs_groups[i]
        #else: train_set[i] = docs_groups[i]
        # This is for getting an smaller subset
        if id % 20 == 0: train_set[i] = docs_groups[i]
        elif id % 20 == 1: test_set[i] = docs_groups[i] 

    xs = {'train': [], 'test': []}
    xs['train'] = vectorizer.fit_transform([docs_paths[k] for k in train_set.keys()]).toarray()
    xs['test'] = vectorizer.transform(docs_paths[k] for k in test_set.keys()).toarray()
    ys = {'train': [], 'test': []}
    ys['train'] = [docs_groups[k] for k in train_set.keys()]
    ys['test'] = [docs_groups[k] for k in test_set.keys()]

    data = {'x_train': xs['train'], 'y_train': ys['train'],
            'x_test': xs['test'], 'y_test': ys['test'],
            'labels': labels, 'vocabulary': vectorizer.get_feature_names_out()}
    
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
    filtered_train_docs = selector.fit_transform(train_docs, train_classes)
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


    # Decision Tree Classifier
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