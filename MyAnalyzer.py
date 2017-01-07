#classify text for politics,spam or legitimate email
#opininon/sentiment mining
# 2 choices specifically(e.g spam/no spam)

import nltk
import random #to shuffle the dataset
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    
    def __init__(self,*classifiers):
        self._classifiers = classifiers
        
    def classify(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes) #give the popular vote

    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))# count the number of occurrence of popular vote
        conf = choice_votes / len(votes)
        return conf
    
short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

all_words=[]
##allowed_words=["J"]
##for r in short_pos.split('\n'):
##    words= nltk.word_tokenize(r)
##    tagged = nltk.pos_tag(words)
##    for w in tagged:
##        if w[1][0] in allowed_words:
##            all_words.append(w[0].lower())
##    
##    
##for r in short_neg.split('\n'):
##    words= nltk.word_tokenize(r)
##    tagged = nltk.pos_tag(words)
##    for w in tagged:
##        if w[1][0] in allowed_words:
##            all_words.append(w[0].lower())

allwords =open("alladjectives.pickle","rb")
all_words = pickle.load(allwords)
allwords.close()

documents= []
documentsP = open("docs.pickle","rb")
documents = pickle.load(documentsP)
random.shuffle(documents)
##savedocuments = open("alladjectives.pickle","wb")
##pickle.dump(all_words,savedocuments)
##savedocuments.close()

all_words = nltk.FreqDist(all_words)
##print(all_words.most_common(15))

word_features = list(all_words.keys())[:5000]
openfile=open("wordfeatures.pickle","wb")
pickle.dump(word_features,openfile)
openfile.close()



def find_features(document):
    words = nltk.word_tokenize(document) #no repetition included(get one iteration)
    features = {} # empty dictionary
    for w in word_features:
        features[w] = (w in words) #boolean value if w is in words or not
    return features

featuresets=[(find_features(rev),category)for (rev,category) in documents]
random.shuffle(featuresets)

featurepickle = open("featuresets.pickle","wb")
pickle.dump(featuresets[:4000],featurepickle)
featurepickle.close()

featurepickle = open("featuresets(1).pickle","wb")
pickle.dump(featuresets[4000:8000],featurepickle)
featurepickle.close
featurepickle = open("featuresets(2).pickle","wb")
pickle.dump(featuresets[8000:],featurepickle)
featurepickle.close()
'''
code in '##' rpresents the pickle files when our model got trained
after running this code, uncomment the below code and comment out the above 9 line of code atrting from
featurepickle = open("featuresets.pickle","wb")
'''

##open_file = open("featuresets.pickle","rb")
##featuresets1=pickle.load(open_file)
##open_file.close()
##
##open_file = open("featuresets(1).pickle","rb")
##featuresets2=(pickle.load(open_file))
##open_file.close()
##
##open_file = open("featuresets(2).pickle","rb")
##featuresets3=pickle.load(open_file)
##open_file.close()
##
##featuresets=featuresets1
##for w in featuresets2:
##    featuresets.append(w)
##for w in featuresets3:
##    featuresets.append(w)


training_set = featuresets[9000:]

#naivebayes algorithm (scalable)

# posterior = prior occurrences X likelihood / evidence




classifier_f = open("naivebayes.pickle","rb")
classifier=pickle.load(classifier_f)
classifier_f.close()

##print("Original Naive Bayes Algo accuracy: ",(nltk.classify.accuracy(classifier,testing_set))*100)
#classifier.show_most_informative_features(15)

#pickles
##classifier = nltk.NaiveBayesClassifier.train(training_set)
##save_classifier = open("naivebayes.pickle","wb")
##pickle.dump(classifier,save_classifier)
##save_classifier.close()

##MNB_classifier = SklearnClassifier(MultinomialNB())
###MNB_classifier.show_most_informative_features(15)
##MNB_classifier.train(training_set)
##save_classifier = open("MNB.pickle","wb")
##pickle.dump(MNB_classifier,save_classifier)
##save_classifier.close()

openfile = open("MNB.pickle","rb")
MNB_classifier = pickle.load(openfile)
openfile.close()


classifier_f3 = open("BernoulliNB.pickle","rb")
BernoulliNB_classifier = pickle.load(classifier_f3)
classifier_f3.close()
##
##BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
##BernoulliNB_classifier.train(training_set)
##print("BernoulliNB_classifier accuracy: ",(nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

##save_classifier2 = open("BernoulliNB.pickle","wb")
##pickle.dump(BernoulliNB_classifier,save_classifier2)
##save_classifier2.close()

##LogisticRegression_classifier = SklearnClassifier(LogisticRegression())


classifier_f4 = open("LogisticRegression.pickle","rb")
LogisticRegression_classifier = pickle.load(classifier_f4)
classifier_f4.close()
##LogisticRegression_classifier.train(training_set)
##print("LogisticRegression_classifier accuracy: ",(nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)
##save_classifier = open("LogisticRegression.pickle","wb")
##pickle.dump(LogisticRegression_classifier,save_classifier)
##save_classifier.close()

classifier_f5 = open("SGDClassifier.pickle","rb")
SGDclassifier = pickle.load(classifier_f5)
classifier_f5.close()

##SGDclassifier = SklearnClassifier(SGDClassifier())
##SGDclassifier.train(training_set)
##print("SGDclassifier accuracy: ",(nltk.classify.accuracy(SGDclassifier,testing_set))*100)

##save_classifier2 = open("SGDClassifier.pickle","wb")
##pickle.dump(SGDclassifier,save_classifier2)
##save_classifier2.close()


####classifier_f6 = open("SVC.pickle","rb")
####SVC_classifier = pickle.load(classifier_f6)
####classifier_f6.close()
####
#####SVC_classifier = SklearnClassifier(SVC())
####SVC_classifier.train(training_set)
####print("SVC_classifier accuracy: ",(nltk.classify.accuracy(SVC_classifier,testing_set))*100)


##save_classifier3 = open("SVC.pickle","wb")
##pickle.dump(SVC_classifier,save_classifier3)
##save_classifier3.close()


classifier_f7 = open("LinearSVC.pickle","rb")
LinearSVC_classifier = pickle.load(classifier_f7)
classifier_f7.close()
##LinearSVC_classifier = SklearnClassifier(LinearSVC())
##LinearSVC_classifier.train(training_set)
##print("LinearSVC_classifier accuracy: ",(nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)
##save_classifier4 = open("LinearSVC.pickle","wb")
##pickle.dump(LinearSVC_classifier,save_classifier4)
##save_classifier4.close()

classifier_f8 = open("NuSVC.pickle","rb")
NuSVC_classifier = pickle.load(classifier_f8)
classifier_f8.close()
##NuSVC_classifier = SklearnClassifier(NuSVC())
##NuSVC_classifier.train(training_set)
##print("NuSVC_classifier accuracy: ",(nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)

##save_classifier5 = open("NuSVC.pickle","wb")
##pickle.dump(NuSVC_classifier,save_classifier5)
##save_classifier5.close()


#voting procedure
voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDclassifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

##print("Voted Classifier accuracy: ",(nltk.classify.accuracy(voted_classifier,testing_set))*100)

##print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %: ", voted_classifier.confidence(testing_set[0][0]))
##print("Classification: ", voted_classifier.classify(testing_set[1][0]), "Confidence %: ", voted_classifier.confidence(testing_set[1][0]))
##print("Classification: ", voted_classifier.classify(testing_set[2][0]), "Confidence %: ", voted_classifier.confidence(testing_set[2][0]))
##print("Classification: ", voted_classifier.classify(testing_set[3][0]), "Confidence %: ", voted_classifier.confidence(testing_set[3][0]))
##print("Classification: ", voted_classifier.classify(testing_set[4][0]), "Confidence %: ", voted_classifier.confidence(testing_set[4][0]))
##print("Classification: ", voted_classifier.classify(testing_set[5][0]), "Confidence %: ", voted_classifier.confidence(testing_set[5][0]))

def sentiment(text):
    feats = find_features(text.lower())
    print(voted_classifier.confidence(feats)*100,"%")
    return voted_classifier.classify(feats)

print(sentiment('''the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'''))




























































        
