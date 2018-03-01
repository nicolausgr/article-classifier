#!/usr/bin/python

#
# CLUSTERING
#
# Nikos Grivas
# M1480
#
# National & Kapodistrian University of Athens
# Department of Informatics & Telecommunications
# Mining Techniques for Big Data
#
# Dependencies:
# pip install numpy
# pip install scipy
# pip install -U scikit-learn
# pip install --upgrade gensim
#

from __future__ import division
from __future__ import print_function

import sys
from time import time
import random
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import gensim


#
# Read and store train data
#
print("Reading and storing data of train_set.csv ...", end="")
sys.stdout.flush()
t0 = time()

train_file = open("train_set.csv", "r")
lines = train_file.readlines()
# Remove 1st line (column titles)
lines = lines[1:]

traindatalist = []
categories = {}

for line in lines:
	# Ignore if line without data
	if len(line) < 5:
		continue
	row = line.split('\t')
	# Strip items from leading and trailing whitespaces
	for i in range(len(row)):
		row[i] = row[i].strip()
	# Strip content from leading and trailing quotes
	if len(row[3]) > 0 and row[3][0] == '\"':
		row[3] = row[3][1:]
	if len(row[3]) > 0 and row[3][-1] == '\"':
		row[3] = row[3][:-1]
	if row[4] in categories:
		categories[row[4]] += 1
	else:
		categories[row[4]] = 1
	# Append row of data in the datalist
	traindatalist.append({"id":row[1],
	                      "title":row[2],
	                      "content":row[3],
	                      "category":row[4]})

print(" done in %0.3fs\n" % (time() - t0))

#
# Read and store test data
#
print("Reading and storing data of test_set.csv ...", end="")
sys.stdout.flush()
t0 = time()

test_file = open("test_set.csv", "r")
lines = test_file.readlines()
# Remove 1st line (column titles)
lines = lines[1:]

testdatalist = []

for line in lines:
	# Ignore if line without data
	if len(line) < 5:
		continue
	row = line.split('\t')
	# Strip items from leading and trailing whitespaces
	for i in range(len(row)):
		row[i] = row[i].strip()
	# Strip content from leading and trailing quotes
	if len(row[3]) > 0 and row[3][0] == '\"':
		row[3] = row[3][1:]
	if len(row[3]) > 0 and row[3][-1] == '\"':
		row[3] = row[3][:-1]
	# Append row of data in the datalist
	testdatalist.append({"id":row[1],
	                     "title":row[2],
	                     "content":row[3]})

print(" done in %0.3fs\n" % (time() - t0))

print("Train data: #" + str(len(traindatalist)))
print("Test data:  #" + str(len(testdatalist)))
print()

train_texts = []
train_labels = []
test_texts = []
test_ids = []

for row in traindatalist:
	train_texts.append(row["title"] + " " + row["content"])
	train_labels.append(row["category"])
train_labels_array = np.array(train_labels)

for row in testdatalist:
	test_texts.append(row["title"] + " " + row["content"])
	test_ids.append(row["id"])


#
# Bag of Words
#
print("Creating the Bag of Words...", end="")
sys.stdout.flush()
t0 = time()
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5)
train_bow_matrix = vectorizer.fit_transform(train_texts)
test_bow_matrix = vectorizer.transform(test_texts)
print(" done in %0.3f sec\n" % (time() - t0))

print("BoW train matrix size: " + str(train_bow_matrix.shape))
print("BoW test matrix size:  " + str(test_bow_matrix.shape) + "\n")

#
# SVD
#
print("Applying SVD...", end="")
sys.stdout.flush()
t0 = time()
svd = TruncatedSVD(n_components=500)
train_svd_matrix = svd.fit_transform(train_bow_matrix)
test_svd_matrix = svd.transform(test_bow_matrix)
print(" done in %0.3f sec\n" % (time() - t0))

print("SVD train matrix size: " + str(train_svd_matrix.shape))
print("SVD test matrix size:  " + str(test_svd_matrix.shape) + "\n")

#
# word2vec
#
print("Applying word2vec...", end="")
sys.stdout.flush()
t0 = time()
labeled_docs = []
for i in range(len(train_texts)):
	wordlist = re.sub("[^\w]", " ", train_texts[i].lower()).split()
	labeled_docs.append(gensim.models.doc2vec.TaggedDocument(wordlist,['Sentence'+str(i)]))

word2vec = gensim.models.Doc2Vec(labeled_docs, size=300, workers=4)
train_w2v_matrix = []
#test_w2v_matrix = []
for i in range(len(labeled_docs)):
	train_w2v_matrix.append(word2vec.infer_vector(labeled_docs[i][0]))
train_w2v_matrix = np.array(train_w2v_matrix)
#for i in range(len(test_texts)):
#	test_w2v_matrix.append(word2vec.infer_vector(re.sub("[^\w]", " ", test_texts[i].lower()).split()))
#	test_w2v_matrix = np.array(test_w2v_matrix)
print(" done in %0.3f sec\n" % (time() - t0))

print("Word2Vec train matrix size: " + str(train_w2v_matrix.shape))
#print("Word2Vec test matrix size: " + str(test_w2v_matrix.shape))

#
# Metrics
#
metrics = ['Accuracy', 'Precision', 'Recall', 'F-Measure']

#
# KFold cross-validation
#
kf = KFold(n_splits=10, shuffle=True)

#
# SVM (BoW)
#
print("Training, predicting and cross-validating with SVM (BoW)...", end="")
sys.stdout.flush()
t0 = time()

svm_bow = {}
for metric in metrics:
	svm_bow[metric] = 0
	
for train_i, test_i in kf.split(train_bow_matrix):
	X_train, X_test = train_bow_matrix[train_i], train_bow_matrix[test_i]
	y_train, y_test = train_labels_array[train_i], train_labels_array[test_i]

	clf = svm.SVC()
	clf.fit(X_train, y_train)
	
	y_pred = clf.predict(X_test)
	svm_bow['Accuracy'] += accuracy_score(y_test, y_pred)
	svm_bow['Precision'] += precision_score(y_test, y_pred, average='weighted')
	svm_bow['Recall'] += recall_score(y_test, y_pred, average='weighted')
	svm_bow['F-Measure'] += f1_score(y_test, y_pred, average='weighted')

print(" done in %0.3f sec" % (time() - t0))

for metric in metrics:
	svm_bow[metric] = svm_bow[metric]/10.0
	print(metric + ": %.3f" % svm_bow[metric])

#
# Random Forest (BoW)
#
print("Training, predicting and cross-validating with Random Forest (BoW)...", end="")
sys.stdout.flush()
t0 = time()

rf_bow = {}
for metric in metrics:
	rf_bow[metric] = 0
	
for train_i, test_i in kf.split(train_bow_matrix):
	X_train, X_test = train_bow_matrix[train_i], train_bow_matrix[test_i]
	y_train, y_test = train_labels_array[train_i], train_labels_array[test_i]

	clf = RandomForestClassifier()
	clf.fit(X_train, y_train)
	
	y_pred = clf.predict(X_test)
	rf_bow['Accuracy'] += accuracy_score(y_test, y_pred)
	rf_bow['Precision'] += precision_score(y_test, y_pred, average='weighted')
	rf_bow['Recall'] += recall_score(y_test, y_pred, average='weighted')
	rf_bow['F-Measure'] += f1_score(y_test, y_pred, average='weighted')

print(" done in %0.3f sec" % (time() - t0))

for metric in metrics:
	rf_bow[metric] = rf_bow[metric]/10.0
	print(metric + ": %.3f" % rf_bow[metric])

#
# SVM (SVD)
#
print("Training, predicting and cross-validating with SVM (SVD)...", end="")
sys.stdout.flush()
t0 = time()

svm_svd = {}
for metric in metrics:
	svm_svd[metric] = 0
	
for train_i, test_i in kf.split(train_svd_matrix):
	X_train, X_test = train_svd_matrix[train_i], train_svd_matrix[test_i]
	y_train, y_test = train_labels_array[train_i], train_labels_array[test_i]

	clf = svm.SVC()
	clf.fit(X_train, y_train)
	
	y_pred = clf.predict(X_test)
	svm_svd['Accuracy'] += accuracy_score(y_test, y_pred)
	svm_svd['Precision'] += precision_score(y_test, y_pred, average='weighted')
	svm_svd['Recall'] += recall_score(y_test, y_pred, average='weighted')
	svm_svd['F-Measure'] += f1_score(y_test, y_pred, average='weighted')

print(" done in %0.3f sec" % (time() - t0))

for metric in metrics:
	svm_svd[metric] = svm_svd[metric]/10.0
	print(metric + ": %.3f" % svm_svd[metric])

#
# Random Forest (SVD)
#
print("Training, predicting and cross-validating with Random Forest (SVD)...", end="")
sys.stdout.flush()
t0 = time()

rf_svd = {}
for metric in metrics:
	rf_svd[metric] = 0
	
for train_i, test_i in kf.split(train_svd_matrix):
	X_train, X_test = train_svd_matrix[train_i], train_svd_matrix[test_i]
	y_train, y_test = train_labels_array[train_i], train_labels_array[test_i]

	clf = RandomForestClassifier()
	clf.fit(X_train, y_train)
	
	y_pred = clf.predict(X_test)
	rf_svd['Accuracy'] += accuracy_score(y_test, y_pred)
	rf_svd['Precision'] += precision_score(y_test, y_pred, average='weighted')
	rf_svd['Recall'] += recall_score(y_test, y_pred, average='weighted')
	rf_svd['F-Measure'] += f1_score(y_test, y_pred, average='weighted')

print(" done in %0.3f sec" % (time() - t0))

for metric in metrics:
	rf_svd[metric] = rf_svd[metric]/10.0
	print(metric + ": %.3f" % svm_svd[metric])

#
# SVM (W2V)
#
print("Training, predicting and cross-validating with SVM (W2V)...", end="")
sys.stdout.flush()
t0 = time()

svm_w2v = {}
for metric in metrics:
	svm_w2v[metric] = 0
	
for train_i, test_i in kf.split(train_w2v_matrix):
	X_train, X_test = train_w2v_matrix[train_i], train_w2v_matrix[test_i]
	y_train, y_test = train_labels_array[train_i], train_labels_array[test_i]

	clf = svm.SVC()
	clf.fit(X_train, y_train)
	
	y_pred = clf.predict(X_test)
	svm_w2v['Accuracy'] += accuracy_score(y_test, y_pred)
	svm_w2v['Precision'] += precision_score(y_test, y_pred, average='weighted')
	svm_w2v['Recall'] += recall_score(y_test, y_pred, average='weighted')
	svm_w2v['F-Measure'] += f1_score(y_test, y_pred, average='weighted')

print(" done in %0.3f sec" % (time() - t0))

for metric in metrics:
	svm_w2v[metric] = svm_w2v[metric]/10.0
	print(metric + ": %.3f" % svm_w2v[metric])

#
# Random Forest (w2v)
#
print("Training, predicting and cross-validating with Random Forest (W2V)...", end="")
sys.stdout.flush()
t0 = time()

rf_w2v = {}
for metric in metrics:
	rf_w2v[metric] = 0
	
for train_i, test_i in kf.split(train_w2v_matrix):
	X_train, X_test = train_w2v_matrix[train_i], train_w2v_matrix[test_i]
	y_train, y_test = train_labels_array[train_i], train_labels_array[test_i]

	clf = RandomForestClassifier()
	clf.fit(X_train, y_train)
	
	y_pred = clf.predict(X_test)
	rf_w2v['Accuracy'] += accuracy_score(y_test, y_pred)
	rf_w2v['Precision'] += precision_score(y_test, y_pred, average='weighted')
	rf_w2v['Recall'] += recall_score(y_test, y_pred, average='weighted')
	rf_w2v['F-Measure'] += f1_score(y_test, y_pred, average='weighted')

print(" done in %0.3f sec" % (time() - t0))

for metric in metrics:
	rf_w2v[metric] = rf_w2v[metric]/10.0
	print(metric + ": %.3f" % rf_w2v[metric])

#
# My Method
#
print("Training, predicting and cross-validating with My Method...", end="")
sys.stdout.flush()
t0 = time()

my_method = {}
for metric in metrics:
	my_method[metric] = 0
	
for train_i, test_i in kf.split(train_svd_matrix):
	X_train, X_test = train_svd_matrix[train_i], train_svd_matrix[test_i]
	y_train, y_test = train_labels_array[train_i], train_labels_array[test_i]

	clf = RandomForestClassifier(n_estimators=50)
	clf.fit(X_train, y_train)
	
	y_pred = clf.predict(X_test)
	my_method['Accuracy'] += accuracy_score(y_test, y_pred)
	my_method['Precision'] += precision_score(y_test, y_pred, average='weighted')
	my_method['Recall'] += recall_score(y_test, y_pred, average='weighted')
	my_method['F-Measure'] += f1_score(y_test, y_pred, average='weighted')

print(" done in %0.3f sec" % (time() - t0))

for metric in metrics:
	my_method[metric] = my_method[metric]/10.0
	print(metric + ": %.3f" % my_method[metric])

print()


#
# Write metric data to csv file
#
print("Writing csv train data metrics file (classification_output/EvaluationMetric_10fold.csv)...\n")

csv_file = open("classification_output/EvaluationMetric_10fold.csv", "w")

first_line = "Statistic Measure\tSVM (BoW)\tRandom Forest (BoW)\tSVM (SVD)\tRandom Forest (SVD)\tSVM (W2V)\tRandom Forest (W2V)\tMy Method"
print(first_line)
csv_file.write(first_line + "\n")

for metric in metrics:
	curr_line = metric + "\t" + str(svm_bow[metric]) \
	                   + "\t" + str(rf_bow[metric]) \
	                   + "\t" + str(svm_svd[metric]) \
	                   + "\t" + str(rf_svd[metric]) \
	                   + "\t" + str(svm_w2v[metric]) \
	                   + "\t" + str(rf_w2v[metric]) \
	                   + "\t" + str(my_method[metric])
	print(curr_line)
	csv_file.write(curr_line + "\n")
print()
csv_file.close()


#
# Predict test data
#
print("Training by My Method...", end="")
sys.stdout.flush()
t0 = time()

clf = RandomForestClassifier(n_estimators=50)
clf.fit(train_svd_matrix, train_labels)

print(" done in %0.3f sec" % (time() - t0))

print("Predicting through My Method...", end="")
sys.stdout.flush()
t0 = time()

my_method_pred = clf.predict(test_svd_matrix)

print(" done in %0.3f sec" % (time() - t0))

print()

print("Writing csv prediction file (classification_output/testSet_categories.csv)...\n")

csv_file = open("classification_output/testSet_categories.csv", "w")

first_line = "Test_Document_ID\tPredicted_Category"
csv_file.write(first_line + "\n")

for i in range(len(testdatalist)):
	curr_line = str(testdatalist[i]["id"]) + "\t" + my_method_pred[i]
	csv_file.write(curr_line + "\n")
csv_file.close()
