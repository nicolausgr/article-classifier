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
#

from __future__ import division
from __future__ import print_function

import sys
from time import time
import re
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.decomposition import TruncatedSVD
import gensim

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

CLUSTERS = 5

print("Reading and storing data of train_set.csv ...", end="")
sys.stdout.flush()
t0 = time()

train_file = open("train_set.csv", "r")
lines = train_file.readlines()
# Remove 1st line (column titles)
lines = lines[1:]

datalist = []
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
	datalist.append({"id":row[1],
	                 "title":row[2],
	                 "content":row[3],
	                 "category":row[4]})

textlist = []
for article in datalist:
	textlist.append(article["title"] + " " + article["content"])

print(" done in %0.3fs\n" % (time() - t0))

for category in categories:
	print(category + ": " + str(categories[category]))
print()


#
# TfIDf
#
#print("Creating the Bag of Words...", end="")
#sys.stdout.flush()
#t0 = time()
#tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=1.5/len(textlist), max_df=0.5, binary=True)
#tfidf_matrix = tfidf_vectorizer.fit_transform(textlist)
#print(" done in %0.3f sec\n" % (time() - t0))
#
# SVD
#
#print("Applying SVD...", end="")
#sys.stdout.flush()
#t0 = time()
#svd = TruncatedSVD(n_components=500)
#svd_matrix = svd.fit_transform(tfidf_matrix)
#print(" done in %0.3f sec\n" % (time() - t0))


#
# word2vec
#
print("Applying word2vec ...", end="")
sys.stdout.flush()
t0 = time()

labeled_docs = []
for i in range(len(textlist)):
	wordlist = re.sub("[^\w]", " ", textlist[i].lower()).split()
	labeled_docs.append(gensim.models.doc2vec.TaggedDocument(wordlist,['Sentence'+str(i)]))

word2vec = gensim.models.Doc2Vec(labeled_docs, size=300)
w2v_matrix = []
for i in range(len(labeled_docs)):
	w2v_matrix.append(word2vec.infer_vector(labeled_docs[i][0]))

print(" done in %0.3f sec\n" % (time() - t0))


#
# K-means functions definition
#
def find_clusters(matrix, centers):
	clusters = []
	for i in range(len(matrix)):
		point = matrix[i]
		cluster = 0
		min_distance = np.linalg.norm(point-centers[0])
		for j in range(1, len(centers)):
			distance = np.linalg.norm(point-centers[j])
			if distance < min_distance:
				min_distance = distance
				cluster = j
		clusters.append(cluster)
	return clusters

def find_centers(matrix, clusters, n):
	cluster_points = []
	for i in range(n):
		cluster_points.append([])
	for i in range(len(matrix)):
		cluster_points[clusters[i]].append(matrix[i])
	centers = []
	for i in range(n):
		centers.append(np.mean(cluster_points[i], axis = 0))
	return centers

def unchanged_centers(newc, oldc):
	newset = set([tuple(point) for point in newc])
	oldset = set([tuple(point) for point in oldc])
	return newset == oldset

def KMeans(matrix, nclusters):
	oldcenters = []
	newcenters = []
	for i in np.random.randint(len(matrix), size=nclusters):
		newcenters.append(matrix[i])
	while not unchanged_centers(newcenters, oldcenters):
		oldcenters = newcenters
		clusters = find_clusters(matrix, oldcenters)
		newcenters = find_centers(matrix, clusters, nclusters)
	return clusters


#kmeans = KMeans(n_clusters=CLUSTERS, init='random', n_init=5)
#kmeans.fit(d2v_matrix)
#kmeans_labels = kmeans.labels_

#
# Repeat clustering until at least 83% of each cluster contains articles of a single category
#
while True:
	print("Clustering ...", end="")
	sys.stdout.flush()
	t0 = time()
	
	kmeans_labels = KMeans(w2v_matrix, CLUSTERS)

	print(" done in %0.3fs" % (time() - t0))

	clusters_table = []
	clusters_counter = []
	for i in range(CLUSTERS):
		cat_counter = {}
		for category in categories:
			cat_counter[category] = 0
		clusters_table.append(cat_counter)
		clusters_counter.append(0)

	for i in range(len(datalist)):
		this_dict = clusters_table[kmeans_labels[i]]
		this_article = datalist[i]
		this_dict[this_article["category"]] += 1
		clusters_counter[kmeans_labels[i]] += 1

	for i in clusters_table:
		print(i)
	print()
	
	cluster_results = []
	i = 0
	all_clusters_ok = True
	for cluster in clusters_table:
		this_cluster_results = []
		cluster_ok = False
		for cat in sorted(cluster.keys()):
			percentage = cluster[cat]/clusters_counter[i]
			if percentage >= 0.83:
				cluster_ok = True
			this_cluster_results.append(percentage)
		cluster_results.append(this_cluster_results)
		if not cluster_ok:
			all_clusters_ok = False
		i += 1
	
	if all_clusters_ok:
		break

#
# Write cluster data to csv file
#
print("Writing csv file (clustering_output/clustering_KMeans.csv) ...\n")

csv_file = open("clustering_output/clustering_KMeans.csv", "w")

first_line = ""
for cat in sorted(categories.keys()):
	first_line += "\t" + cat
print(first_line)
csv_file.write(first_line + "\n")
i = 0
for this_cluster_results in cluster_results:
	cluster_line = "Cluster" + str(i+1)
	for percentage in this_cluster_results:
		cluster_line += "\t" + str("{:.2f}".format(percentage))
	print(cluster_line)
	csv_file.write(cluster_line + "\n")
	i += 1
print()
csv_file.close()


#
# Create cluster data visualization using matplotlib's pcolor
#
print("Creating cluster visualization (clustering_output/clustering_KMeans.png) ...", end="")
sys.stdout.flush()
t0 = time()

column_labels = sorted(categories.keys())
row_labels = []
for i in range(CLUSTERS):
	row_labels.append("Cluster" + str(i+1))

data = np.array(cluster_results)
fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

# Table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xticklabels(column_labels, minor=False)
ax.set_yticklabels(row_labels, minor=False)

# Save figure to a png file
fig = ax.get_figure()
fig.savefig("clustering_output/clustering_KMeans.png")

print(" done in %0.3fs" % (time() - t0))
