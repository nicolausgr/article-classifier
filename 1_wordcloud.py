#!/usr/bin/python

#
# WORDCLOUD
#
# Nikos Grivas
# M1480
#
# National & Kapodistrian University of Athens
# Department of Informatics & Telecommunications
# Mining Techniques for Big Data
#
# Dependencies:
# pip install wordcloud
# pip install numpy
# pip install -U pip setuptools
# pip install matplotlib
#

from __future__ import print_function

import sys
from time import time
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

print("Reading data from train_set.csv ...", end="")
sys.stdout.flush()
t0 = time()

train_file = open("train_set.csv", "r")
lines = train_file.readlines()
# Remove 1st line (column titles)
lines = lines[1:]

print(" done in %0.3f sec" % (time() - t0))

datalist = []
content_by_category = {}

print("Editing data and sorting content by category ...", end="")
sys.stdout.flush()
t0 = time()

for line in lines:
	# Ignore if empty line
	if len(line) < 5:
		continue
	row = line.split('\t')
	# Strip items from leading and trailing whitespaces
	for i in range(len(row)):
		row[i] = row[i].strip()
	aid = row[1]
	title = row[2]
	content = row[3]
	category = row[4]
	# Strip content from leading and trailing quotes
	if len(content) > 0 and content[0] == '\"':
		content = content[1:]
	if len(content) > 0 and content[-1] == '\"':
		content = content[:-1]
	if category in content_by_category:
		content_by_category[category] += title + " " + content + " "
	else:
		content_by_category[category] = title + " " + content + " "

print(" done in %0.3f sec\n" % (time() - t0))

# Create the stopword list
stopwordlist = ["said","say","says","will","also"]
stopwords = set(STOPWORDS)
for word in stopwordlist:
	stopwords.add(word)

# Use a cloud shape mask for the wordclouds
cloud_mask = np.array(Image.open("cloud.png"))

# Create the wordcloud png image for each one of the categories
for category in content_by_category:
	print("Creating the wordcloud for " + category + " (wordcloud_output/" + category + ".png) ...", end="")
	sys.stdout.flush()
	t0 = time()
	wordcloud = WordCloud(background_color="white", max_words=320, stopwords=stopwords, mask=cloud_mask).generate(content_by_category[category])
	wordcloud.to_file("wordcloud_output/" + category + ".png")
	print(" done in %0.3f sec" % (time() - t0))
print()
