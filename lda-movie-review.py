import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

reviews = pd.read_csv("movie_data.csv", header=None, skiprows=1)

# load_files returns a branch, containing training texts and training labels
# text_train, y_train = reviews_train.data, reviews_train.target
text, y = reviews[0], reviews[1]
print("length of text: {}".format(len(text)))

# It seems that data consists of some line breaks (<br />). Hence we need to clean the data
text_cleaned = [line.replace("<br />", " ") for line in text]

# Let us check whether data is evenly distributed per class
print("Samples per class: {}".format(np.bincount(y)))

# Divide dataset into train/test set
text_train, text_test, y_train, y_test = train_test_split(text_cleaned, y, stratify=y,
                                                          test_size=.49,
                                                          random_state=42)
print("Number of documents in text_train: {}".format(len(text_train)))
print("Number of documents in text_test: {}".format(len(text_test)))

# let is use CountVectorizer which consists of the tokenization of the training data
# and building of the vocabulary
vect = CountVectorizer(max_features=10000, max_df=0.15)
X = vect.fit_transform(text_train)

# Apply LDA
lda = LatentDirichletAllocation(n_components=10, learning_method="batch",
                                max_iter=25, random_state=0)

# We build the model and transform the data in one step
# Computing transform takes some time,
# and we can save time by doing both at once
document_topics = lda.fit_transform(X)
print("The size of lda.components_: {}".format(lda.components_.shape))

# Most important words from each topic
# We make sorting in descending order
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# Get the feature names
feature_names = np.array(vect.get_feature_names())

# Print out for 10 topics
mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,
                           sorting=sorting, topics_per_chunk=5, n_words=10)

# Let us consider 100 topics
lda100 = LatentDirichletAllocation(n_components=100, learning_method="batch",
                                   max_iter=25, random_state=0)

document_topics100 = lda100.fit_transform(X)
sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
mglearn.tools.print_topics(topics=[10, 20, 30, 40, 50, 60, 70, 80, 90],
                           feature_names=feature_names,
                           sorting=sorting, topics_per_chunk=7, n_words=20)

# let us view contents of one random topic
any_topic = np.argsort(document_topics100[:, 60])[::-1]
for i in any_topic[:10]:
    print(b".".join(text_train[i].split(b".")[:2]) + b".\n")

# Let us plot now
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
topic_names = ["{:>2} ".format(i) + " ".join(words)
               for i, words in enumerate(feature_names[sorting[:, :2]])]

# 2 column bar chart
for col in [0, 1]:
    start = col * 50
    end = (col + 1) * 50
    ax[col].barh(np.arange(50), np.sum(document_topics100, axis=0)[start:end])
    ax[col].set_yticks(np.arange(50))
    ax[col].set_ytickslabels(topic_names[start:end], ha="left", va="top")
    ax[col].invert_yaxis()
    ax[col].set_xlim(0, 2000)
    yax = ax[col].get_yaxis()
    yax.set_tick_params(pad=130)
plt.tight_layout()
plt.show(block=True)

