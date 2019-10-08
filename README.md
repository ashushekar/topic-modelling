# Topic Modelling and Document Clustering

## Topic Modelling

An umbrella term describing the task of assigning each document to one or mutiple _topics_, usually without
supervision. A good example for this is news data, which might be categorized into topics like "politics", 
"sports", "finance", and so on.

Often, when people talk about topic modelling, they refer to one particular decomposition method called
_Latent Dirichlet Allocation_ (LDA)

### Latent Dirichlet Allocation

Let us apply LDA to IMDb movie review dataset. For unsupervised text document models, it is often good to
remove very common words, as they might otherwise dominate the analysis. We will remove words that appear 
in at at least 20 percent of the documents, and we will limit the bag-of-words model to the 10,000 words
that are most common after removing the top 20 percent.

```sh
vect = CountVectorizer(max_features=10000, max_df=0.15)
X = vect.fit_transform(text_train)
``` 

#### With only 10 topics

We will learn a topic modelling with 10 topics. We will use the "batch" learning method, which is somewhat 
slower than the default "online" but usually provides better results, and increase "max_iter" which can also
lead to better models.

```sh
lda = LatentDirichletAllocation(n_components=10, learning_method="batch",
                                max_iter=25, random_state=0)

# We build the model and transform the data in one step
# Computing transform takes some time,
# and we can save time by doing both at once
document_topics = lda.fit_transform(X)

print("The size of lda.components_: {}".format(lda.components_.shape))
```

To understand better what the different topics mean, we will look for most important words for each of the 
topics.

```sh
# We make sorting in decending order
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# Get the feature names
feature_names = np.array(vect.get_feature_names())

# Print out for 10 topics
mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,
                           sorting=sorting, topics_per_chunk=5, n_words=10)

```

Output

```sh
The size of lda.components_: (10, 10000)
topic 0       topic 1       topic 2       topic 3       topic 4       
--------      --------      --------      --------      --------      
cast          didn          horror        wife          dvd           
director      nothing       effects       woman         years         
role          worst         pretty        father        version       
performance   actors        re            young         original      
michael       actually      gore          john          saw           
actors        minutes       around        old           kids          
production    script        guy           family        children      
version       thought       special       mother        old           
john          want          blood         gets          animation     
actor         going         budget        husband       video         


topic 5       topic 6       topic 7       topic 8       topic 9       
--------      --------      --------      --------      --------      
music         action        between       war           show          
funny         police        work          world         series        
girl          role          director      american      funny         
fun           crime         us            us            episode       
school        guy           may           our           tv            
old           lee           real          documentary   comedy        
song          plays         each          men           shows         
songs         car           own           history       episodes      
girls         performance   beautiful     against       season        
lot           cop           world         real          jokes
```

#### With 100 topics

We will consider 100 topics now. Using more topics makes the analysis much harder, but makes it more likely 
that topics can specialize to interesting subsets of the data
```sh
lda100 = LatentDirichletAllocation(n_components=100, learning_method="batch",
                                   max_iter=25, random_state=0)

document_topics100 = lda100.fit_transform(X)
sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
mglearn.tools.print_topics(topics=[10, 20, 30, 40, 50, 60, 70, 80, 90],
                           feature_names=feature_names,
                           sorting=sorting, topics_per_chunk=7, n_words=20)

```

This topics extracted seems to be more specific, though many are hard to interpret. 
 