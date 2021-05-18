```python
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
```


<style>.container { width:100% !important; }</style>



```python
#for structured data we use Spark SQL, SparkSession acts a pipeline between data and sql statements
from pyspark.sql import SparkSession
```


```python
# sparksession is like a class and we need to create an instance of a class to utilize
spark = SparkSession.builder.appName("NLP_5A_Data_Processing").getOrCreate()
```


```python
#Reading the csv file data
Movie_Reviews_DF = spark.read.csv("D:/Movie_reviews.csv", inferSchema = True, header = True)
```


```python
#Seeing the shape of the dataset
print("Shape:", (Movie_Reviews_DF.count(), len(Movie_Reviews_DF.columns)))
```

    Shape: (7087, 2)



```python
#Looking at the schema
#both columns are of string type
Movie_Reviews_DF.printSchema()
```

    root
     |-- Review: string (nullable = true)
     |-- Sentiment: string (nullable = true)
    



```python
#Loding the random function
from pyspark.sql.functions import rand 
#Displaying random observations from the data
Movie_Reviews_DF.orderBy(rand()).show(10,False)  # Note Sentiment values are read in as string
```

    +------------------------------------------------------------------------+---------+
    |Review                                                                  |Sentiment|
    +------------------------------------------------------------------------+---------+
    |Harry Potter dragged Draco Malfoy ’ s trousers down past his hips and   |0        |
    |the Da Vinci Code sucked.                                               |0        |
    |Oh, and Brokeback Mountain is a TERRIBLE movie...                       |0        |
    |Brokeback Mountain is fucking horrible..                                |0        |
    |The Da Vinci Code is awesome!!                                          |1        |
    |I want to be here because I love Harry Potter, and I really want a place|1        |
    |Harry Potter dragged Draco Malfoy ’ s trousers down past his hips and   |0        |
    |As I sit here, watching the MTV Movie Awards, I am reminded of how much |0        |
    |I love Harry Potter..                                                   |1        |
    |, she helped me bobbypin my insanely cool hat to my head, and she laughe|0        |
    +------------------------------------------------------------------------+---------+
    only showing top 10 rows
    



```python
#Cleaning the data
#Filtering the data for data only with only either 0 or 1 sentiment value
Movie_Reviews_DF = Movie_Reviews_DF.filter(((Movie_Reviews_DF.Sentiment =='1') | (Movie_Reviews_DF.Sentiment =='0')))
```


```python
#Checking the count to see if any rows are deleted (rows with different sentiment values)
Movie_Reviews_DF.count()
```




    6990




```python
#Grouping by sentiment values to see balance of data
#(Fairly balanced)
Movie_Reviews_DF.groupBy('Sentiment').count().show()
```

    +---------+-----+
    |Sentiment|count|
    +---------+-----+
    |        0| 3081|
    |        1| 3909|
    +---------+-----+
    



```python
#Looking at the schema again
Movie_Reviews_DF.printSchema()
```

    root
     |-- Review: string (nullable = true)
     |-- Sentiment: string (nullable = true)
    



```python
#in order to perform logistic regression 
#we should have sentiment value of numeric datatype
#Adding a column label to store converted float values 
#from string value in Sentiment (and dropping the Sentiment(String type) column)
Movie_Reviews_DF = Movie_Reviews_DF.withColumn("Label", Movie_Reviews_DF.Sentiment.cast('float')).drop('Sentiment')
```


```python
#Looking at the schema again
Movie_Reviews_DF.printSchema()
```

    root
     |-- Review: string (nullable = true)
     |-- Label: float (nullable = true)
    



```python
#Displaying random data
Movie_Reviews_DF.orderBy(rand()).show(10,False)
```

    +----------------------------------------------------------------------------+-----+
    |Review                                                                      |Label|
    +----------------------------------------------------------------------------+-----+
    |I am going to start reading the Harry Potter series again because that i    |1.0  |
    |Harry Potter is AWESOME I don't care if anyone says differently!..          |1.0  |
    |I want to be here because I love Harry Potter, and I really want a place    |1.0  |
    |But if Crash won the academy award, Brokeback Mountain must have sucked     |0.0  |
    |Because I would like to make friends who like the same things I like, an    |1.0  |
    |Brokeback Mountain was an AWESOME movie.                                    |1.0  |
    |the last stand and Mission Impossible 3 both were awesome movies.           |1.0  |
    |Other than that, all I've heard is that the Da Vinci Code kinda sucks!      |0.0  |
    |"I think the movie "" Brokeback Mountain "" was stupid and overexagerated.."|0.0  |
    |Brokeback Mountain is fucking horrible..                                    |0.0  |
    +----------------------------------------------------------------------------+-----+
    only showing top 10 rows
    



```python
#Checking for the values after transformation
Movie_Reviews_DF.groupBy('label').count().show()
```

    +-----+-----+
    |label|count|
    +-----+-----+
    |  1.0| 3909|
    |  0.0| 3081|
    +-----+-----+
    



```python
# Adding length column to the dataframe
#Length of the review might matter because repetition of words would occur in the same review
#Loading length function 
from pyspark.sql.functions import length
```


```python
#For each row calculating length of review and adding it to a new column
Movie_Reviews_DF = Movie_Reviews_DF.withColumn('length',length(Movie_Reviews_DF['Review']))
```


```python
#Displaying the data
Movie_Reviews_DF.orderBy(rand()).show(5,False)
```

    +------------------------------------------------------------------------+-----+------+
    |Review                                                                  |Label|length|
    +------------------------------------------------------------------------+-----+------+
    |I hate Harry Potter.                                                    |0.0  |20    |
    |the last stand and Mission Impossible 3 both were awesome movies.       |1.0  |65    |
    |we're gonna like watch Mission Impossible or Hoot.(                     |1.0  |51    |
    |I think I hate Harry Potter because it outshines much better reading mat|0.0  |72    |
    |I hate Harry Potter.                                                    |0.0  |20    |
    +------------------------------------------------------------------------+-----+------+
    only showing top 5 rows
    



```python
#Average length of a review for a 0 and 1 sentiment review(negative and positive)
#Fairly close
Movie_Reviews_DF.groupBy('Label').agg({'Length':'mean'}).show()
```

    +-----+-----------------+
    |Label|      avg(Length)|
    +-----+-----------------+
    |  1.0|47.61882834484523|
    |  0.0|50.95845504706264|
    +-----+-----------------+
    



```python
# Data Preprocessing for NLP
```


```python
#Tokenization
#Importing the Tokenizer function
from pyspark.ml.feature import Tokenizer
```


```python
#Taking review column and creating new column tokens for storing the tokens created from review column
tokenization = Tokenizer(inputCol='Review',outputCol='Tokens')
```


```python
#Applying the Tokenizer function to the dataframe
Tokenized_DF = tokenization.transform(Movie_Reviews_DF)
```


```python
#looking at the tokens columns
Tokenized_DF.select('Tokens').show(10, False)
```

    +----------------------------------------------------------------------------------------+
    |Tokens                                                                                  |
    +----------------------------------------------------------------------------------------+
    |[the, da, vinci, code, book, is, just, awesome.]                                        |
    |[this, was, the, first, clive, cussler, i've, ever, read,, but, even, books, like, rel] |
    |[i, liked, the, da, vinci, code, a, lot.]                                               |
    |[i, liked, the, da, vinci, code, a, lot.]                                               |
    |[i, liked, the, da, vinci, code, but, it, ultimatly, didn't, seem, to, hold, it's, own.]|
    |[that's, not, even, an, exaggeration, ), and, at, midnight, we, went, to, wal-mart, to] |
    |[i, loved, the, da, vinci, code,, but, now, i, want, something, better, and, different] |
    |[i, thought, da, vinci, code, was, great,, same, with, kite, runner.]                   |
    |[the, da, vinci, code, is, actually, a, good, movie...]                                 |
    |[i, thought, the, da, vinci, code, was, a, pretty, good, book.]                         |
    +----------------------------------------------------------------------------------------+
    only showing top 10 rows
    



```python
#To get a count of tokens for each row before removing the stop words
# importing size function from sql functions
from pyspark.sql.functions import size
```


```python
#Selecting all columns from dataframe and adding a new column based on no of tokens in each observation
# Size is a sql function to count number of items in a list
Tokenized_DF = Tokenized_DF.select('*',size('Tokens').alias('Tokens_Count'))
```


```python
#looking at the tokens and tokens count columns
#Tokenization converts sentences to lower case and then creates tokens
Tokenized_DF.select('Tokens','Tokens_Count').show(10, False)
```

    +----------------------------------------------------------------------------------------+------------+
    |Tokens                                                                                  |Tokens_Count|
    +----------------------------------------------------------------------------------------+------------+
    |[the, da, vinci, code, book, is, just, awesome.]                                        |8           |
    |[this, was, the, first, clive, cussler, i've, ever, read,, but, even, books, like, rel] |14          |
    |[i, liked, the, da, vinci, code, a, lot.]                                               |8           |
    |[i, liked, the, da, vinci, code, a, lot.]                                               |8           |
    |[i, liked, the, da, vinci, code, but, it, ultimatly, didn't, seem, to, hold, it's, own.]|15          |
    |[that's, not, even, an, exaggeration, ), and, at, midnight, we, went, to, wal-mart, to] |14          |
    |[i, loved, the, da, vinci, code,, but, now, i, want, something, better, and, different] |14          |
    |[i, thought, da, vinci, code, was, great,, same, with, kite, runner.]                   |11          |
    |[the, da, vinci, code, is, actually, a, good, movie...]                                 |9           |
    |[i, thought, the, da, vinci, code, was, a, pretty, good, book.]                         |11          |
    +----------------------------------------------------------------------------------------+------------+
    only showing top 10 rows
    



```python
#Removal of stopwords
#Importing the StopWordsRemover function
from pyspark.ml.feature import StopWordsRemover
```


```python
#Taking Tokens column and creating new column Refined Tokens for storing the tokens after removal of stopwords
stopword_removal=StopWordsRemover(inputCol='Tokens',outputCol='Refined_Tokens')
```


```python
#Applying the StopWordsRemover function to the dataframe
Refined_DF = stopword_removal.transform(Tokenized_DF)
```


```python
#Selecting only the refined tokens column which has tokens after stop words have been removed
Refined_DF.select(['Refined_Tokens']).show(10,False)
```

    +-------------------------------------------------------------+
    |Refined_Tokens                                               |
    +-------------------------------------------------------------+
    |[da, vinci, code, book, awesome.]                            |
    |[first, clive, cussler, ever, read,, even, books, like, rel] |
    |[liked, da, vinci, code, lot.]                               |
    |[liked, da, vinci, code, lot.]                               |
    |[liked, da, vinci, code, ultimatly, seem, hold, own.]        |
    |[even, exaggeration, ), midnight, went, wal-mart]            |
    |[loved, da, vinci, code,, want, something, better, different]|
    |[thought, da, vinci, code, great,, kite, runner.]            |
    |[da, vinci, code, actually, good, movie...]                  |
    |[thought, da, vinci, code, pretty, good, book.]              |
    +-------------------------------------------------------------+
    only showing top 10 rows
    



```python
#To get a count of tokens for each row after removing the stop words
#importing size function from sql functions
from pyspark.sql.functions import size
```


```python
#Selecting all columns from dataframe and adding a new column based on no of refined tokens in each observation
#Size is a sql function to count number of items in a list
Refined_DF = Refined_DF.select('*',size('Refined_Tokens').alias('Refined_Tokens_Count'))
```


```python
#Looking at the tokens,tokens count and refined tokens, refined tokens count columns
#To see if the counts vary which indicates removal of stop words in tokens 
Refined_DF.select('Tokens','Tokens_Count','Refined_Tokens','Refined_Tokens_Count').show(10, False)
```

    +----------------------------------------------------------------------------------------+------------+-------------------------------------------------------------+--------------------+
    |Tokens                                                                                  |Tokens_Count|Refined_Tokens                                               |Refined_Tokens_Count|
    +----------------------------------------------------------------------------------------+------------+-------------------------------------------------------------+--------------------+
    |[the, da, vinci, code, book, is, just, awesome.]                                        |8           |[da, vinci, code, book, awesome.]                            |5                   |
    |[this, was, the, first, clive, cussler, i've, ever, read,, but, even, books, like, rel] |14          |[first, clive, cussler, ever, read,, even, books, like, rel] |9                   |
    |[i, liked, the, da, vinci, code, a, lot.]                                               |8           |[liked, da, vinci, code, lot.]                               |5                   |
    |[i, liked, the, da, vinci, code, a, lot.]                                               |8           |[liked, da, vinci, code, lot.]                               |5                   |
    |[i, liked, the, da, vinci, code, but, it, ultimatly, didn't, seem, to, hold, it's, own.]|15          |[liked, da, vinci, code, ultimatly, seem, hold, own.]        |8                   |
    |[that's, not, even, an, exaggeration, ), and, at, midnight, we, went, to, wal-mart, to] |14          |[even, exaggeration, ), midnight, went, wal-mart]            |6                   |
    |[i, loved, the, da, vinci, code,, but, now, i, want, something, better, and, different] |14          |[loved, da, vinci, code,, want, something, better, different]|8                   |
    |[i, thought, da, vinci, code, was, great,, same, with, kite, runner.]                   |11          |[thought, da, vinci, code, great,, kite, runner.]            |7                   |
    |[the, da, vinci, code, is, actually, a, good, movie...]                                 |9           |[da, vinci, code, actually, good, movie...]                  |6                   |
    |[i, thought, the, da, vinci, code, was, a, pretty, good, book.]                         |11          |[thought, da, vinci, code, pretty, good, book.]              |7                   |
    +----------------------------------------------------------------------------------------+------------+-------------------------------------------------------------+--------------------+
    only showing top 10 rows
    



```python
#Looking at random data
Refined_DF.orderBy(rand()).show(4, False)
```

    +------------------------------------------------------------------------+-----+------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+--------------------+
    |Review                                                                  |Label|length|Tokens                                                                                   |Tokens_Count|Refined_Tokens                                   |Refined_Tokens_Count|
    +------------------------------------------------------------------------+-----+------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+--------------------+
    |My dad's being stupid about brokeback mountain...                       |0.0  |49    |[my, dad's, being, stupid, about, brokeback, mountain...]                                |7           |[dad's, stupid, brokeback, mountain...]          |4                   |
    |Brokeback Mountain was an AWESOME movie.                                |1.0  |40    |[brokeback, mountain, was, an, awesome, movie.]                                          |6           |[brokeback, mountain, awesome, movie.]           |4                   |
    |Brokeback Mountain was a shitty movie.                                  |0.0  |38    |[brokeback, mountain, was, a, shitty, movie.]                                            |6           |[brokeback, mountain, shitty, movie.]            |4                   |
    |I want to be here because I love Harry Potter, and I really want a place|1.0  |72    |[i, want, to, be, here, because, i, love, harry, potter,, and, i, really, want, a, place]|16          |[want, love, harry, potter,, really, want, place]|7                   |
    +------------------------------------------------------------------------+-----+------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+--------------------+
    only showing top 4 rows
    



```python
#Looking at schema
Refined_DF.printSchema()
```

    root
     |-- Review: string (nullable = true)
     |-- Label: float (nullable = true)
     |-- length: integer (nullable = true)
     |-- Tokens: array (nullable = true)
     |    |-- element: string (containsNull = true)
     |-- Tokens_Count: integer (nullable = false)
     |-- Refined_Tokens: array (nullable = true)
     |    |-- element: string (containsNull = true)
     |-- Refined_Tokens_Count: integer (nullable = false)
    



```python
# Count Vectorization
```


```python
#Creating features based on Count Vectorization in PySpark using the Refined dataframe
```


```python
#Imprting function for CV calculation
from pyspark.ml.feature import CountVectorizer
```


```python
#Taking refined tokens column and creating new column CV features 
count_vec=CountVectorizer(inputCol='Refined_Tokens', outputCol='CV_features')
```


```python
CV_DF=count_vec.fit(Refined_DF).transform(Refined_DF)
```


```python
CV_DF.select(['Refined_Tokens','CV_features']).show(10,False)
```

    +-------------------------------------------------------------+----------------------------------------------------------------------------------+
    |Refined_Tokens                                               |CV_features                                                                       |
    +-------------------------------------------------------------+----------------------------------------------------------------------------------+
    |[da, vinci, code, book, awesome.]                            |(2302,[0,1,4,43,236],[1.0,1.0,1.0,1.0,1.0])                                       |
    |[first, clive, cussler, ever, read,, even, books, like, rel] |(2302,[11,51,229,237,275,742,824,1087,1250],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])|
    |[liked, da, vinci, code, lot.]                               |(2302,[0,1,4,53,356],[1.0,1.0,1.0,1.0,1.0])                                       |
    |[liked, da, vinci, code, lot.]                               |(2302,[0,1,4,53,356],[1.0,1.0,1.0,1.0,1.0])                                       |
    |[liked, da, vinci, code, ultimatly, seem, hold, own.]        |(2302,[0,1,4,53,655,1339,1427,1449],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])            |
    |[even, exaggeration, ), midnight, went, wal-mart]            |(2302,[46,229,271,1150,1990,2203],[1.0,1.0,1.0,1.0,1.0,1.0])                      |
    |[loved, da, vinci, code,, want, something, better, different]|(2302,[0,1,22,30,111,219,389,535],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])              |
    |[thought, da, vinci, code, great,, kite, runner.]            |(2302,[0,1,4,228,1258,1716,2263],[1.0,1.0,1.0,1.0,1.0,1.0,1.0])                   |
    |[da, vinci, code, actually, good, movie...]                  |(2302,[0,1,4,33,226,258],[1.0,1.0,1.0,1.0,1.0,1.0])                               |
    |[thought, da, vinci, code, pretty, good, book.]              |(2302,[0,1,4,223,226,228,262],[1.0,1.0,1.0,1.0,1.0,1.0,1.0])                      |
    +-------------------------------------------------------------+----------------------------------------------------------------------------------+
    only showing top 10 rows
    



```python
#this method takes each word in the BoW and counts how many times that word appears in each document. It is basically computing Term Frequency (TF) or the number of times each word occurs in each document.
```


```python
CV_DF_Model = CV_DF.select(['CV_features','Label'])
```


```python
CV_DF_Model.select(['Label','CV_features']).show(10,False)
```

    +-----+----------------------------------------------------------------------------------+
    |Label|CV_features                                                                       |
    +-----+----------------------------------------------------------------------------------+
    |1.0  |(2302,[0,1,4,43,236],[1.0,1.0,1.0,1.0,1.0])                                       |
    |1.0  |(2302,[11,51,229,237,275,742,824,1087,1250],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])|
    |1.0  |(2302,[0,1,4,53,356],[1.0,1.0,1.0,1.0,1.0])                                       |
    |1.0  |(2302,[0,1,4,53,356],[1.0,1.0,1.0,1.0,1.0])                                       |
    |1.0  |(2302,[0,1,4,53,655,1339,1427,1449],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])            |
    |1.0  |(2302,[46,229,271,1150,1990,2203],[1.0,1.0,1.0,1.0,1.0,1.0])                      |
    |1.0  |(2302,[0,1,22,30,111,219,389,535],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])              |
    |1.0  |(2302,[0,1,4,228,1258,1716,2263],[1.0,1.0,1.0,1.0,1.0,1.0,1.0])                   |
    |1.0  |(2302,[0,1,4,33,226,258],[1.0,1.0,1.0,1.0,1.0,1.0])                               |
    |1.0  |(2302,[0,1,4,223,226,228,262],[1.0,1.0,1.0,1.0,1.0,1.0,1.0])                      |
    +-----+----------------------------------------------------------------------------------+
    only showing top 10 rows
    



```python
#To see the schema of the dataset
CV_DF_Model.printSchema()
```

    root
     |-- CV_features: vector (nullable = true)
     |-- Label: float (nullable = true)
    



```python
#Splitting the data of CV model
CV_Training_DF,CV_Test_DF = CV_DF_Model.randomSplit([0.75,0.25])
```


```python
#Checking the balance of training dataframe of CV model
CV_Training_DF.groupBy('Label').count().show()
```

    +-----+-----+
    |Label|count|
    +-----+-----+
    |  1.0| 2921|
    |  0.0| 2312|
    +-----+-----+
    



```python
#Checking the balance of testing dataframe of CV model
CV_Test_DF.groupBy('Label').count().show()
```

    +-----+-----+
    |Label|count|
    +-----+-----+
    |  1.0|  988|
    |  0.0|  769|
    +-----+-----+
    



```python
#Term Frequency(TF) and Inverse Document Frequency(IDF)
```


```python
#Creating features based on TF-IDF in PySpark using the Refined dataframe
```


```python
#Imprting function for TF and IDF calculation
from pyspark.ml.feature import HashingTF,IDF
```


```python
#TERM FREQUENCY
#It is the score based on the number of times the word appears in current dataframe
#Taking refined tokens column and creating new column tf features for storing the tf value created
hashing_vec=HashingTF(inputCol='Refined_Tokens',outputCol='TF_features')
```


```python
#Applying the HashingTF function to the dataframe
Hashing_DF = hashing_vec.transform(Refined_DF)
```


```python
#Looking at the refined tokens and corresoponding TF features columns
Hashing_DF.select(['Refined_Tokens','TF_features']).show(4,False)
```

    +------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
    |Refined_Tokens                                              |TF_features                                                                                                  |
    +------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
    |[da, vinci, code, book, awesome.]                           |(262144,[93284,111793,189113,212976,235054],[1.0,1.0,1.0,1.0,1.0])                                           |
    |[first, clive, cussler, ever, read,, even, books, like, rel]|(262144,[47372,82111,113624,120246,139559,174966,203802,208258,227467],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])|
    |[liked, da, vinci, code, lot.]                              |(262144,[32675,93284,111793,227152,235054],[1.0,1.0,1.0,1.0,1.0])                                            |
    |[liked, da, vinci, code, lot.]                              |(262144,[32675,93284,111793,227152,235054],[1.0,1.0,1.0,1.0,1.0])                                            |
    +------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
    only showing top 4 rows
    



```python
#The term TF represents term frequency and computes a ratio of the number of times a word appears in a document divided by the total number of terms in that document. It therefore attempts to measure “importance” – higher the value, the more frequently the term was used with respect to other terms in the document. 
```


```python
#[da, vinci, code, book, awesome.]---(262144,[93284,111793,189113,212976,235054],[1.0,1.0,1.0,1.0,1.0])  
#262144 - Total number of tokens in the dataframe
#93284 - frequency of the word da
#111793 - frequency of the word vinci
#189113 - frequency of the word code
#212976 - frequency of the word book
#235054 - frequency of the word awesome
#[1.0,1.0,1.0,1.0,1.0])-- list indicating the presence of the words [da, vinci, code, book, awesome.] in the review with 1
```


```python
#INVERSE DOCUMENT FREQUENCY
#It is calculated by taking the total number of documents, dividing it by the number of documents that contain a word, and calculating the logarithm
#Taking TF features column and creating new column TF-IDF features for storing the TF-IDF value created
TF_IDF_vec=IDF(inputCol='TF_features',outputCol='TF_IDF_features')
```


```python
#The term IDF, computes a measure of relative importance of the term with respect to the same term used in all other documents in the corpus. Thus, if a term appears in all documents, it’s not helping in differentiating documents. Such a term will therefore be assigned a very low relative importance.
```


```python
#Applying the IDF function to the dataframe
TF_IDF_DF = TF_IDF_vec.fit(Hashing_DF).transform(Hashing_DF)
```


```python
#Looking at the refined tokens and corresoponding TF-IDF features columns
#Multiplying these TF and IDF results in the TF-IDF score of a word in a document. 
#The higher the score, the more relevant that word is in that particular document.
TF_IDF_DF.select(['Refined_Tokens','TF_IDF_features']).show(10,False)
```

    +-------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Refined_Tokens                                               |TF_IDF_features                                                                                                                                                                                                                           |
    +-------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[da, vinci, code, book, awesome.]                            |(262144,[93284,111793,189113,212976,235054],[1.469010739519602,1.2610218398134343,6.079790164272204,4.0401945311395675,1.2620319409094194])                                                                                               |
    |[first, clive, cussler, ever, read,, even, books, like, rel] |(262144,[47372,82111,113624,120246,139559,174966,203802,208258,227467],[6.144328685409775,4.247208700523894,7.7537665978438755,8.15923170595204,8.15923170595204,5.716884670582836,6.772937344832149,2.7039105905943384,8.15923170595204])|
    |[liked, da, vinci, code, lot.]                               |(262144,[32675,93284,111793,227152,235054],[4.267411407841413,1.469010739519602,1.2610218398134343,7.242940974077885,1.2620319409094194])                                                                                                 |
    |[liked, da, vinci, code, lot.]                               |(262144,[32675,93284,111793,227152,235054],[4.267411407841413,1.469010739519602,1.2610218398134343,7.242940974077885,1.2620319409094194])                                                                                                 |
    |[liked, da, vinci, code, ultimatly, seem, hold, own.]        |(262144,[5765,32675,93284,111793,178453,193996,235054,237388],[7.7537665978438755,4.267411407841413,1.469010739519602,1.2610218398134343,8.15923170595204,8.15923170595204,1.2620319409094194,8.15923170595204])                          |
    |[even, exaggeration, ), midnight, went, wal-mart]            |(262144,[105591,146139,174966,197340,243418,248625],[8.15923170595204,4.151898520719569,5.716884670582836,6.772937344832149,8.15923170595204,8.15923170595204])                                                                           |
    |[loved, da, vinci, code,, want, something, better, different]|(262144,[33933,111793,115917,173297,179666,190256,224769,235054],[3.3229497990005616,1.2610218398134343,4.433538278715387,7.7537665978438755,5.326018361895824,4.309084104241982,7.7537665978438755,1.2620319409094194])                  |
    |[thought, da, vinci, code, great,, kite, runner.]            |(262144,[2000,33552,37254,93284,111793,235054,242361],[8.15923170595204,8.15923170595204,8.15923170595204,1.469010739519602,1.2610218398134343,1.2620319409094194,5.716884670582836])                                                     |
    |[da, vinci, code, actually, good, movie...]                  |(262144,[93284,111793,113432,132975,171076,235054],[1.469010739519602,1.2610218398134343,5.761336433153669,6.5497937935179396,3.710715330009325,1.2620319409094194])                                                                      |
    |[thought, da, vinci, code, pretty, good, book.]              |(262144,[23661,93284,111793,113432,175449,235054,242361],[6.5497937935179396,1.469010739519602,1.2610218398134343,5.761336433153669,5.485083056525511,1.2620319409094194,5.716884670582836])                                              |
    +-------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    only showing top 10 rows
    



```python
#The product of TF and IDF gives us the TL-IDF score or weight that ranks each term by its relative importance. 

```


```python
TF_IDF_DF_Model = TF_IDF_DF.select(['TF_IDF_features','Label'])
```


```python
TF_IDF_DF_Model.select(['Label','TF_IDF_features']).show(10,False)
```

    +-----+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Label|TF_IDF_features                                                                                                                                                                                                                           |
    +-----+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |1.0  |(262144,[93284,111793,189113,212976,235054],[1.469010739519602,1.2610218398134343,6.079790164272204,4.0401945311395675,1.2620319409094194])                                                                                               |
    |1.0  |(262144,[47372,82111,113624,120246,139559,174966,203802,208258,227467],[6.144328685409775,4.247208700523894,7.7537665978438755,8.15923170595204,8.15923170595204,5.716884670582836,6.772937344832149,2.7039105905943384,8.15923170595204])|
    |1.0  |(262144,[32675,93284,111793,227152,235054],[4.267411407841413,1.469010739519602,1.2610218398134343,7.242940974077885,1.2620319409094194])                                                                                                 |
    |1.0  |(262144,[32675,93284,111793,227152,235054],[4.267411407841413,1.469010739519602,1.2610218398134343,7.242940974077885,1.2620319409094194])                                                                                                 |
    |1.0  |(262144,[5765,32675,93284,111793,178453,193996,235054,237388],[7.7537665978438755,4.267411407841413,1.469010739519602,1.2610218398134343,8.15923170595204,8.15923170595204,1.2620319409094194,8.15923170595204])                          |
    |1.0  |(262144,[105591,146139,174966,197340,243418,248625],[8.15923170595204,4.151898520719569,5.716884670582836,6.772937344832149,8.15923170595204,8.15923170595204])                                                                           |
    |1.0  |(262144,[33933,111793,115917,173297,179666,190256,224769,235054],[3.3229497990005616,1.2610218398134343,4.433538278715387,7.7537665978438755,5.326018361895824,4.309084104241982,7.7537665978438755,1.2620319409094194])                  |
    |1.0  |(262144,[2000,33552,37254,93284,111793,235054,242361],[8.15923170595204,8.15923170595204,8.15923170595204,1.469010739519602,1.2610218398134343,1.2620319409094194,5.716884670582836])                                                     |
    |1.0  |(262144,[93284,111793,113432,132975,171076,235054],[1.469010739519602,1.2610218398134343,5.761336433153669,6.5497937935179396,3.710715330009325,1.2620319409094194])                                                                      |
    |1.0  |(262144,[23661,93284,111793,113432,175449,235054,242361],[6.5497937935179396,1.469010739519602,1.2610218398134343,5.761336433153669,5.485083056525511,1.2620319409094194,5.716884670582836])                                              |
    +-----+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    only showing top 10 rows
    



```python
#To see the schema of the dataset
TF_IDF_DF_Model.printSchema()
```

    root
     |-- TF_IDF_features: vector (nullable = true)
     |-- Label: float (nullable = true)
    



```python
#Splitting the data of TFIDF model
TFIDF_Training_DF,TFIDF_Test_DF = TF_IDF_DF_Model.randomSplit([0.75,0.25])
```


```python
#Checking the balance of training dataframe
TFIDF_Training_DF.groupBy('Label').count().show()
```

    +-----+-----+
    |Label|count|
    +-----+-----+
    |  1.0| 2937|
    |  0.0| 2313|
    +-----+-----+
    



```python
#Checking the balance of testing dataframe
TFIDF_Test_DF.groupBy('Label').count().show()
```

    +-----+-----+
    |Label|count|
    +-----+-----+
    |  1.0|  972|
    |  0.0|  768|
    +-----+-----+
    



```python
#Importing Logistic Regression
from pyspark.ml.classification import LogisticRegression
```


```python
#Logistic Regression model(Using CV)
CV_log_reg=LogisticRegression(featuresCol='CV_features',labelCol='Label').fit(CV_Training_DF)
```


```python
#Get Training Summary(Using CV)
```


```python
CV_training_summary = CV_log_reg.summary
print("Area Under ROC:" + str(CV_training_summary.areaUnderROC))
print("Weighted Accuracy:" + str(CV_training_summary.accuracy))
print("Weighted Recall:" + str(CV_training_summary.weightedRecall))
print("Weighted Precision:" + str(CV_training_summary.weightedPrecision))
print("Weighted F1 Measure:" + str(CV_training_summary.weightedFMeasure()))
```

    Area Under ROC:0.9999984452165384
    Weighted Accuracy:1.0
    Weighted Recall:1.0
    Weighted Precision:1.0
    Weighted F1 Measure:1.0



```python
#Evaluation of test data (Using CV)
CV_results=CV_log_reg.evaluate(CV_Test_DF).predictions
```


```python
#Displaying the results of TFIDF
CV_results.show(10, False)
```

    +-------------------------------------------------------------------------------------+-----+----------------------------------------+------------------------------------------+----------+
    |CV_features                                                                          |Label|rawPrediction                           |probability                               |prediction|
    +-------------------------------------------------------------------------------------+-----+----------------------------------------+------------------------------------------+----------+
    |(2302,[0,1,4,5,12,305,340],[1.0,1.0,1.0,1.0,1.0,1.0,1.0])                            |1.0  |[-26.653836401563705,26.653836401563705]|[2.656965585654917E-12,0.999999999997343] |1.0       |
    |(2302,[0,1,4,5,64,2029],[1.0,1.0,1.0,1.0,1.0,1.0])                                   |1.0  |[-16.511341767538717,16.511341767538717]|[6.748625867178597E-8,0.9999999325137413] |1.0       |
    |(2302,[0,1,4,5,220,247,338,636,1706],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])          |1.0  |[2.6182268092636036,-2.6182268092636036]|[0.9320254537061838,0.06797454629381615]  |0.0       |
    |(2302,[0,1,4,5,303],[1.0,1.0,1.0,1.0,1.0])                                           |1.0  |[-25.349099013548834,25.349099013548834]|[9.79549026155353E-12,0.9999999999902045] |1.0       |
    |(2302,[0,1,4,5,449],[1.0,1.0,1.0,1.0,1.0])                                           |1.0  |[-20.59352738933587,20.59352738933587]  |[1.1385305576935607E-9,0.9999999988614694]|1.0       |
    |(2302,[0,1,4,5,652],[1.0,1.0,1.0,1.0,1.0])                                           |1.0  |[-14.000816360022458,14.000816360022458]|[8.308494789964832E-7,0.999999169150521]  |1.0       |
    |(2302,[0,1,4,10,14,100,524,1294,1957,2049],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])|0.0  |[31.880716800010358,-31.880716800010358]|[0.9999999999999858,1.426857540299771E-14]|0.0       |
    |(2302,[0,1,4,12,16,236,238,247],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])                   |1.0  |[-14.595468879956611,14.595468879956611]|[4.5842490699912783E-7,0.999999541575093] |1.0       |
    |(2302,[0,1,4,12,25,53,223,464,956],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])            |1.0  |[-42.80294422752487,42.80294422752487]  |[2.575831872851638E-19,1.0]               |1.0       |
    |(2302,[0,1,4,12,33],[1.0,1.0,1.0,1.0,1.0])                                           |1.0  |[-21.07250649368786,21.07250649368786]  |[7.052233925854608E-10,0.9999999992947766]|1.0       |
    +-------------------------------------------------------------------------------------+-----+----------------------------------------+------------------------------------------+----------+
    only showing top 10 rows
    



```python
CV_results.select('Label', 'prediction','probability').show(20,False)
```

    +-----+----------+------------------------------------------+
    |Label|prediction|probability                               |
    +-----+----------+------------------------------------------+
    |1.0  |1.0       |[2.656965585654917E-12,0.999999999997343] |
    |1.0  |1.0       |[6.748625867178597E-8,0.9999999325137413] |
    |1.0  |0.0       |[0.9320254537061838,0.06797454629381615]  |
    |1.0  |1.0       |[9.79549026155353E-12,0.9999999999902045] |
    |1.0  |1.0       |[1.1385305576935607E-9,0.9999999988614694]|
    |1.0  |1.0       |[8.308494789964832E-7,0.999999169150521]  |
    |0.0  |0.0       |[0.9999999999999858,1.426857540299771E-14]|
    |1.0  |1.0       |[4.5842490699912783E-7,0.999999541575093] |
    |1.0  |1.0       |[2.575831872851638E-19,1.0]               |
    |1.0  |1.0       |[7.052233925854608E-10,0.9999999992947766]|
    |1.0  |1.0       |[7.052233925854608E-10,0.9999999992947766]|
    |1.0  |1.0       |[7.052233925854608E-10,0.9999999992947766]|
    |1.0  |1.0       |[7.052233925854608E-10,0.9999999992947766]|
    |1.0  |1.0       |[7.052233925854608E-10,0.9999999992947766]|
    |1.0  |1.0       |[7.052233925854608E-10,0.9999999992947766]|
    |1.0  |1.0       |[7.052233925854608E-10,0.9999999992947766]|
    |1.0  |1.0       |[7.052233925854608E-10,0.9999999992947766]|
    |1.0  |1.0       |[7.052233925854608E-10,0.9999999992947766]|
    |1.0  |1.0       |[7.052233925854608E-10,0.9999999992947766]|
    |1.0  |1.0       |[7.052233925854608E-10,0.9999999992947766]|
    +-----+----------+------------------------------------------+
    only showing top 20 rows
    



```python
#confusion matrix for CV results
CV_true_positives = CV_results[(CV_results.Label == 1) & (CV_results.prediction == 1)].count()
CV_true_negatives = CV_results[(CV_results.Label == 0) & (CV_results.prediction == 0)].count()
CV_false_positives = CV_results[(CV_results.Label == 0) & (CV_results.prediction == 1)].count()
CV_false_negatives = CV_results[(CV_results.Label == 1) & (CV_results.prediction == 0)].count()
```


```python
#Displaying Confurion matrix of CV
print("CV_true_postives ARE :", CV_true_positives , "CV_true_negatives ARE :", CV_true_negatives)
print("CV_false_postives ARE :" ,CV_false_positives , "CV_false_negatives ARE :" ,CV_false_negatives)
```

    CV_true_postives ARE : 978 CV_true_negatives ARE : 743
    CV_false_postives ARE : 26 CV_false_negatives ARE : 10



```python
#CV Recall Value
CV_recall = float(CV_true_positives)/(CV_true_positives + CV_false_negatives)
print("CV Recall Value is :" ,CV_recall)
```

    CV Recall Value is : 0.9898785425101214



```python
#CV Precision Value
CV_precision = float(CV_true_positives) / (CV_true_positives + CV_false_positives)
print("CV Precision Value is :" ,CV_precision)
```

    CV Precision Value is : 0.9741035856573705



```python
#CV Accuracy Value
CV_accuracy=float((CV_true_positives + CV_true_negatives) /(CV_results.count()))
print("Cv Accuracy Value is :" ,CV_accuracy)
```

    Cv Accuracy Value is : 0.9795105293113261



```python
#Logistic Regression model(Using TF-IDF)
TFIDF_log_reg=LogisticRegression(featuresCol='TF_IDF_features',labelCol='Label').fit(TFIDF_Training_DF)
```


```python
#Get Training Summary(Using TF-IDF)
```


```python
TFIDF_training_summary = TFIDF_log_reg.summary
print("Area Under ROC:" + str(TFIDF_training_summary.areaUnderROC))
print("Weighted Accuracy:" + str(TFIDF_training_summary.accuracy))
print("Weighted Recall:" + str(TFIDF_training_summary.weightedRecall))
print("Weighted Precision:" + str(TFIDF_training_summary.weightedPrecision))
print("Weighted F1 Measure:" + str(TFIDF_training_summary.weightedFMeasure()))
```

    Area Under ROC:0.9999981599465707
    Weighted Accuracy:1.0
    Weighted Recall:1.0
    Weighted Precision:1.0
    Weighted F1 Measure:1.0



```python
#Evaluation of test data (Using TF-IDF)
TFIDF_results=TFIDF_log_reg.evaluate(TFIDF_Test_DF).predictions
```


```python
#Displaying the results of TFIDF
TFIDF_results.show(10, False)
```

    +----------------------------------------------------------------------------------------------------------+-----+----------------------------------------+------------------------------------------+----------+
    |TF_IDF_features                                                                                           |Label|rawPrediction                           |probability                               |prediction|
    +----------------------------------------------------------------------------------------------------------+-----+----------------------------------------+------------------------------------------+----------+
    |(262144,[14,535,31179,197995],[3.203404648350779,1.8474968967991254,4.398031590258477,1.3535091525350544])|0.0  |[20.055594793030522,-20.055594793030522]|[0.9999999980503087,1.9496912797521345E-9]|0.0       |
    |(262144,[14,535,31179,197995],[3.203404648350779,1.8474968967991254,4.398031590258477,1.3535091525350544])|0.0  |[20.055594793030522,-20.055594793030522]|[0.9999999980503087,1.9496912797521345E-9]|0.0       |
    |(262144,[14,535,31179,197995],[3.203404648350779,1.8474968967991254,4.398031590258477,1.3535091525350544])|0.0  |[20.055594793030522,-20.055594793030522]|[0.9999999980503087,1.9496912797521345E-9]|0.0       |
    |(262144,[14,535,31179,197995],[3.203404648350779,1.8474968967991254,4.398031590258477,1.3535091525350544])|0.0  |[20.055594793030522,-20.055594793030522]|[0.9999999980503087,1.9496912797521345E-9]|0.0       |
    |(262144,[14,535,31179,197995],[3.203404648350779,1.8474968967991254,4.398031590258477,1.3535091525350544])|0.0  |[20.055594793030522,-20.055594793030522]|[0.9999999980503087,1.9496912797521345E-9]|0.0       |
    |(262144,[14,535,31179,197995],[3.203404648350779,1.8474968967991254,4.398031590258477,1.3535091525350544])|0.0  |[20.055594793030522,-20.055594793030522]|[0.9999999980503087,1.9496912797521345E-9]|0.0       |
    |(262144,[14,535,31179,197995],[3.203404648350779,1.8474968967991254,4.398031590258477,1.3535091525350544])|0.0  |[20.055594793030522,-20.055594793030522]|[0.9999999980503087,1.9496912797521345E-9]|0.0       |
    |(262144,[14,535,31179,197995],[3.203404648350779,1.8474968967991254,4.398031590258477,1.3535091525350544])|0.0  |[20.055594793030522,-20.055594793030522]|[0.9999999980503087,1.9496912797521345E-9]|0.0       |
    |(262144,[14,535,31179,197995],[3.203404648350779,1.8474968967991254,4.398031590258477,1.3535091525350544])|0.0  |[20.055594793030522,-20.055594793030522]|[0.9999999980503087,1.9496912797521345E-9]|0.0       |
    |(262144,[14,535,31179,197995],[3.203404648350779,1.8474968967991254,4.398031590258477,1.3535091525350544])|0.0  |[20.055594793030522,-20.055594793030522]|[0.9999999980503087,1.9496912797521345E-9]|0.0       |
    +----------------------------------------------------------------------------------------------------------+-----+----------------------------------------+------------------------------------------+----------+
    only showing top 10 rows
    



```python
TFIDF_results.select('Label', 'prediction','probability').show(20,False)
```

    +-----+----------+-------------------------------------------+
    |Label|prediction|probability                                |
    +-----+----------+-------------------------------------------+
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |0.0       |[0.9999999980503087,1.9496912797521345E-9] |
    |0.0  |1.0       |[0.4794483458035445,0.5205516541964554]    |
    |1.0  |1.0       |[2.7314263108247142E-11,0.9999999999726856]|
    +-----+----------+-------------------------------------------+
    only showing top 20 rows
    



```python
#For TF-IDF results BinaryClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```


```python
#confusion matrix for TFIDF results
TFIDF_true_positives = TFIDF_results[(TFIDF_results.Label == 1) & (TFIDF_results.prediction == 1)].count()
TFIDF_true_negatives = TFIDF_results[(TFIDF_results.Label == 0) & (TFIDF_results.prediction == 0)].count()
TFIDF_false_positives = TFIDF_results[(TFIDF_results.Label == 0) & (TFIDF_results.prediction == 1)].count()
TFIDF_false_negatives = TFIDF_results[(TFIDF_results.Label == 1) & (TFIDF_results.prediction == 0)].count()
```


```python
#Displaying Confurion matrix of TFIDF
print("TFIDF_true_postives ARE :", TFIDF_true_positives , " and TFIDF_true_negatives ARE :", TFIDF_true_negatives)
print("TFIDF_false_postives ARE :" ,TFIDF_false_positives , " and TFIDF_false_negatives ARE :" ,TFIDF_false_negatives)
```

    TFIDF_true_postives ARE : 960  and TFIDF_true_negatives ARE : 738
    TFIDF_false_postives ARE : 30  and TFIDF_false_negatives ARE : 12



```python
#TFIDF Recall Value
TFIDF_recall = float(TFIDF_true_positives)/(TFIDF_true_positives + TFIDF_false_negatives)
print("TFIDF Recall Value is :" ,TFIDF_recall)
```

    TFIDF Recall Value is : 0.9876543209876543



```python
#TFIDF Precision Value
TFIDF_precision = float(TFIDF_true_positives) / (TFIDF_true_positives + TFIDF_false_positives)
print("TFIDF Precision Value is :" ,TFIDF_precision)
```

    TFIDF Precision Value is : 0.9696969696969697



```python
#TFIDF Accuracy Value
TFIDF_accuracy=float((TFIDF_true_positives+TFIDF_true_negatives) /(TFIDF_results.count()))
print("TFIDF Accuracy Value is :" ,TFIDF_accuracy)
```

    TFIDF Accuracy Value is : 0.9758620689655172



```python
#Precision is about the number of actual positive cases out of all the positive
#cases predicted by the model
#CV Precision Value is : 0.9741035856573705(97%)
#TFIDF Precision Value is : 0.9696969696969697(96%)
```


```python
#Recall:
#It talks about the quality of the machine learning model when it comes
#to predicting a positive class. So out of total positive classes, how many
#was the model able to predict correctly? This metric is widely used as
#evaluation criteria for classification models.
#CV Recall Value is : 0.9898785425101214(98%)
#TFIDF Recall Value is : 0.9876543209876543(98%)
```


```python
#TFIDF Accuracy Value is : 0.9758620689655172(97%)
#Cv Accuracy Value is : 0.9795105293113261(97%)
```


```python
#Even if the accuracy is wrong we can see the models are equally performing well by precision and recall
#Both method are equally accurate
#Where as using TFIDF has a better precision

```
