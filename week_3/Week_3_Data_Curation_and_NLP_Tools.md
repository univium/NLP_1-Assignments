# Week 3 Lab: Data Curation + NLP Tools (NLTK, spaCy, TextBlob)



## Overview
In this combined lab and homework assignment, you will explore why data curation is crucial for NLP, work hands-on with basic curation using real Twitter data, and compare three popular NLP libraries: NLTK, spaCy, and TextBlob. The emphasis is on practical understanding and real-world application.


## Goal
By completing this assignment, you will:
- Understand the role of data curation in building reliable NLP systems
- Explore and clean a real-world sentiment dataset
- Use NLTK, spaCy, and TextBlob for common NLP tasks
- Reflect on dark data and its implications


## Learning Objectives
By the end of this activity, you will be able to:
- Explain why data curation matters and outline its key steps
- Identify common issues in raw text data and suggest curation fixes
- Perform tokenization, part-of-speech tagging, and sentiment analysis using different tools
- Compare the strengths of NLTK, spaCy, and TextBlob
- Discuss dark data in the context of modern NLP applications


## Instructions
Work through the sections below during the lab session using `Week_3_Data_Curation_and_NLP_Tools.ipynb`.
Run the provided code, answer questions in markdown cells, and add your observations.

You will need:
- The provided `train.csv` file (Twitter sentiment dataset)
- Your course environment set up with required packages and models

## Submission
1. Complete all questions and code cells in the notebook.
2. Export as HTML: **File → Download as → HTML (.html)**
3. Submit the HTML file in Canvas by the posted deadline.


## Lab Setup
In this section we’ll confirm imports and required resources so the rest of the notebook runs smoothly.



```python
# Core
import re
import pandas as pd
import numpy as np

```

### Notes on dependencies
- NLTK may require downloaded tokenizers/taggers.
- spaCy requires an English model (e.g., `en_core_web_sm`) installed and loadable.
- TextBlob may require additional corpora depending on features used.



```python
import nltk

# Core requirements (work in all recent NLTK versions)
# Already downloaded if nlp_setup.py has been run in the environment
nltk.download('punkt')                  # For sentence tokenization (legacy compatibility)
nltk.download('punkt_tab')              # New format required for word_tokenize in NLTK 3.9+
nltk.download('averaged_perceptron_tagger')     # Legacy tagger (still useful)
nltk.download('averaged_perceptron_tagger_eng') # Required for default English pos_tag in newer NLTK
```

    [nltk_data] Downloading package punkt to /home/nate/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package punkt_tab to /home/nate/nltk_data...
    [nltk_data]   Package punkt_tab is already up-to-date!
    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     /home/nate/nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!
    [nltk_data] Downloading package averaged_perceptron_tagger_eng to
    [nltk_data]     /home/nate/nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger_eng is already up-to-
    [nltk_data]       date!





    True



With the help of nltk.tokenize.word_tokenize() method, we are able to extract the tokens from string of characters by using tokenize.word_tokenize() method.

You can learn more about it here: https://www.geeksforgeeks.org/python-nltk-nltk-tokenizer-word_tokenize/

Now, the libraries are loaded for you. <b>Tokenize the given paragraph and print the first ten tokens</b> 


```python
from nltk.tokenize import word_tokenize
from nltk.text import Text


my_string = "Two plus two is four, minus one that's three — quick maths. Every day man's on the block. Smoke trees. See your girl in the park, that girl is an uckers. When the thing went quack quack quack, your men were ducking! Hold tight Asznee, my brother. He's got a pumpy. Hold tight my man, my guy. He's got a frisbee. I trap, trap, trap on the phone. Moving that cornflakes, rice crispies. Hold tight my girl Whitney."

#Tokenize the given paragraph and print the first ten tokens
#put your code here
tokens = word_tokenize(my_string)
tokens[:10]
```




    ['Two', 'plus', 'two', 'is', 'four', ',', 'minus', 'one', 'that', "'s"]



<b>The Punkt tokenizer </b>

This tokenizer divides a text into a list of sentences by using an unsupervised algorithm. The NLTK data package includes a pre-trained Punkt tokenizer for English. You can read more about this tokenizer here: https://www.nltk.org/api/nltk.tokenize.html

Let's take a look at another tokenizer: <b>TweetTokenizer</b>. 

TweetTokenizer is a subset of word_tokenize. TweetTokenizer keeps hashtags intact while word_tokenize doesn't.

Use the tweet tokenizer to tokenize the given tweet.


```python
from nltk.tokenize import TweetTokenizer
tweet = "One guess only and no googling. Answer at 10pm. Best of luck. #guessthemysterycelebrity"
#your code here
tknzr = TweetTokenizer()
tknzr.tokenize(tweet)
```




    ['One',
     'guess',
     'only',
     'and',
     'no',
     'googling',
     '.',
     'Answer',
     'at',
     '10pm',
     '.',
     'Best',
     'of',
     'luck',
     '.',
     '#guessthemysterycelebrity']



<b>NLTK Part of Speech Tagging</b>

Part-of-speech tagging is the process of marking up a word in a text as corresponding to a particular part of speech, based on both its definition and its context. For instance: 

I am a boy

Here I is <b>pronoun</b>, am is verb, boy is <b>noun</b>.

You can read more about Part of Speech tagging here: https://www.geeksforgeeks.org/part-speech-tagging-stop-words-using-nltk-python/

Some of the possible pos tags are:

1) CC: conjunction, coordinating

2) DT: determiner

3) NN: noun, common, singular or mass

4) NNS: noun, common, plural

5) PRP: pronoun, personal

6) VB: verb, base form

7) VBD: verb, past tense

8) VBG: verb, present participle or gerund

9) JJS: adjective, superlative etc.




```python
from nltk import pos_tag
from nltk.tokenize import word_tokenize
text = word_tokenize("Roger played tennis all day and became the best")
#use the pos_tag from NLTK to do part of speech tagging
pos_tag(text)
```




    [('Roger', 'NNP'),
     ('played', 'VBD'),
     ('tennis', 'NN'),
     ('all', 'DT'),
     ('day', 'NN'),
     ('and', 'CC'),
     ('became', 'VBD'),
     ('the', 'DT'),
     ('best', 'JJS')]



## Spacy

spaCy is a free open-source library for Natural Language Processing in Python. It features NER, POS tagging, dependency parsing, word vectors and more.

Read more about Spacy here: https://spacy.io/

Import Spacy like this:


```python
import spacy
nlp = spacy.load("en_core_web_sm")
```

We have seen about 2 NLP tools till now: NLTK and Spacy and did some tasks with NLTK. Let's put Spacy into action now. 

<b>Lemmatization</b> 

In grammar, inflection is the modification of a word to express different grammatical categories such as tense, case, voice, aspect, person, number, gender, and mood. 

Lemmatization helps us to achieve the root forms  of inflected (derived) words

![Image](http://kavita-ganesan.com/wp-content/uploads/2019/02/Screen-Shot-2019-02-20-at-4.49.08-PM.png)

More about lemmatization: https://towardsdatascience.com/lemmatization-in-natural-language-processing-nlp-and-machine-learning-a4416f69a7b6

Let's try lemmatization with <b>spacy</b>


```python
doc = nlp("All human beings are born free and equal in dignity and rights. They are endowed with reason and conscience.")#This is called loading a document. You have to use the nlp class
#in spacy, lemma of a word can be printed with lemma_
#write code for printing the lemmatised form of a word alongside each word in doc
for word in doc:
    print(word.text, word.lemma_)

```

    All all
    human human
    beings being
    are be
    born bear
    free free
    and and
    equal equal
    in in
    dignity dignity
    and and
    rights right
    . .
    They they
    are be
    endowed endow
    with with
    reason reason
    and and
    conscience conscience
    . .


Now, let us extract all the adjectives from a piece of text. You already have used part of speech tagging with NLTK. NOw , let's do this with spacy. The syntax is simple for spacy.


```python
doc = nlp("Sachin and cricket")
for word in doc:
    print(word,word.pos_)
```

    Sachin ADJ
    and CCONJ
    cricket NOUN


Your task now is to extract the adjectives from the given piece of text.


```python
doc = nlp("Only these topics: A mathematical puzzle, A biological experiment, A wooden boat.")
#Find the adjectives
#put your code here
adjectives = []
for item in doc:
    if item.pos_ == 'ADJ':
        adjectives.append(item.text)
print(adjectives)
```

    ['mathematical', 'biological', 'wooden']


## TextBlob

TextBlob is an extension of NLTK and is often used for easy NLP tasks. It provides a simple way for diving into common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.

Read more about TextBlob here: https://textblob.readthedocs.io/en/dev/

Import textblob like this:


```python
from textblob import TextBlob
```

We can load any sentence using the textblob class like this:


```python
wiki = TextBlob("Python is a high-level, general-purpose programming language.")
```

Let's do a very common task in the NLP domain called sentiment analysis using TextBlob. Sentiment analysis is a natural language processing technique used to determine whether data is positive, negative or neutral.)

You can learn more about TextBlob here: https://textblob.readthedocs.io/


```python
testimonial = TextBlob("Alexa has been great so far. I am still learning about some the features")
#Analyze the sentiment of the sentence using TextBlob
testimonial.sentiment
```




    Sentiment(polarity=0.45, subjectivity=0.875)



## Curated Datasets

Let's start with the Huggingface dataset. It can found here: https://huggingface.co/datasets/viewer/


Over 135 datasets for many NLP tasks like text classification, question answering, language modeling, etc, are provided on the HuggingFace Hub. Let's load a dataset and see how it looks.



```python
# This may be necessary to clean the dataset cache. ONLY RUN IF NEXT CELL FAILS
import shutil
from datasets import config
from pathlib import Path

cache_dir = Path(config.HF_DATASETS_CACHE)
print("HF cache:", cache_dir)

# Common SQuAD cache locations
candidates = [
    cache_dir / "squad",
    cache_dir / "squad" / "plain_text"
]

for path in candidates:
    if path.exists():
        print("Removing:", path)
        shutil.rmtree(path, ignore_errors=True)

print("Cache cleanup complete. Restart kernel, then reload the dataset.")


```

    HF cache: /home/nate/.cache/huggingface/datasets
    Removing: /home/nate/.cache/huggingface/datasets/squad
    Cache cleanup complete. Restart kernel, then reload the dataset.



```python
from datasets import load_dataset #load_dataset is a function used to load a dataset
datasets = load_dataset('squad')
print(datasets)

```


    Generating train split:   0%|          | 0/87599 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/10570 [00:00<?, ? examples/s]


    DatasetDict({
        train: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 87599
        })
        validation: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 10570
        })
    })


Now, load the 'imdb' dataset from the huggingface repository and:

1) Print the dataset info

2) Print the length of the dataset

3) Print the shape of the dataset

Hint: You can refer to https://huggingface.co/docs/datasets/


```python

from datasets import load_dataset
dataset = load_dataset('imdb')
print(dataset)
print(len(dataset))
print(dataset.shape)
```

    DatasetDict({
        train: Dataset({
            features: ['text', 'label'],
            num_rows: 25000
        })
        test: Dataset({
            features: ['text', 'label'],
            num_rows: 25000
        })
        unsupervised: Dataset({
            features: ['text', 'label'],
            num_rows: 50000
        })
    })
    3
    {'train': (25000, 2), 'test': (25000, 2), 'unsupervised': (50000, 2)}


## Data Visualization

Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, NLP data can be made more understandable. 

We would learn more about visulization in the next week. But, here is a teaser of what is to come.

You just need to run the cells below and see the output.


```python
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import nlplot  # We'll address this below
from plotly.subplots import make_subplots
import plotly.express as px

TRAIN_PATH = r"/home/nate/NextCloud/Roam/Classes/NLP/week_3/data/train (1).csv"

# Load the train.csv file (adjust path if needed)
train = pd.read_csv(TRAIN_PATH)  # Use forward slashes or raw string

# Quick check
print(train.shape)
train.head()
```

    (27481, 4)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>textID</th>
      <th>text</th>
      <th>selected_text</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cb774db0d1</td>
      <td>I`d have responded, if I were going</td>
      <td>I`d have responded, if I were going</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>549e992a42</td>
      <td>Sooo SAD I will miss you here in San Diego!!!</td>
      <td>Sooo SAD</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>088c60f138</td>
      <td>my boss is bullying me...</td>
      <td>bullying me</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9642c003ef</td>
      <td>what interview! leave me alone</td>
      <td>leave me alone</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>358bd9e861</td>
      <td>Sons of ****, why couldn`t they put them on t...</td>
      <td>Sons of ****,</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>
</div>




```python
train = train.sample(n=1000, random_state=0)
# Convert text to lowercase
train['text'] = train['text'].apply(lambda x: x.lower())
display(train.head(), train.shape)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>textID</th>
      <th>text</th>
      <th>selected_text</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20149</th>
      <td>80a1e6bc32</td>
      <td>i just saw a shooting star... i made my wish</td>
      <td>wish</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>12580</th>
      <td>863097735d</td>
      <td>gosh today sucks! i didnt get my tax returns! ...</td>
      <td>gosh today sucks!</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>13135</th>
      <td>264cd5277f</td>
      <td>tired and didn`t really have an exciting satur...</td>
      <td>tired and didn`t really have an exciting Satur...</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>14012</th>
      <td>baee1e6ffc</td>
      <td>i`ve been eating cheetos all morning..</td>
      <td>i`ve been eating cheetos all morning..</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>21069</th>
      <td>67d06a8dee</td>
      <td>haiiii sankq i`m fineee ima js get a checkup ...</td>
      <td>haiiii sankQ i`m fineee ima js get a checkup c...</td>
      <td>neutral</td>
    </tr>
  </tbody>
</table>
</div>



    (1000, 4)



```python
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

# Plotly renderer fix for Jupyter environments
pio.renderers.default = "notebook_connected"

# Load data
train = pd.read_csv(TRAIN_PATH)

# Create count plot
df = train.groupby('sentiment').size().reset_index(name='count')

fig = px.bar(
    df,
    x='sentiment',
    y='count',
    text='count',
    title='Sentiment Counts',
    labels={'sentiment': 'Sentiment', 'count': 'Number of Tweets'}
)

fig.update_traces(texttemplate='%{text}', textposition='outside')
fig.update_layout(width=700, height=500)

fig.show()

```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
</script>
<script type="module">import "https://cdn.plot.ly/plotly-3.3.0.min"</script>




<div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.3.0.min.js" integrity="sha256-bO3dS6yCpk9aK4gUpNELtCiDeSYvGYnK7jFI58NQnHI=" crossorigin="anonymous"></script>                <div id="71515ff0-432f-456d-987e-1591eeedfc77" class="plotly-graph-div" style="height:500px; width:700px;"></div>            <script type="text/javascript">                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("71515ff0-432f-456d-987e-1591eeedfc77")) {                    Plotly.newPlot(                        "71515ff0-432f-456d-987e-1591eeedfc77",                        [{"hovertemplate":"Sentiment=%{x}\u003cbr\u003eNumber of Tweets=%{text}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"","orientation":"v","showlegend":false,"text":{"dtype":"f8","bdata":"AAAAAABlvkAAAAAAALfFQAAAAAAAw8BA"},"textposition":"outside","x":["negative","neutral","positive"],"xaxis":"x","y":{"dtype":"i2","bdata":"ZR5uK4Yh"},"yaxis":"y","type":"bar","texttemplate":"%{text}"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermap":[{"type":"scattermap","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Sentiment"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Number of Tweets"}},"legend":{"tracegroupgap":0},"title":{"text":"Sentiment Counts"},"barmode":"relative","width":700,"height":500},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('71515ff0-432f-456d-987e-1591eeedfc77');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };            </script>        </div>


### May need to install nlplot
pip install nlplot


```python
import nlplot

# Initialize NLPlot with text column
npt = nlplot.NLPlot(train, target_col='text')

stopwords = npt.get_stopword(top_n=30, min_freq=0)

npt.bar_ngram(
    title='uni-gram',
    xaxis_label='word_count',
    yaxis_label='word',
    ngram=1,
    top_n=50,
    width=800,
    height=1100,
    stopwords=stopwords,
)
```

    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27480/27480 [00:00<00:00, 95766.57it/s]



<div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.3.0.min.js" integrity="sha256-bO3dS6yCpk9aK4gUpNELtCiDeSYvGYnK7jFI58NQnHI=" crossorigin="anonymous"></script>                <div id="d144d1f8-f6ad-4353-a5c0-70f474c9fe39" class="plotly-graph-div" style="height:1100px; width:800px;"></div>            <script type="text/javascript">                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("d144d1f8-f6ad-4353-a5c0-70f474c9fe39")) {                    Plotly.newPlot(                        "d144d1f8-f6ad-4353-a5c0-70f474c9fe39",                        [{"hovertemplate":"word_count=%{text}\u003cbr\u003eword=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"pattern":{"shape":""}},"name":"","orientation":"h","showlegend":false,"text":{"dtype":"f8","bdata":"AAAAAACAhEAAAAAAAKCEQAAAAAAAqIRAAAAAAAAIhUAAAAAAABiFQAAAAAAAOIVAAAAAAAC4hUAAAAAAAMiFQAAAAAAAqIZAAAAAAAAYh0AAAAAAAFCHQAAAAAAAiIdAAAAAAACoh0AAAAAAANCHQAAAAAAA6IdAAAAAAAAoiEAAAAAAAGiIQAAAAAAAeIhAAAAAAADAiEAAAAAAAOCIQAAAAAAAYIlAAAAAAACoiUAAAAAAALiJQAAAAAAAwIlAAAAAAADAiUAAAAAAADCKQAAAAAAASIpAAAAAAACAikAAAAAAAICKQAAAAAAAkIpAAAAAAABIi0AAAAAAAHCLQAAAAAAAkIxAAAAAAACYjkAAAAAAAFCPQAAAAAAAkJBAAAAAAACYkEAAAAAAALSQQAAAAAAA4JBAAAAAAADskUAAAAAAACiSQAAAAAAAKJJAAAAAAABYkkAAAAAAAGiSQAAAAAAA0JJAAAAAAABYlEAAAAAAAFyUQAAAAAAAtJRAAAAAAAAkl0AAAAAAAPqgQA=="},"textposition":"auto","x":{"dtype":"i2","bdata":"kAKUApUCoQKjAqcCtwK5AtUC4wLqAvEC9QL6Av0CBQMNAw8DGAMcAywDNQM3AzgDOANGA0kDUANQA1IDaQNuA5ID0wPqAyQEJgQtBDgEewSKBIoElgSaBLQEFgUXBS0FyQV9CA=="},"xaxis":"x","y":["think","how","still","when","&","as","want","lol","new","time","can`t","am","know","if","see","some","back","can","too","****","its","we","had","about","im","one","really","u","what","don`t","will","work","happy","from","do","love","going","got","now","-","go","your","it`s","no","up","out","like","good","day","i`m"],"yaxis":"y","type":"bar","texttemplate":"%{text:.2s}"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermap":[{"type":"scattermap","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"word_count"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"word"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"barmode":"relative","title":{"text":"uni-gram"},"width":800,"height":1100},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('d144d1f8-f6ad-4353-a5c0-70f474c9fe39');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };            </script>        </div>


## Part 1: Why Data Curation Matters (15 points)
In markdown cells, complete the following:

    1. In your own words (3–5 sentences), explain why data curation is essential for NLP applications. Consider noise, bias, model performance, and real-world text variability.

    2. List the main steps of data curation in NLP (in your own words; bullet points are fine).

#### 1. I think 'essential' is like 'important' where it implies a kind of air of voluntaryness: that isn't what this Data Curation is.
Data Curation is the very first spot at looking into the data more deeply, seeking if a hypothsis is true from piror, rough analytics. Such piror analytics often sorts though near raw data that, like the quesiton brings up, is often noisey, has possible bias in collection, etc; data curation is the process of removing those things way or either coping with it's natural existence. 

#### 2. The main steps of data cureation in NLP are,
- Data Aquisition & Collection
  Grather of raw text for the given project. This step will often include that just simply HTML from webpages but could also include with it JSON, PDFs, etc.
- Text Extraction & Normalization
  Extraction can be understood as getting the desired text out of some collection of raw dat, like that of text on a website without the advert's. Normalization is changing the extraxted data in some way so as to standarize it's consumption; this can include using a UTF-8 to convert the text into a uniform format.
- Quality Filtering
  Where the notion of 'cleaning' takes place. Extracted & uniform data may still contain text that may be demend 'low-quility' or either illrlevent. Such information can be filtered out by using tequnices like that of Heuristic Filters.
- Data Anotation & Labeling
  Per in the name such a process is using different methods, like manual annotation or synthetic data via LLMs, to create the ground truth for the model, enabling a desired classification. 
- Tokenization & preprocessing
  Tokenization is converting text into smaller units while preprocessing can be vocabulary building or applying templates for tokens based off the foundation model.
- Dataset Analysis & Documentation
  With the model created the model itself can be used for dataset analysis checking for topic diversity, sequence length, and so on.

# Part 2: Hands-On with a Real Dataset (30 points)
Load and explore the Twitter sentiment dataset.

import pandas as pd 
df = pd.read_csv('train.csv') 
print(df.shape) df.head(10)

Answer the following in markdown cells:

    1. How many tweets are in the dataset? What are the three sentiment labels and their approximate distribution?

    2. Show five random positive, five random negative, and five random neutral tweets.

        Example approach: df[df.sentiment == 'positive'].sample(5)

    3. Identify at least five different data quality issues visible in the raw text (examples may include URLs, emojis, abbreviations, mixed casing, extra punctuation). For each issue:

        Copy and paste one example tweet

    4. For each issue you identified, suggest one practical way to fix or mitigate it during curation (for example: lowercase text, remove URLs using regex, expand contractions).


```python
#2
import pandas as pd

df=pd.read_csv('/home/nate/NextCloud/Roam/Classes/NLP/week_3/data/train (1).csv')

categories = ['positive', 'neutral', 'negative']
all_samples = []

for cat in categories:
    print(df[df.sentiment == cat].sample(5))
    
```

               textID                                               text  \
    26374  47fad4f4cd  had the BEST Italian meal EVER last night! twa...   
    5979   0834346d7b                        good luck with your auction   
    21731  f75795c56d                       Baking WIN! Thanks for that!   
    21469  be6c44f4e4  Watched Australia last night and got to say bl...   
    17667  195c550682  what a great day for a massage! book your appo...   
    
                                selected_text sentiment  
    26374                                BEST  positive  
    5979          good luck with your auction  positive  
    21731                              Thanks  positive  
    21469           say bloody fantastic film  positive  
    17667  what a great day for a massage! bo  positive  
               textID                                               text  \
    1953   8ea8d240e3  Paid bills. We get water and electricity for a...   
    7638   817eaddad2  Wine..beer..and champagne..lets see how that g...   
    11907  b97a3861c5           DUCKED OFF! I`LL BE BACK IN GA TOMORROW.   
    4594   f4b44547ce  _Kay Morning! Hows your day. Hope you`re not a...   
    7566   121b097b0b  Tidied & hoovered the whole flat - and all b4 ...   
    
                                               selected_text sentiment  
    1953   Paid bills. We get water and electricity for a...   neutral  
    7638   Wine..beer..and champagne..lets see how that g...   neutral  
    11907           DUCKED OFF! I`LL BE BACK IN GA TOMORROW.   neutral  
    4594   Morning! Hows your day. Hope you`re not anothe...   neutral  
    7566   Tidied & hoovered the whole flat - and all b4 ...   neutral  
               textID                                               text  \
    2090   411a54feb1  painting my nails green in an attempt to look ...   
    7973   3b9fe0f1e2                               Iï¿½m sorry for that   
    10608  5e407e618c  One is not supposed to have a headache on a Fr...   
    18182  1538798e8c  Nnnnoooooo!!!! Just learned we`ve got a frost ...   
    17780  c31448b37c     http://bit.ly/5pBLz  for McCoy`s initial rant.   
    
                                               selected_text sentiment  
    2090   annoyed that everyone seems to tan apart from ...  negative  
    7973                                Iï¿½m sorry for that  negative  
    10608                             That`s just not right.  negative  
    18182                                    a frost warning  negative  
    17780                                      initial rant.  negative  


#### 1. In total there 27481 tweets witin the dataset; their sentiment labels are positive, negative, and neutral, and their approximate distribution is normal due to a similer count between positive & negative tweets.

#### 2. Shown in code cell above.

#### 3. Some of the data quality issues I see are,
- Abbreviations, like that of "FYI" & etc.
- URLs like that witin the tweet "1587   3b50246867  Link per my daughter  us.mobile.reuters.com/mo..."
- Grammar or spelling errors.
- Mixed punctuation within the same sentence.
- Someone's address is contained within the dataset.

#### 4. A few things that I could've done was to use REGEX to remove anything of the form of a personal address; another could be using a lightweight LLM for fixing spelling or expanding contractions if such information is needed.
Regardless of the tool use to symantically clean data, it would still be the case that an overview of changes & if they were correct would be in order.

# Part 3: Comparing NLP Tools in Action (35 points)
You will process the same sample tweet with each library.





```python
sample = "I luv my iPhone!!! 😍 But sometimes it's super slow... #Apple"
```


```python
# NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

tokens_nltk = word_tokenize(sample.lower())
print("Tokens:", tokens_nltk)
print("POS tags:", pos_tag(tokens_nltk))

```

    Tokens: ['i', 'luv', 'my', 'iphone', '!', '!', '!', '😍', 'but', 'sometimes', 'it', "'s", 'super', 'slow', '...', '#', 'apple']
    POS tags: [('i', 'NN'), ('luv', 'VBP'), ('my', 'PRP$'), ('iphone', 'NN'), ('!', '.'), ('!', '.'), ('!', '.'), ('😍', 'NN'), ('but', 'CC'), ('sometimes', 'VBZ'), ('it', 'PRP'), ("'s", 'VBZ'), ('super', 'JJ'), ('slow', 'NN'), ('...', ':'), ('#', '#'), ('apple', 'NN')]



```python
#spaCy
import spacy 
nlp = spacy.load("en_core_web_sm") 
doc = nlp(sample) 
print("Tokens:", [token.text for token in doc]) 
print("POS tags:", [(token.text, token.pos_) for token in doc])
```

    Tokens: ['I', 'luv', 'my', 'iPhone', '!', '!', '!', '😍', 'But', 'sometimes', 'it', "'s", 'super', 'slow', '...', '#', 'Apple']
    POS tags: [('I', 'PRON'), ('luv', 'NOUN'), ('my', 'PRON'), ('iPhone', 'PROPN'), ('!', 'PUNCT'), ('!', 'PUNCT'), ('!', 'PUNCT'), ('😍', 'PROPN'), ('But', 'CCONJ'), ('sometimes', 'ADV'), ('it', 'PRON'), ("'s", 'AUX'), ('super', 'ADV'), ('slow', 'ADJ'), ('...', 'PUNCT'), ('#', 'SYM'), ('Apple', 'NOUN')]



```python
#TextBlob

from textblob import TextBlob
blob = TextBlob(sample) 
print("Spelling correction:", blob.correct()) 
print("Sentiment:", blob.sentiment) 
# polarity (-1 to 1), subjectivity (0 to 1)
```

    Spelling correction: I lui my shone!!! 😍 But sometimes it's super slow... #Apple
    Sentiment: Sentiment(polarity=0.016666666666666635, subjectivity=0.5333333333333333)


### Now answer the following in markdown cells:

    1. How did tokenization differ between the tools? Which seemed to handle emojis and punctuation better?


    2. Compare POS tagging quality and level of detail (for example: coarse vs. fine-grained tags).


    3. TextBlob provides sentiment and spelling correction out of the box. What does this suggest about its design focus?

    4. Based on your observations, when would you choose NLTK, spaCy, or TextBlob for a project? Provide a brief reason for each.


#### 1.

This question results in the topic of model interpretability as it will be shown that TextBlob suffers from it slightly. To walk back for a moment, I would remark that spaCy seems to handle emojis & punctuation a bit better than NLTK; the latter uses a default word type entry for emojis --unlike spaCy which classifies them as symbols and punctuation --while including possibly redundant information on the use of three periods for a pause. TextBlob on the other hand doesn't show outright how it might be interpreting the emoji; I could go so far as to say it could not due to it giving the tweet such a low polarity.

#### 2.

If it were to be desired to have such information like that of a mark of silence '...' then NLTK could be considered as having a greater amount of detail than spaCy. If it is quality, I would go with spaCy due to the classification of emojis and intuitive tags.

#### 3.

That overall it's for quick sentiment analysis on a given dataset: more focus should be put on 'quick' here; I think its use is to be what you would run before constructing your own.

#### 4.

I think it always rests on the use case, but I'm leaning towards spaCy more than NLTK due to the reasons of intuitive use and possible classification of other symbols of meaning like emojis. TextBlob is something I would use as well, but there my intended goal might be different.

