
# Table of Contents

1.  [Vocabulary](#org876f90f)
2.  [Task 1: Bag of Words Table](#org72e7be3)
3.  [Task 2: Document Frequency (DF) and IDF](#org2d338e7)
4.  [Task 2: TF-IDF Table](#org959ec97)
    1.  [Notes](#orge1eede8)
    2.  [Answer each question in 3–5 sentences](#orgd2e8611)
        1.  [Which word has the highest TF-IDF score in Sentence 4, and why does it have such a high value?](#orga12bc51)
        2.  [Why does the very common word “is” have a low TF-IDF score across most sentences?](#orgda1a606)
        3.  [How does TF-IDF improve upon simple Bag of Words when representing document importance?](#orgf24219e)



<a id="org876f90f"></a>

# Vocabulary

a, ago, from, improve, is, long, more, novel, novels, read, should, the, this, time, to, victorian, vocabulary, you, your

Total unique words: 19


<a id="org72e7be3"></a>

# Task 1: Bag of Words Table

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Sentence</th>
<th scope="col" class="org-right">a</th>
<th scope="col" class="org-right">ago</th>
<th scope="col" class="org-right">from</th>
<th scope="col" class="org-right">improve</th>
<th scope="col" class="org-right">is</th>
<th scope="col" class="org-right">long</th>
<th scope="col" class="org-right">more</th>
<th scope="col" class="org-right">novel</th>
<th scope="col" class="org-right">novels</th>
<th scope="col" class="org-right">read</th>
<th scope="col" class="org-right">should</th>
<th scope="col" class="org-right">the</th>
<th scope="col" class="org-right">this</th>
<th scope="col" class="org-right">time</th>
<th scope="col" class="org-right">to</th>
<th scope="col" class="org-right">victorian</th>
<th scope="col" class="org-right">vocabulary</th>
<th scope="col" class="org-right">you</th>
<th scope="col" class="org-right">your</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">S1</td>
<td class="org-right">1</td>
<td class="org-right">1</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
</tr>

<tr>
<td class="org-left">S2</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
</tr>

<tr>
<td class="org-left">S3</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
<td class="org-right">1</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
</tr>

<tr>
<td class="org-left">S4</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">0</td>
<td class="org-right">2</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
<td class="org-right">0</td>
<td class="org-right">1</td>
</tr>
</tbody>
</table>


<a id="org2d338e7"></a>

# Task 2: Document Frequency (DF) and IDF

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Word</th>
<th scope="col" class="org-right">DF</th>
<th scope="col" class="org-left">Appears in</th>
<th scope="col" class="org-right">IDF \(\frac{(log_{10}}{(4/DF))}\)</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">a</td>
<td class="org-right">1</td>
<td class="org-left">S1</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">ago</td>
<td class="org-right">1</td>
<td class="org-left">S1</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">from</td>
<td class="org-right">1</td>
<td class="org-left">S1</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">improve</td>
<td class="org-right">1</td>
<td class="org-left">S4</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">is</td>
<td class="org-right">3</td>
<td class="org-left">S1 S2 S4</td>
<td class="org-right">0.125</td>
</tr>

<tr>
<td class="org-left">long</td>
<td class="org-right">1</td>
<td class="org-left">S1</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">more</td>
<td class="org-right">1</td>
<td class="org-left">S3</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">novel</td>
<td class="org-right">2</td>
<td class="org-left">S1 S2</td>
<td class="org-right">0.301</td>
</tr>

<tr>
<td class="org-left">novels</td>
<td class="org-right">2</td>
<td class="org-left">S3 S4</td>
<td class="org-right">0.301</td>
</tr>

<tr>
<td class="org-left">read</td>
<td class="org-right">2</td>
<td class="org-left">S3 S4</td>
<td class="org-right">0.301</td>
</tr>

<tr>
<td class="org-left">should</td>
<td class="org-right">1</td>
<td class="org-left">S3</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">the</td>
<td class="org-right">1</td>
<td class="org-left">S2</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">this</td>
<td class="org-right">1</td>
<td class="org-left">S1</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">time</td>
<td class="org-right">1</td>
<td class="org-left">S1</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">to</td>
<td class="org-right">1</td>
<td class="org-left">S4</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">victorian</td>
<td class="org-right">1</td>
<td class="org-left">S2</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">vocabulary</td>
<td class="org-right">1</td>
<td class="org-left">S4</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">you</td>
<td class="org-right">1</td>
<td class="org-left">S3</td>
<td class="org-right">0.602</td>
</tr>

<tr>
<td class="org-left">your</td>
<td class="org-right">1</td>
<td class="org-left">S4</td>
<td class="org-right">0.602</td>
</tr>
</tbody>
</table>


<a id="org959ec97"></a>

# Task 2: TF-IDF Table

(Rounded to 3 decimal places)

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Sentence</th>
<th scope="col" class="org-right">a</th>
<th scope="col" class="org-right">ago</th>
<th scope="col" class="org-right">from</th>
<th scope="col" class="org-right">improve</th>
<th scope="col" class="org-right">is</th>
<th scope="col" class="org-right">long</th>
<th scope="col" class="org-right">more</th>
<th scope="col" class="org-right">novel</th>
<th scope="col" class="org-right">novels</th>
<th scope="col" class="org-right">read</th>
<th scope="col" class="org-right">should</th>
<th scope="col" class="org-right">the</th>
<th scope="col" class="org-right">this</th>
<th scope="col" class="org-right">time</th>
<th scope="col" class="org-right">to</th>
<th scope="col" class="org-right">victorian</th>
<th scope="col" class="org-right">vocabulary</th>
<th scope="col" class="org-right">you</th>
<th scope="col" class="org-right">your</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">S1</td>
<td class="org-right">0.602</td>
<td class="org-right">0.602</td>
<td class="org-right">0.602</td>
<td class="org-right">0.000</td>
<td class="org-right">0.125</td>
<td class="org-right">0.602</td>
<td class="org-right">0.000</td>
<td class="org-right">0.301</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.602</td>
<td class="org-right">0.602</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
</tr>

<tr>
<td class="org-left">S2</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.125</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.301</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.602</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.602</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
</tr>

<tr>
<td class="org-left">S3</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.602</td>
<td class="org-right">0.000</td>
<td class="org-right">0.301</td>
<td class="org-right">0.301</td>
<td class="org-right">0.602</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.602</td>
<td class="org-right">0.000</td>
</tr>

<tr>
<td class="org-left">S4</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.602</td>
<td class="org-right">0.125</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.301</td>
<td class="org-right">0.301</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">0.000</td>
<td class="org-right">1.204</td>
<td class="org-right">0.000</td>
<td class="org-right">0.602</td>
<td class="org-right">0.000</td>
<td class="org-right">0.602</td>
</tr>
</tbody>
</table>


<a id="orge1eede8"></a>

## Notes

-   It seems that documents are understood to be as sentences too.
-   The result of the TF-IDF table seems to suggest a kind of matrix is created between the two features of the sentences or documents.
    I say feaures because it seems that Document Frequency (DF) and Inverse Document Frequency (IDF) are two different features of documents, where TF-IDF is just the ratio of such documents over the document feq of a word, all as an image of a log in base 10.
-   TF = raw term frequency (from BoW table)
-   IDF = log₁₀(4 / DF)
-   TF-IDF = TF × IDF


<a id="orgd2e8611"></a>

## Answer each question in 3–5 sentences


<a id="orga12bc51"></a>

### Which word has the highest TF-IDF score in Sentence 4, and why does it have such a high value?

The word within document four which has the highest value is &rsquo;to&rsquo;. Without looking into why just yet, I know that by the TF-IDF expression $TF-IDF = TF * IDF$  means any large vaule would be either connected to the Term Frequency of that word or Inverse Document Frequency; looking into the IDF of the word &rsquo;to&rsquo; I find that it was on the high end, meaning it&rsquo;s Document Frequency was small & only appeard in one document; looking into the table for Bag of Words I find too that though it appeared twice it was only in it&rsquo;s sentence rather then distributed like most others were. This would amount to it&rsquo;s expression being $TF-IDF = 2 * 0.602 = 1.204$, which can now be seen as having the largest TF while one of the higest IDF; so, I would say that the term within the expression that explains it&rsquo;s vaule is that of it&rsquo;s TF or rather the raw cell count in the Bag of Words table; but why is it&rsquo;s attribute of having the largest occuernce in a single document rather then a couple arises this properity of it?     

This properity I&rsquo;m discriping above is central function of TF-IDF; it&rsquo;s to weight or reward words which are repeated a lot in a single document as it&rsquo;s supposed that such an attribute of a word means it&rsquo;s thematically important to that document. This is why it&rsquo;s raw term frequency is used while being a mutiple of it&rsquo;s IDF, being a weight of the whole corpus for the rareity of words while penalizing ones more frequent. 


<a id="orgda1a606"></a>

### Why does the very common word “is” have a low TF-IDF score across most sentences?

Much like the prior analysis for the word &rsquo;to&rsquo; the reason for it&rsquo;s low TF-IDF score is due to it&rsquo;s distribution accross the documents(sentences). As I noted before, the factor for TF-IDF is that of IDF which will penalize terms which appear frequentlly accross the documnts of the corpus. This can be seen in this case by $..log<sub>10</sub>\frac{N}{DF}$ where N=4(the documents) & DF=3(Doc Freq). 


<a id="orgf24219e"></a>

### How does TF-IDF improve upon simple Bag of Words when representing document importance?

Unlike the raw amount of term&rsquo;s apperence within the given documents give by the Bag of Words algorithm, TF-IDF generates a value representive of it&rsquo;s apperence relative to it&rsquo;s rarity which given by the IDF of the word.

