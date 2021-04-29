# Report

## 1. Evaluate spaCy NER on CoNLL 2003 data
In order to do evaluation on spaCy NER it is necessary first to load the CoNLL 2003 data. An easy way to do it is to exploit the `read_corpus_conll('data/conll2003/train.txt', ' ')` function provided in the conll.py file. This function returns a list of lists containing the word with some additional information as POS tag, syntactic chunk tag and the named entity tag.

Here an example of the data format:
```
Only RB B-NP O
France NNP I-NP B-LOC
and CC I-NP O
Britain NNP I-NP B-LOC
backed VBD B-VP O
Fischler NNP B-NP B-PER
's POS B-NP O
proposal NN I-NP O
. . O O
```

Then the doc object is created by passing the words to the Doc constructor as in this function `doc = Doc(nlp.vocab, words=[w[0] for w in sent])`. Since the objective is to evaluate the performances of spacy pretrained model the words need to be processed by the pipeline:
```python
for name, proc in nlp.pipeline:
            doc = proc(doc)
```

### report token-level performance (per class and total)
In order to evaluate SpaCy model a list of references and a list of hypothesis is used to get the accuracy.
One important remark is that CoNLL NE tags and SpaCy tags are different and need to be converted. To do that a dictionary containing the mappings is used.
The accuracy is obtained through the function `classAndTotAccuracy` by passing it the hypothesis and the references.
The function simply count the times in which the two tag are equal and subdivide it by the total number of encountered tag.

The results are reported below:
```python
TOTAL:  0.800
O:      0.953
B-ORG:  0.457
B-MISC: 0.095
B-PER:  0.794
I-PER:  0.815
I-ORG:  0.476
I-MISC: 0.038
B-LOC:  0.632
I-LOC:  0.457
```

### report CoNLL chunk-level performance (per class and total)
To evaluate the chunk-level performances the provided `evaluate()` function in the conll.py file has been used.

The results are reported below:
```python
              p         r         f      s
LOC    0.626794  0.018347  0.035651   7140
MISC   0.088448  0.551193  0.152435   3438
PER    0.770980  0.631970  0.694588   6600
ORG    0.407710  0.284449  0.335104   6321
total  0.254181  0.340227  0.290976  23499
```

## 2. Grouping of Entities

The `doc.noun_chunks` attribute returns (generator object of) a list of spans, the chunks. Instead `doc.ents` attributes return the entities of the doc object. Unfortunately, the ents returned by the chunks are not the same of the ones that returns `doc.ents`. So a function to group them is needed.

First, the entities of the chunks are extracted and the entities deriving from `doc.ents` are put into a list word by word.

Then the list that will contain the grouped entities is created copying the list of the entities derived by `doc.noun_chunks`.

Afterwards, if the single entity is not present in the grouped list, it is added at the right index, preserving ordering.

It can happen that this function provides empty lists that are filtered with these lines of code:
```python
for doc in docList:
    rawEntityGroups = extractGroupEntity(doc)
    filteredEntityGroups.append([x for x in rawEntityGroups if x])
```

In this way it will be obtained a list of lists of entities, where entities are grouped by chunks.

A frequency analysis over the obtained list has been made. It simply counts how many times a particular group is present. If two groups have the same labels but inverted they are considered different groups.

A part of the results is reported below:
```python
-GPE-: 5485
-CARDINAL-: 4422
-PERSON-: 4404
-ORG-: 3649
-DATE-: 3570
-NORP-: 1467
-ORDINAL-: 658
-MONEY-: 398
-QUANTITY-: 330
-TIME-: 325
-CARDINAL-PERSON-: 310
-PERCENT-: 287
-NORP-PERSON-: 197
-LOC-: 174
-FAC-: 123
-GPE-PERSON-: 119
-EVENT-: 110
-CARDINAL-ORG-: 107
-ORG-PERSON-: 93
-PRODUCT-: 79
-CARDINAL-NORP-: 78
-GPE-ORG-: 58
-CARDINAL-GPE-: 53
-GPE-GPE-: 45
-DATE-TIME-: 44
-WORK_OF_ART-: 40
-ORG-ORG-: 38
-PERSON-PERSON-: 33
-LANGUAGE-: 29
[...]
-ORG-CARDINAL-CARDINAL-ORG-: 1
-CARDINAL-GPE-PERSON-: 1
-CARDINAL-PERSON-GPE-CARDINAL-: 1
-ORDINAL-CARDINAL-CARDINAL-: 1
```

## 3. Fix segmentation errors

This part of the code aims at improving the segmentation by using the `'compound'` dependency relation to extend the entity span to cover the full noun-compounds.

First, the segmentation given by spaCy for the entire sentence is obtained in the form `('text','iob-ent_type')`, for example: `('Apple', 'B-ORG')`, and stored in a list.

Then, if `token.dep_ == 'compound'` the children and the entity type of the `token.head` are obtained through the function `childrenOfCompound(token)`. Afterwards, the code cycle through all the children and assign them the right IOB and entity type if their dependency relation is `'compound'`. If not, the code checks if the dependency relation is unrelevant for the entity span and put `'O'` as entity relation. Two cases that I considered irrelevant basing on the annotation on the dataset are `det` and `case`.

Since it can happen that the `token.head.dep_` is `'compound'` a list called `alreadyProcessedIndexList` is used to take track of the yet processed tokens.

For example in the sentence:
`He said a proposal last month by EU Farm Commissioner Franz Fischler to ban sheep brains`
"EU.dep_" is compound. The "EU.head" is Commissioner that is itself a compound of Fischler.
If such list didn't exist the algorithm would have modified first `EU Farm Commissioner Franz Fischler` and then override the entities of `Commissioner Franz Fischler`.
So the output would have been:
``` python
[('He', 'O'), ('said', 'O'), ('a', 'O'), ('proposal', 'O'), ('last', 'B-DATE'), ('month', 'I-DATE'), ('by', 'O'), ('EU', 'B-PERSON'), ('Farm', 'I-PERSON'), ('Commissioner', 'B-PERSON'), ('Franz', 'I-PERSON'), ('Fischler', 'I-PERSON'), ('to', 'O'), ('ban', 'O'), ('sheep', 'O'), ('brains', 'O')]
```
instead of:
``` python
[('He', 'O'), ('said', 'O'), ('a', 'O'), ('proposal', 'O'), ('last', 'B-DATE'), ('month', 'I-DATE'), ('by', 'O'), ('EU', 'B-PERSON'), ('Farm', 'I-PERSON'), ('Commissioner', 'I-PERSON'), ('Franz', 'I-PERSON'), ('Fischler', 'I-PERSON'), ('to', 'O'), ('ban', 'O'), ('sheep', 'O'), ('brains', 'O')]
```

Then, the evaluation of the results is computed exploiting the conll.py's `evaluation()` function. As it can be seen the performances obtained are worst than before.

```python
              p         r         f   s
PER    0.333333  0.416667  0.370370  12
LOC    1.000000  0.071429  0.133333  14
MISC   0.285714  0.714286  0.408163  14
ORG    0.600000  0.750000  0.666667  12
total  0.378788  0.480769  0.423729  52
```
### Personal interpretation of the results

I think that this loss of performances is not caused by the algorithm. In fact there are some issues with some particular sentences that may disrupt the results.
Here are reported two of them:
```
He said a proposal last month by EU Farm Commissioner Franz Fischler to ban sheep brains

Germany's representative to the European Union's veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer.
```

In the first one the issue is rapresented by the part `EU Farm Commissioner Franz Fischler`. Here the algorithm consider this part a person as a whole by exploiting the `compound` relation. 
```python
('EU', 'B-PERSON'), ('Farm', 'I-PERSON'), ('Commissioner', 'I-PERSON'), ('Franz', 'I-PERSON'), ('Fischler', 'I-PERSON')
```
Insead the dataset consider the following types leaving a discrepancy in the results.
```python
('EU', 'B-ORG'), ('Farm', 'O'), ('Commissioner', 'O'), ('Franz', 'B-PERSON'), ('Fischler', 'I-PERSON')
```

In the second sentence the issue arise with the part `committee Werner Zwingmann`. Here spacy states that `committee` has a `compound` relation, thus the code convert the piece of sentence in: 
```python
('committee', 'B-PERSON'), ('Werner', 'I-PERSON'), ('Zwingmann', 'I-PERSON')
```
Instead the dataset do not consider `committee` a `'B-PERSON'` leaving a difference in the results. In this case the problem is systematic and is caused by spaCy.

