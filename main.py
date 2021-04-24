import spacy
from spacy.tokens import Doc
from conll import evaluate, read_corpus_conll, get_chunks, parse_iob
import pandas as pd

spacyToConllMap = {
    # https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
    # https://spacy.io/api/annotation#named-entities
    "PERSON": "PER",
    "NORP": "MISC",
    "FACILITY": "ORG",
    "FAC": "ORG",
    "ORG": "ORG",
    "GPE": "MISC",
    "LOC": "LOC",
    "PRODUCT": "MISC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC",
    "DATE": "MISC",
    "TIME": "MISC",
    "PERCENT": "MISC",
    "MONEY": "MISC",
    "QUANTITY": "MISC",
    "ORDINAL": "MISC",
    "CARDINAL": "MISC",
    "PER": "MISC",
    "MISC": "MISC",
    "EVT": "MISC",
    "PROD": "MISC",
    "DRV": "MISC",
    "GPE_LOC": "LOC",
    "GPE_ORG": "ORG",
    "": ""
}

nlp = spacy.load('en_core_web_sm')

trainData = read_corpus_conll('data/conll2003/train.txt', ' ')[:500]
testData = read_corpus_conll('data/conll2003/test.txt', ' ')[:500]

# https://spacy.io/usage/processing-pipelines
docList = []

for sent in trainData:
    doc = Doc(nlp.vocab, words=[w[0] for w in sent])
    for name, proc in nlp.pipeline:
        doc = proc(doc)
    docList.append(doc)

refs = [[(text, iob) for text, pos, syntChunck, iob in sent]
        for sent in trainData]
hyps = []

for doc in docList:
    tmpList = []
    for token in doc:
        tmp = token.ent_iob_
        if token.ent_iob_ != 'O':
            tmp += '-' + spacyToConllMap[token.ent_type_]
        tmpList.append((token.text, tmp))
    hyps.append(tmpList)

# possible ent_type of Conll
entitySet = get_chunks('data/conll2003/train.txt', fs=' ')

countDict = {}
for entity in entitySet:
    countDict[entity] = [0, 0]

# 1.1
if len(hyps) != len(refs):
    raise ValueError(
        "Size Mismatch: ref: {} & hyp: {}".format(len(ref), len(hyp)))
else:
    count = 0
    total = 0
    for i, sent in enumerate(hyps):
        for j, tHyps in enumerate(sent):
            key = parse_iob(tHyps[1])[1]
            total += 1
            if tHyps == refs[i][j]:
                count += 1
            if key in countDict:
                countDict[key][1] += 1
                if key == parse_iob(refs[i][j][1])[1]:
                    countDict[key][0] += 1

    totAccuracy = count/total
    print("Question 1.1: Evaluate accuracy\n")
    for entity in entitySet:
        print("{}:\t{}".format(
            entity, countDict[entity][0]/countDict[entity][1]))
    print("Total:\t{}\n".format(totAccuracy))

# 1.2
results = evaluate(refs, hyps)

pd_tbl = pd.DataFrame().from_dict(results, orient='index')
pd_tbl.round(decimals=3)
print("Question 1.2: Evaluate chunk-level performances\n")
print('{}\n'.format(pd_tbl))

# 2
print("Question 2: Group entities\n")

globalGroupEntities = []
for doc in docList:
    sentence = "Apple's Steve Jobs died in 2011 in Palo Alto, California."

    # sen = nlp(sentence)
    nounChunkList = list(doc.noun_chunks)
    # [Apple's Steve Jobs, Palo Alto, California]
    entityList = doc.ents
    # (Apple, Steve Jobs, 2011, Palo Alto, California)

    nounChunkEnts = [span.ents for span in doc.noun_chunks]
    # [[Apple, Steve Jobs], [Palo Alto], [California]]

    singleEntList = []
    for entity in entityList:
        for e in entity:
            singleEntList.append(e)
    # [Apple, Steve, Jobs, 2011, Palo, Alto, California]

    # groupedEntities = nounChunkEnts.copy()
    groupedEntities = [[e.label_ for e in chunk]for chunk in nounChunkEnts]

    k = 0
    for elem in singleEntList:
        present = False
        j = 0
        for i, span in enumerate(nounChunkEnts):
            for group in span:
                for e in group:
                    if e == elem:
                        present = True
                        j = i
                        break
        if present == False:
            k += 1
            groupedEntities.insert(j+k, [elem.ent_type_])
            break
    globalGroupEntities.append([x for x in groupedEntities if x])

print(globalGroupEntities)

frequencyDict = {}
# in case of ['NORP', 'PERSON'] or ['PERSON', 'NORP'] the key is considered different
for sent in globalGroupEntities:
    for group in sent:
        if len(group) > 1:
            key = "-"
            for e in group:
                key = key + e + '-'
            if key in frequencyDict:
                frequencyDict[key] += 1
            else:
                frequencyDict[key] = 1

frequencyDict=dict(sorted(frequencyDict.items(), key=lambda item: item[1], reverse=True))
print(frequencyDict)

