import spacy
from spacy.tokens import Doc
from conll import evaluate, read_corpus_conll, get_chunks, parse_iob
import pandas as pd


def loadTokenizedListInSpacy(data):
    # https://spacy.io/usage/processing-pipelines
    docList = []
    for sent in data:
        doc = Doc(nlp.vocab, words=[w[0] for w in sent])
        for name, proc in nlp.pipeline:
            doc = proc(doc)
        docList.append(doc)
    return docList


def extractGroupEntity(doc):

    nounChunkList = list(doc.noun_chunks)
    entityList = doc.ents
    nounChunkEnts = [span.ents for span in doc.noun_chunks]
    singleEntList = []
    for entity in entityList:
        for e in entity:
            singleEntList.append(e)
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
    return groupedEntities


def groupFrequencyCount(groupEntities):
    frequencyDict = {}
    # in case of ['NORP', 'PERSON'] or ['PERSON', 'NORP'] the key is considered different
    for sent in groupEntities:
        for group in sent:
            key = "-"
            for e in group:
                key = key + e + '-'
            if key in frequencyDict:
                frequencyDict[key] += 1
            else:
                frequencyDict[key] = 1

    frequencyDict = dict(sorted(frequencyDict.items(),
                                key=lambda item: item[1], reverse=True))
    return frequencyDict


def classAndTotAccuracy(hyps, refs, entityList):
    accuracyDict = {}
    countDict = {'TOTAL': [0, 0]}

    if len(hyps) != len(refs):
        raise ValueError(
            "Size Mismatch: ref: {} & hyp: {}".format(len(ref), len(hyp)))
    else:
        for i, sent in enumerate(hyps):
            for j, tHyps in enumerate(sent):
                key = tHyps[1]
                countDict['TOTAL'][1] += 1
                if tHyps == refs[i][j]:
                    countDict['TOTAL'][0] += 1
                if key in countDict:
                    countDict[key][1] += 1
                    if key == refs[i][j][1]:
                        countDict[key][0] += 1
                else:
                    countDict[key] = [1, 1]

        for key in countDict.keys():
            accuracyDict[key] = countDict[key][0]/countDict[key][1]
    return accuracyDict


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

docList = loadTokenizedListInSpacy(trainData)

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

# 1.1 report token-level performance (per class and total)
accuracyDict = classAndTotAccuracy(hyps, refs, entitySet)
print('Question 1.1: Evaluate accuracy\n')

for key in accuracyDict.keys():
    print('{}:\t{:.3f}'.format(key, accuracyDict[key]))
print('\n')

# 1.2 report CoNLL chunk-level performance (per class and total)
results = evaluate(refs, hyps)

pd_tbl = pd.DataFrame().from_dict(results, orient='index')
pd_tbl.round(decimals=3)
print('Question 1.2: Evaluate chunk-level performances\n')
print('{}\n'.format(pd_tbl))

# 2 Grouping of Entities
print('Question 2: Group entities\n')

filteredEntityGroups = []
for doc in docList:
    rawEntityGroups = extractGroupEntity(doc)
    filteredEntityGroups.append([x for x in rawEntityGroups if x])

frequencyDict = groupFrequencyCount(filteredEntityGroups)
for i, key in enumerate(frequencyDict.keys()):
    if i < 15:  # show only the 15 most relevant combinations
        print('{}: {}'.format(key, frequencyDict[key]))
    else:
        break
print('\n')
