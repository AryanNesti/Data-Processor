import string
from stringprep import in_table_a1
import nltk
import csv
import math
import json
from nltk.corpus import stopwords
from nltk.stem.porter import *

def getMax2(dict, cat):
  temp = []
  test = {}
  count = 0
  i = 0
  for tok in dict:
    c = tok[1].split('\n')
    if c[0] == cat:
      count += 1
      if i == 0:
        temp = dict[tok]
        i += 1
        continue
      for x in range(len(dict[tok])):
          temp[x] += dict[tok][x]
  for i in range(len(temp)):
    temp[i] = temp[i] / count

        

  max1 = 0
  max2 = 0
  max3 = 0
  i1 = 0
  i2 = 0
  i3 = 0
  for i in temp:
    if Dict[temp.index(i)] == Dict[i1] or Dict[temp.index(i)] == Dict[i2] or Dict[temp.index(i)] == Dict[i3]:
      continue
    if i > max1:
      max3 = max2
      i3 = i2
      max2 = max1
      i2 = i1
      max1 = i
      i1 = temp.index(i)
    elif i > max2:
      max3 = max2
      i3 = i2
      max2 = i
      i2 = temp.index(i)
    elif i > max3:
      max3 = i
      i3 = temp.index(i)
  test[Dict[i1]] = max1
  test[Dict[i2]] = max2
  test[Dict[i3]] = max3
  return test

# Function to get maxes
def getMax(dict, cat):
  temp = []
  test = {}
  count = 0
  i = 0
  for tok in dict:
    c = tok[1].split('\n')
    if c[0] == cat:
      if i == 0:
        temp = dict[tok]
        i += 1
        continue
      for x in range(len(dict[tok])):
          temp[x] += dict[tok][x]

        

  max1 = 0
  max2 = 0
  max3 = 0
  i1 = 0
  i2 = 0
  i3 = 0
  for i in temp:
    if Dict[temp.index(i)] == Dict[i1] or Dict[temp.index(i)] == Dict[i2] or Dict[temp.index(i)] == Dict[i3]:
      continue
    if i > max1:
      max3 = max2
      i3 = i2
      max2 = max1
      i2 = i1
      max1 = i
      i1 = temp.index(i)
    elif i > max2:
      max3 = max2
      i3 = i2
      max2 = i
      i2 = temp.index(i)
    elif i > max3:
      max3 = i
      i3 = temp.index(i)
  test[Dict[i1]] = max1
  test[Dict[i2]] = max2
  test[Dict[i3]] = max3
  return test

# stemming tool from nltk
stemmer = PorterStemmer()
# a mapping dictionary that help remove punctuations
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def get_tokens(text):
  # turn document into lowercase
  lowers = text.lower()
  # remove punctuations
  no_punctuation = lowers.translate(remove_punctuation_map)
  # tokenize document
  tokens = nltk.word_tokenize(no_punctuation)
  # remove stop words
  filtered = [w for w in tokens if not w in stopwords.words('english')]
  # stemming process
  stemmed = []
  for item in filtered:
      stemmed.append(stemmer.stem(item))
  # final unigrams
  return stemmed
Dict = []
Check = {}
FreqD = {}
MaxD = {}
mD = {}
TF = {}
IDF = {}
FDict = {}
ExtraD = {}
#Dictionary of Word to Each Frequency

# Opening and Creating a dictionary
with open('dictionary.txt', 'r') as dict:
  for line in dict:
    l = line.strip('\n')
    Dict.append(l)
    mD[l] = 0

#Opening File and creating a dictionary of all collums
with open("news-train.csv", 'r') as file:
  csvreader = csv.reader(file)
  next(csvreader)
  for line in file:
    collums = line.split(",")
    Check[(collums[0],collums[2])] = get_tokens(collums[1])
    #loop for each list of words in a artcle
  for tok in Check:
    max = 0
    Freq = []
    #for loop for each word in list
    temp = []
    for word in Dict:
      if word in Check[tok]:
        if word in temp:
          continue
        temp.append(word)
        Freq.append(Check[tok].count(word))
        mD[word] += 1
      else:
        Freq.append(0)
    FreqD[tok] = Freq
    for i in Freq:
      if i > max:
        max = i
    MaxD[tok] = max
  for tok in FreqD:
    temp = []
    temp1 = []
    for i in FreqD[tok]:
      if MaxD[tok] == 0:
        temp.append(0)
        continue
      temp.append(i / MaxD[tok])
    TF[tok] = temp
    for i in mD:
      if mD[i] == 0:
        temp1.append(0)
        continue
      temp1.append(math.log(1490/mD[i], 10))
    IDF[tok] = temp1
  for tok in TF:
    temp = []
    for i in range(len(TF[tok])):
        temp.append(TF[tok][i] * IDF[tok][i])
    FDict[tok] = temp
  
  with open("matrix.txt", "w") as outfile:
    for row in FDict:
      for col in FDict[row]:
        outfile.write(str(col) + ", ")

  business = getMax2(FDict, 'business')
  tech = getMax2(FDict, 'tech')
  entertainment = getMax2(FDict, 'entertainment')
  politics = getMax2(FDict, 'politics')
  sport = getMax2(FDict, 'sport')

#creates a json file for the scores
json_object1 = json.dumps(business, indent = 4)
json_object2 = json.dumps(tech, indent = 4)
json_object3 = json.dumps(entertainment, indent = 4)
json_object4 = json.dumps(politics, indent = 4)
json_object5 = json.dumps(sport, indent = 4)
with open("score.json", "w") as outfile:
  outfile.write('business ')
  outfile.write(json_object1)
  outfile.write('\ntech ')
  outfile.write(json_object2)
  outfile.write('\nentertainment ')
  outfile.write(json_object3)
  outfile.write('\npolitics ')
  outfile.write(json_object4)
  outfile.write('\nsport ')
  outfile.write(json_object5)

business = getMax(FreqD, 'business')
tech = getMax(FreqD, 'tech')
entertainment = getMax(FreqD, 'entertainment')
politics = getMax(FreqD, 'politics')
sport = getMax(FreqD, 'sport')
json_object1 = json.dumps(business, indent = 4)
json_object2 = json.dumps(tech, indent = 4)
json_object3 = json.dumps(entertainment, indent = 4)
json_object4 = json.dumps(politics, indent = 4)
json_object5 = json.dumps(sport, indent = 4)
with open("frequency.json", "w") as outfile:
  outfile.write('business ')
  outfile.write(json_object1)
  outfile.write('\ntech ')
  outfile.write(json_object2)
  outfile.write('\nentertainment ')
  outfile.write(json_object3)
  outfile.write('\npolitics ')
  outfile.write(json_object4)
  outfile.write('\nsport ')
  outfile.write(json_object5)