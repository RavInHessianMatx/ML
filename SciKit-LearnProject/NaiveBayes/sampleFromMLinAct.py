from sklearn import naive_bayes as NB
import numpy as np
import feedparser
import pprint as pp


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def calcMostFreq(vocabList,fullText,topwordNum):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:topwordNum]

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def localWords(feed1,feed0,topwordNum):
    docList=[]
    classList = []
    fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    topWords = calcMostFreq(vocabList,fullText,topwordNum)
    for pairW in topWords:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2*minLen)
    testSet=[]
    for i in range(int(len(trainingSet) * 0.25)):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    x_train=[]
    y_train = []
    x_test = []
    y_test = []
    for docIndex in trainingSet:
        x_train.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        y_train.append(classList[docIndex])
    for docIndex in testSet:
        x_test.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        y_test.append(classList[docIndex])


    gnb = NB.GaussianNB()
    # bnb = NB.BernoulliNB()
    # mnb = NB.MultinomialNB()
    y_predict = gnb.fit(x_train, y_train).predict(x_test)
    return (y_predict != y_test).sum()

if __name__ == '__main__':

    # UKJinRong = feedparser.parse('http://www.ftchinese.com/rss/feed')                          # 100
    # wineworld = feedparser.parse('http://www.wine-world.com/articlerss/rss.aspx')              # 75
    nasa = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')                # 60
    zhihuMeiRi = feedparser.parse('https://www.zhihu.com/rss')                                 # 60
    # toutiao = feedparser.parse('http://news.163.com/special/00011K6L/rss_newstop.xml')         # 50
    # weifeng = feedparser.parse('http://news.feng.com/rss.xml')                                 # 40
    # caijingZhoukan = feedparser.parse('http://blog.163.com/cbn.weekly/rss/')                   # 40
    # geekpark = feedparser.parse('http://www.geekpark.net/rss')                                 # 30
    # yahoo = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')                  # 15
    # nature = feedparser.parse('http://feeds.nature.com/news/rss/most_recent')                  # 15
    # cnblog = feedparser.parse('http://feed.cnblogs.com/blog/u/161528/rss')                     # 10
    # matrix67 = feedparser.parse('http://www.matrix67.com/blog/feed')                           # 10
    # chaijinSina = feedparser.parse('http://blog.sina.com.cn/rss/1219548027.xml')               # 10




    for _ in range(100):
        errors = localWords(nasa, zhihuMeiRi,0)
        print(errors)
        print(errors/float(30))