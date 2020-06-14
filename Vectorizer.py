class Vectorizer(object):
    def __init__(self):
        self.vocabulary = None

    def loadVocabulary(self, trainTexts):
        self.vocabulary = dict()
        indexCounter = 0
        vectorizedTexts = []

        for authorTweets in trainTexts:
            vectorizedText = []
            for tweet in authorTweets:
                for word in tweet:
                    wordIndex = self.vocabulary.get(word)
                    if not wordIndex:
                        self.vocabulary[word] = indexCounter
                        indexCounter += 1
                    vectorizedText.append(self.vocabulary.get(word))
            vectorizedTexts.append(vectorizedText)

        self.vocabulary["<unk>"] = 0

        return vectorizedTexts

    def vectorizeText(self, testTexts):

        vectorizedTexts = []

        for authorTweets in testTexts:
            vectorizedText = []
            for tweet in authorTweets:
                for word in tweet:
                    wordIndex = self.vocabulary.get(word)
                    if not wordIndex:
                        wordIndex = self.vocabulary.get("<unk>")
                    vectorizedText.append(wordIndex)
            vectorizedTexts.append(vectorizedText)

        return vectorizedTexts

    def getVocabularyLength(self):
        return len(self.vocabulary)
