import argparse
import tensorflow as tf
import numpy as np

from EmbeddingHandler import EmbeddingHandler
from TxtUtilities import getAuthorsAndFiles
from Vectorizer import Vectorizer
from XmlUtilities import parseXml
from TokenUtilities import tokenize


def main():
    args = getParsedArgs()

    authorFilesTrain, authorsTrain = getAuthorsAndFiles(f"{args.language}\\{args.training}")
    authorFilesTest, authorsTest = getAuthorsAndFiles(f"{args.language}\\{args.test}")

    labelsTrain = getLabels(authorsTrain)
    authorTweetListTrain = getTweets(
        authorFilesTrain
    )  # tweetsList - list of tweets of all authors
    authorTokenListTrain = getTokens(
        authorTweetListTrain
    )  # tokensList - list of tokenized tweets of all authors

    labelsTest = getLabels(authorsTest)
    authorTweetListTest = getTweets(
        authorFilesTest
    )  # tweetsList - list of tweets of all authors
    authorTokenListTest = getTokens(
        authorTweetListTest
    )  # tokensList - list of tokenized tweets of all authors

    embeddingsHandler = EmbeddingHandler(
        filename=args.embedding, filePath=args.directory
    )
    embeddingsHandler.loadEmbeddings()
    # embeddings = getEmbeddings(authorTokenListTrain, embeddingsHandler)

    vectorizer = Vectorizer()
    vectorizedTextsTrain = vectorizer.loadVocabulary(authorTokenListTrain)
    vectorizedTextsTest = vectorizer.vectorizeText(authorTokenListTest)
    vocabularyLength = vectorizer.getVocabularyLength()
    embeddingMatrix = getEmbeddingMatrix(embeddingsHandler, vectorizer)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocabularyLength,
                                        embeddingsHandler.embeddingsDim,
                                        embeddings_initializer=tf.keras.layers.Constant(embeddingMatrix),
                                        trainable=True))
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.compile(optimizer="adam")
    model.fit(x=vectorizedTextsTrain, y=np.array(labelsTrain),
              validation_data=(vectorizedTextsTest, labelsTest),
              batch_size=2880, epochs=5)


def getTweets(authorFileList):
    tweets = list()

    for authorFile in authorFileList:
        tweets.append(parseXml(authorFile))
    return tweets


def getTokens(authorTweetList):
    tokens = list()

    for authorTweets in authorTweetList:
        tokens.append(tokenize(tweet)[0] for tweet in authorTweets)
    return tokens


def getEmbeddingMatrix(embeddingHandler, vectorizer):
    embeddingMatrix = np.zeros((vectorizer.getVocabularyLength(), embeddingHandler.embeddingsDim))

    for word, index in vectorizer.vocabulary.items():
        embeddingVector = embeddingHandler.getOrCreateEmbedding(word)
        embeddingMatrix[index] = embeddingVector

    return embeddingMatrix


def getEmbeddings(authorTokensList, embeddingHandler):
    """
    ...
    :param authorTokensList:
    :param embeddingHandler:
    :return:
    """

    embeddings = list()

    for authorTokens in authorTokensList:
        tweetEmbeddings = list()
        for tweetTokens in authorTokens:
            for word in tweetTokens:
                tweetEmbeddings.append(embeddingHandler.getOrCreateEmbedding(word))
        embeddings.append(np.array(tweetEmbeddings))
    return np.array(embeddings)


def getLabels(authors):
    return [1 if author is "bot" else 0 for author in authors]


def getParsedArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--directory", help="Get data from provided directory", default="."
    )
    parser.add_argument(
        "-l", "--language", help="Get data of provided language", default="en"
    )
    parser.add_argument(
        "-e", "--embeddings", help="File containing pre-trained embeddings"
    )
    parser.add_argument("-t", "--training", help="File containing training dataset")
    parser.add_argument("test", nargs="*", help="File containing test dataset")
    return parser.parse_args()


if __name__ == "__main__":
    main()
