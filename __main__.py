import argparse
import tensorflow as tf
import numpy as np

from EmbeddingHandler import EmbeddingHandler
from TxtUtilities import getAuthorsAndFiles
from XmlUtilities import parseXml
from TokenUtilities import tokenize


def main():
    args = getParsedArgs()

    authorFiles, authors = getAuthorsAndFiles(f"{args.language}\\{args.training}")
    labels = getLabels(authors)
    authorTweetList = getTweets(
        authorFiles
    )  # tweetsList - list of tweets of all authors
    authorTokenList = getTokens(
        authorTweetList
    )  # tokensList - list of tokenized tweets of all authors

    embeddingsHandler = EmbeddingHandler(
        filename=args.embedding, filePath=args.directory
    )
    embeddingsHandler.loadEmbeddings()
    embeddings = getEmbeddings(authorTokenList, embeddingsHandler)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.compile(optimizer="adam")
    model.fit(x=embeddings, y=np.array(labels), batch_size=2880, epochs=5)


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


def getEmbeddings(authorTokensList, handler):
    """
    ...
    :param authorTokensList:
    :param handler:
    :return:
    """

    embeddings = list()

    for authorTokens in authorTokensList:
        tweetEmbeddings = list()
        for tweetTokens in authorTokens:
            for word in tweetTokens:
                tweetEmbeddings.append(handler.getOrCreateEmbedding(word))
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
