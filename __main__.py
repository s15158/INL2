import argparse
import random

import tensorflow as tf
import numpy as np
from tensorflow.python import Constant

from EmbeddingHandler import EmbeddingHandler
from TxtUtilities import getAuthorsAndFiles
from Vectorizer import Vectorizer
from XmlUtilities import parseXml
from TokenUtilities import tokenize


def main():
    args = getParsedArgs()

    dataDir = f"{args.directory}\\{args.language}"
    random.seed(42)

    authorsAndFilesTrain = getAuthorsAndFiles(f"{dataDir}\\{args.training}")
    random.shuffle(authorsAndFilesTrain)
    authorFilesTrain = [i[0] for i in authorsAndFilesTrain]
    authorsTrain = [i[1] for i in authorsAndFilesTrain]

    authorsAndFilesTest = getAuthorsAndFiles(f"{dataDir}\\{args.test[0]}")
    random.shuffle(authorsAndFilesTest)
    authorFilesTest = [i[0] for i in authorsAndFilesTest]
    authorsTest = [i[1] for i in authorsAndFilesTest]

    labelsTrain = getLabels(authorsTrain)
    authorTweetListTrain = getTweets(authorFilesTrain, dataDir)
    authorTokenListTrain = getTokens(authorTweetListTrain)

    labelsTest = getLabels(authorsTest)
    authorTweetListTest = getTweets(authorFilesTest, dataDir)
    authorTokenListTest = getTokens(authorTweetListTest)

    embeddingsHandler = EmbeddingHandler(
        filename=args.embeddings, filePath=args.directory
    )
    embeddingsHandler.loadEmbeddings()

    vectorizer = Vectorizer()
    vectorizedTextsTrain = vectorizer.loadVocabulary(authorTokenListTrain)
    vectorizedTextsTest = vectorizer.vectorizeText(authorTokenListTest)
    vocabularyLength = vectorizer.getVocabularyLength()
    embeddingMatrix = getEmbeddingMatrix(embeddingsHandler, vectorizer)

    trainingData = tf.keras.preprocessing.sequence.pad_sequences(
        vectorizedTextsTrain, maxlen=1024, padding="post"
    )
    testData = tf.keras.preprocessing.sequence.pad_sequences(
        vectorizedTextsTest, maxlen=1024, padding="post"
    )

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Embedding(
            vocabularyLength,
            embeddingsHandler.embeddingsDim,
            embeddings_initializer=Constant(embeddingMatrix),
            trainable=True,
        )
    )
    model.add(tf.keras.layers.LSTM(32))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.01,
            amsgrad=False,
        ),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    model.fit(
        x=trainingData,
        y=np.array(labelsTrain),
        validation_data=(testData, np.array(labelsTest)),
        batch_size=32,
        epochs=50,
        validation_split=0.3,
    )
    scores = model.evaluate(testData, np.array(labelsTest), verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


def getTweets(authorFileList, path):
    tweets = list()

    for authorFile in authorFileList:
        tweets.append(parseXml(authorFile, path=path))
    return tweets


def getTokens(authorTweetList):
    tokens = list()

    for authorTweets in authorTweetList:
        tokens.append(tokenize(tweet)[0] for tweet in authorTweets)
    return tokens


def getEmbeddingMatrix(embeddingHandler, vectorizer):
    embeddingMatrix = np.zeros(
        (vectorizer.getVocabularyLength(), embeddingHandler.embeddingsDim)
    )

    for word, index in vectorizer.vocabulary.items():
        embeddingVector = embeddingHandler.getOrCreateEmbedding(word)
        embeddingMatrix[index] = embeddingVector
    return embeddingMatrix


def getLabels(authors):
    return [1 if author == "bot" else 0 for author in authors]


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
