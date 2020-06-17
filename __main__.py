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
    filesTrain = [i[0] for i in authorsAndFilesTrain]
    authorsTrain = [i[1] for i in authorsAndFilesTrain]

    authorsAndFilesTest = getAuthorsAndFiles(f"{dataDir}\\{args.test[0]}")
    random.shuffle(authorsAndFilesTest)
    filesTest = [i[0] for i in authorsAndFilesTest]
    authorsTest = [i[1] for i in authorsAndFilesTest]

    labelsTrain = getLabels(authorsTrain)
    tweetListTrain = getTweets(filesTrain, dataDir)
    tokenListTrain = getTokens(tweetListTrain)

    labelsTest = getLabels(authorsTest)
    tweetListTest = getTweets(filesTest, dataDir)
    tokenListTest = getTokens(tweetListTest)

    embeddingsHandler = EmbeddingHandler(
        filename=args.embeddings, filePath=args.directory
    )
    embeddingsHandler.loadEmbeddings()

    vectorizer = Vectorizer()
    vectorizedTextsTrain = vectorizer.loadVocabulary(tokenListTrain)
    vectorizedTextsTest = vectorizer.vectorizeText(tokenListTest)
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
    model.add(tf.keras.layers.LSTM(32, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(16))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            amsgrad=False,
        ),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    history = model.fit(
        x=trainingData,
        y=labelsTrain,
        validation_data=(testData, labelsTest),
        batch_size=32,
        epochs=30,
        validation_split=0.3,
    )

    legend = ["train", "validation"]
    draw2DPlot(history, "model accuracy", "binary_accuracy", "val_binary_accuracy", "accuracy", "epoch", legend)
    draw2DPlot(history, "model loss", "loss", "val_loss", "loss", "epoch", legend)


def getTweets(authorFileList, path):
    return [parseXml(authorFile, path=path) for authorFile in authorFileList]


def getTokens(authorTweetList):
    return [tokenize(tweet) for tweet in [authorTweets for authorTweets in authorTweetList]]


def getEmbeddingMatrix(embeddingHandler, vectorizer):
    embeddingMatrix = np.zeros(
        (vectorizer.getVocabularyLength(), embeddingHandler.embeddingsDim)
    )

    for word, index in vectorizer.vocabulary.items():
        embeddingVector = embeddingHandler.getOrCreateEmbedding(word)
        embeddingMatrix[index] = embeddingVector
    return embeddingMatrix


def getLabels(authors):
    return np.array([1 if author == "bot" else 0 for author in authors])


def draw2DPlot(history, title, x, y, xlabel, ylabel, legend):
    import matplotlib.pyplot as plt

    plt.title(title)
    plt.plot(history.history[x])
    plt.plot(history.history[y])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend, loc="upper left")
    plt.show()


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
    parser.add_argument("test", nargs="*", help="Files containing test dataset")
    return parser.parse_args()


if __name__ == "__main__":
    main()
