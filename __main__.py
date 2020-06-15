import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import Constant

from EmbeddingHandler import EmbeddingHandler
from TxtUtilities import getAuthorsAndFiles
from Vectorizer import Vectorizer
from XmlUtilities import parseXml
from TokenUtilities import tokenize


def main():
    args = getParsedArgs()

    dataDir = f"{args.directory}\\{args.language}"

    authorFilesTrain, authorsTrain = getAuthorsAndFiles(f"{dataDir}\\{args.training}")
    authorFilesTest, authorsTest = getAuthorsAndFiles(f"{dataDir}\\{args.test}")

    labelsTrain = getLabels(authorsTrain)
    authorTweetListTrain = getTweets(
        authorFilesTrain, dataDir
    )  # tweetsList - list of tweets of all authors
    authorTokenListTrain = getTokens(
        authorTweetListTrain
    )  # tokensList - list of tokenized tweets of all authors

    labelsTest = getLabels(authorsTest)
    authorTweetListTest = getTweets(
        authorFilesTest, dataDir
    )  # tweetsList - list of tweets of all authors
    authorTokenListTest = getTokens(
        authorTweetListTest
    )  # tokensList - list of tokenized tweets of all authors

    embeddingsHandler = EmbeddingHandler(
        filename=args.embeddings, filePath=args.directory
    )
    embeddingsHandler.loadEmbeddings()

    vectorizer = Vectorizer()
    vectorizedTextsTrain = vectorizer.loadVocabulary(authorTokenListTrain)
    vectorizedTextsTest = vectorizer.vectorizeText(authorTokenListTest)
    vocabularyLength = vectorizer.getVocabularyLength()
    embeddingMatrix = getEmbeddingMatrix(embeddingsHandler, vectorizer)

    keras1 = tf.keras.preprocessing.sequence.pad_sequences(vectorizedTextsTrain, maxlen=1000, padding="post")
    keras2 = tf.keras.preprocessing.sequence.pad_sequences(vectorizedTextsTest, maxlen=1000, padding="post")

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Embedding(
            vocabularyLength,
            embeddingsHandler.embeddingsDim,
            embeddings_initializer=Constant(embeddingMatrix),
            trainable=True,
        )
    )
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(32))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="relu"))

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        x=keras1,
        y=np.array(labelsTrain),
        validation_data=(keras2, np.array(labelsTest)),
        batch_size=16,
        epochs=30,
    )
    scores = model.evaluate(keras2, np.array(labelsTest), verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


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
        "-d", "--directory", help="Get data from provided directory", default="Data"
    )
    parser.add_argument(
        "-l", "--language", help="Get data of provided language", default="en"
    )
    parser.add_argument(
        "-e", "--embeddings", help="File containing pre-trained embeddings", default="glove.twitter.27B.25d.txt"
    )
    parser.add_argument("-t", "--training", help="File containing training dataset", default="truth-train.txt")
    parser.add_argument("-q", "--quiet", help="Argument responsible for the visible spam in logs")
    parser.add_argument("test", nargs="*", help="File containing test dataset", default="truth-dev.txt")
    return parser.parse_args()


if __name__ == "__main__":
    main()
