import numpy as np


class EmbeddingHandler(object):
    def __init__(self, filename, filePath=None):
        self.embeddingsDict = dict()
        self.filePath = filePath
        self.filename = filename
        self.embeddingsDim = None

    def findEmbeddingsFile(self, path=None, filename=None):
        import os

        if path is None:
            path = os.getcwd()
        if filename:
            embeddingsFile = filename
        else:
            embeddingsFiles = [f for f in os.listdir(path) if f.endswith("d.txt")]
            embeddingsFile = embeddingsFiles[0] if embeddingsFiles else None
        self.filename = f"{path}\\{embeddingsFile}" if embeddingsFile else None

    def loadEmbeddings(self):
        self.findEmbeddingsFile(path=self.filePath, filename=self.filename)
        with open(self.filename, "r", encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefficients = np.array(values[1:], dtype="float32")
                self.embeddingsDict[word] = coefficients
            self.embeddingsDim = coefficients.size

    def getOrCreateEmbedding(self, word):
        embedding = self.embeddingsDict.get(word)
        if embedding is None:
            embedding = self._generateEmbedding()
        return embedding

    def _getEmbedding(self, word):
        return self.embeddingsDict.get(word)

    def _generateEmbedding(self):
        return np.random.randn(self.embeddingsDim)
