# This module defines basic functions for parsing .txt format files.


def getAuthorsAndFiles(filename, path=None):
    import os

    filenames = list()
    authors = list()

    if path is None:
        path = os.getcwd()

    with open(f"{path}\\{filename}", "r") as f:
        for line in f:
            values = line.split(":::")
            filenames.append(values[0])
            authors.append(values[1])
    return filenames, authors
