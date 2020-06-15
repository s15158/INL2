# This module defines basic functions for parsing .txt format files.


def getAuthorsAndFiles(filename, path=None):
    import os

    authorFiles = list()

    if path is None:
        path = os.getcwd()

    with open(f"{path}\\{filename}", "r") as f:
        for line in f:
            values = line.split(":::")
            authorFiles.append([values[0], values[1]])
    return authorFiles
