from emoji import UNICODE_EMOJI

# This module defines basic functions for tokenizing strings.
specialCharacters = [
    u"!",
    u"@",
    u"#",
    u"$",
    u"%",
    u"^",
    u"&",
    u"*",
    u"(",
    u")",
    u"-",
    u"_",
    u"=",
    u"+",
    u"[",
    u"]",
    u"{",
    u"}",
    u"\\",
    u"|",
    u";",
    u":",
    u"'",
    u'"',
    u",",
    u".",
    u"/",
    u"?",
    u"£",
    u"€",
    u"…",
    u"’",
    u"”",
]
# It would probably be best to check for some of the symbols at the end of a string and strip (in a loop) appropriately.
# That would ensure that words like "sh!t" would be tokenized properly.
# Words like "yahoo!", however, would still need separate checking and joining.


def tokenize(document, emojis=True):
    text = u"".join(document).lower().split()
    _tokenizeUrls(text)
    _tokenizeHashes(text)
    _tokenizeMentions(text)
    _tokenizeNumbers(text)
    # _tokenizePossessives(text)
    # _tokenizeAllCaps(text)

    tokens = _tokenizeByList(text, specialCharacters)
    if emojis:
        tokens = _tokenizeByList(tokens, UNICODE_EMOJI)

    return tokens


def _tokenizeByList(text, charList):
    tokens = list()

    for word in text:
        for char in charList:
            word = word.replace(char, f" {char} ")
        tokens.extend(word.split())
    return tokens


def _tokenizeUrls(text):
    for num, word in enumerate(text):
        if u"http" in word or u"www." in word:
            text[num] = u"<url>"


def _tokenizeHashes(text):
    for num, word in enumerate(text):
        if word.startswith(u"#"):
            text[num] = u"<hashtag>"


def _tokenizeMentions(text):
    for num, word in enumerate(text):
        if word.startswith(u"@"):
            text[num] = u"<user>"


def _tokenizeNumbers(text):
    for num, word in enumerate(text):
        if word.isdigit():
            text[num] = u"<number>"
