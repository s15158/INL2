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
]
# It would probably be best to check for some of the symbols at the end of a string and strip (in a loop) appropriately.
# That would ensure that words like "sh!t" would be tokenized properly.
# Words like "yahoo!", however, would still need separate checking and joining.


def tokenize(document, emojis=True):
    text = document.lower().split()
    urls = _tokenizeUrls(text)
    hashtags = _tokenizeHashes(text)
    mentions = _tokenizeMentions(text)
    numbers = _tokenizeNumbers(text)

    if emojis:
        emojis = _tokenizeEmojis(text)

    for special in specialCharacters:
        text = [u"".join(word.replace(special, f" {special} ").split()) for word in text]

    return text, urls, hashtags, mentions, numbers


def _tokenizeUrls(text):
    urls = 0

    for num, word in enumerate(text):
        if u"http" in word or u"www." in word:
            text[num] = u"<url>"
            urls += 1
    return urls


def _tokenizeHashes(text):
    hashtags = 0

    for num, word in enumerate(text):
        if word.startswith(u"#"):
            text[num] = u"<hashtag>"
            hashtags += 1
    return hashtags


def _tokenizeMentions(text):
    mentions = 0

    for num, word in enumerate(text):
        if word.startswith(u"@"):
            text[num] = u"<user>"
            mentions += 1
    return mentions


def _tokenizeNumbers(text):
    numbers = 0

    for num, word in enumerate(text):
        if word.isdigit():
            text[num] = u"<number>"
            numbers += 1
    return numbers


def _tokenizeEmojis(text):
    import emoji

    emojis = 0

    for wordNum, word in enumerate(text):
        for charNum, char in enumerate(word):
            if char in emoji.UNICODE_EMOJI:
                u"".join(text[wordNum].replace(char, f" {char} ").split())
                emojis += 1
    return emojis
