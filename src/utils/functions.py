import re

from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer


def get_terms(text: str) -> tuple[str]:
    """Returns the terms of a text. All the terms empty or with length 1 are
    removed.

    Args:
        text: the text that contain the terms

    Returns:
        A tuple with the terms.
    """
    return tuple(
        filter(
            lambda x: len(x) > 1,   # get rid of small terms
            re.split(r"[\s\,\.\!\?\(\)\[\]/\\\{\}'\"]", text.lower())
        )
    )


def lemmatize_query(query: str):
    lemmatizer = WordNetLemmatizer()
    words = simple_preprocess(query)
    lemmatized = []
    for word in words:
        as_verb = lemmatizer.lemmatize(word, pos='v')
        as_adj = lemmatizer.lemmatize(word, pos='a')
        as_noun = lemmatizer.lemmatize(word, pos='n')

        if word != as_verb:
            lemmatized.append(as_verb)
        elif word != as_adj:
            lemmatized.append(as_adj)
        elif word != as_noun:
            lemmatized.append(as_noun)
        else:
            lemmatized.append(word)

    return " ".join(lemmatized)


def lemmatize_word(word: str):
    lemmatizer = WordNetLemmatizer()

    as_verb = lemmatizer.lemmatize(word, pos='v')
    as_adj = lemmatizer.lemmatize(word, pos='a')
    as_noun = lemmatizer.lemmatize(word, pos='n')

    if word != as_verb:
        return as_verb
    elif word != as_adj:
        return as_adj
    elif word != as_noun:
        return as_noun
    else:
        return word
