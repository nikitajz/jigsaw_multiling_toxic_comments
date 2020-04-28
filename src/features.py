import re
from nltk.tokenize import RegexpTokenizer


def text_stats(docs):
    """Count tokens per document (comment, etc).
    Features:
        - text length
        - number of tokens
        - number of exclamation marks (!)
        - rate of UPPERCASE LETTERS

    Arguments:
        docs {List[str]} -- List of documents.

    Returns:
        [dict] -- dict with above features.
    """

    tokenizer = RegexpTokenizer("[\w+.]+")
    REGEX = re.compile(r'[A-Z]')

    features = {}
    features['comment_len'] = [len(doc) for doc in docs]
    features['num_tokens'] = [len(tokenizer.tokenize(doc)) for doc in docs]
    features['num_excl_marks'] = [doc.count("!") for doc in docs]
    features['rate_uppercase'] = [len(REGEX.findall(doc))/len(doc) for doc in docs]

    return features