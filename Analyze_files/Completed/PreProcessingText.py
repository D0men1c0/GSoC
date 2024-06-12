import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_title(title):
    """
    Preprocesses the title of a news article by tokenizing, lowercasing, and removing stopwords
    :param title: the title of a news article
    :return: the preprocessed title
    """
    tokens = word_tokenize(title)
    # Lowercase all tokens
    tokens = [token.lower() for token in tokens]
    # Remove tokens that are not alphabetic
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


def preprocess_content(content):
    """
    Preprocesses the content of a news article by tokenizing, lowercasing, and removing stopwords
    :param content: the content of a news article
    :return: the preprocessed content
    """
    tokens = word_tokenize(content)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)
