import nltk
import re
from bs4 import BeautifulSoup
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


def clean_sentences(text):
    """
    Cleans a sentence by removing multiple spaces, punctuation, non-alphanumeric characters, leading and trailing spaces,
    numbers, and remaining underscores.
    :param text: the sentence to clean
    :return: the cleaned sentence
    """
    if '<' in text and '>' in text:
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
    # Remove non-alphanumeric characters (excluding spaces and numbers)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Separate numbers from words by adding spaces around numbers
    text = re.sub(r'(\d+)', r' \1 ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing spaces
    text = text.strip()
    # Remove single characters (if desired)
    text = re.sub(r'\b\w\b', '', text)
    # Remove any remaining underscores (if needed)
    text = text.replace('_', '')
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase the text
    text = text.lower()
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text