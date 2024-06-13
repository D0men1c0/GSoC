import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize, ne_chunk
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
    Cleans a sentence by removing URLs, HTML tags, multiple spaces, punctuation, non-alphanumeric characters,
    leading and trailing spaces, and remaining underscores. 
    Separates numbers from words, keeps proper nouns capitalized,and ensures proper capitalization at the 
    beginning of sentences.
    :param text: the sentence to clean
    :return: the cleaned sentence
    """
    try:
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove leading and trailing spaces
        text = text.strip()
        # Remove HTML tags if present
        if '<' in text and '>' in text:
            text = BeautifulSoup(text, "html.parser").get_text()
        # Tokenize the text
        words = word_tokenize(text)
        # Tag parts of speech
        pos_tags = pos_tag(words)
        # Identify named entities
        named_entities = ne_chunk(pos_tags, binary=False)
        
        # Collect proper nouns
        proper_nouns = set()
        for subtree in named_entities:
            if isinstance(subtree, nltk.Tree):
                if subtree.label() in ['PERSON', 'ORGANIZATION', 'GPE', 'LOCATION']:
                    for leaf in subtree.leaves():
                        proper_nouns.add(leaf[0])
        
        # Convert to lowercase but keep proper nouns capitalized
        cleaned_words = []
        for word in words:
            if word in proper_nouns:
                cleaned_words.append(word)
            else:
                cleaned_words.append(word.lower())
        
        text = ' '.join(cleaned_words)
        
        # Remove punctuation (excluding spaces)
        text = re.sub(r'[^\w\s]', '', text)
        # Remove non-alphanumeric characters (excluding spaces and numbers)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Separate numbers from words by adding spaces around numbers
        text = re.sub(r'(\d+)', r' \1 ', text)
        # Remove any remaining underscores (if needed)
        text = text.replace('_', '')
        # Remove single characters (if desired)
        text = re.sub(r'\b\w\b', '', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Ensure the first letter of each sentence is correctly capitalized
        sentences = text.split('. ')
        try:
            cleaned_sentences = []
            for sentence in sentences:
                if sentence:
                    words = sentence.split()
                    # Convert the first word to lowercase unless it's a proper noun
                    if words[0] not in proper_nouns:
                        words[0] = words[0].lower()
                    cleaned_sentences.append(' '.join(words))
        except:
            pass
        text = '. '.join(cleaned_sentences).strip()
    except:
        return ''
    return text