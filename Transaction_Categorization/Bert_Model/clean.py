import re

## Data Preprocessing Steps 

## Removing special characters.
def remove_special_char(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

## Removing all the repeat 'x' patterns
def remove_xs(text):
    text = re.sub(r'(xx+)\b', ' ', text)
    text = re.sub(r'\b(x)\b', ' ', text)
    text = re.sub(r'\b(xx+)([a-zA-Z])', r'xx\2', text)
    return text

## Simplify repeating patterns for amazon and walmart
def standardize_phrase(text):
    text = re.sub(r'\b(amazon|amzn|amz)\b', 'amazon', text)
    text = re.sub(r'\b(wal\smart|wal|wm\ssupercenter|wm\ssuperc|wm)\b', 'walmart', text)
    return text

## Removing multiple spaces
def remove_multiple_spaces(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

## Use regular expressions to remove text after ".com*" 
## and keep the preceding text from ".com"
def clean_text1(text):
    # Use regular expressions to remove text after ".com*" and keep the preceding text from ".com"
    cleaned_text = re.sub(r'\.com\*.*?(?=\s|$)', '', text)
    return cleaned_text

## Removing useless patterns
def remove_key_phrases(text):
    phrases = [
        'pos debit - visa check card xxxx - ',
        'purchase authorized on xx/xx',
        'pos purchase',
        'purchase',
        'pos',
        'web id',
        'terminal id',
        'id'
    ]
    for phrase in phrases:
        text = re.sub(phrase, '', text)
    return text
