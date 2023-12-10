import re
# Data Cleanning Process Part

## Use regular expressions to remove text after ".com*" 
## and keep the preceding text from ".com"
def clean_text1(text):
    # Use regular expressions to remove text after ".com*" and keep the preceding text from ".com"
    cleaned_text = re.sub(r'\.com\*.*?(?=\s|$)', '', text)
    return cleaned_text


## Removing useless pattenrs
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


## Removing special characters.
def remove_special_char(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)


## Removing all the repeat 'x' patterns
def remove_xs(text):
    text = re.sub(r'(xx+)\b', ' ', text)
    text = re.sub(r'\b(x)\b', ' ', text)
    text = re.sub(r'\b(xx+)([a-zA-Z])', r'xx\2', text)
    return text


## Simplify repeating pattenrs for amazon and walmart
def standardize_phrase(text):
    text = re.sub(r'\b(amazon|amzn|amz)\b', 'amazon', text)
    text = re.sub(r'\b(wal\smart|wal|wm\ssupercenter|wm\ssuperc|wm)\b', 'walmart', text)
    return text


## Removing multiple spaces
def remove_multiple_spaces(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

## Removing numbers and "oh" patterns in the end of strings
def remove_numbers_and_oh(sentence):
    # Define the regular expression pattern to match numbers, "oh," and date-like patterns at the end of the sentence
    pattern = re.compile(r'\b(?:\d+\s*|oh\s*|\d{1,2}/\d{1,2})+$')

    # Find the match in the sentence
    match = pattern.search(sentence)

    if match:
        # If there is a match, remove the matched part from the end of the sentence
        sentence = sentence[:match.start()].rstrip()

    return sentence
