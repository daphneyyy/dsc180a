import re
# Data Cleanning Process Part

## Use regular expressions to remove text after ".com" 
## and keep the preceding text from ".com"
def remove_com(text):
    cleaned_text = re.sub(r'\.com(\/bill|\*).*?(?=\s|$)', '', text)
    return cleaned_text


## Removing useless pattenrs
def remove_key_phrases(text):
    phrases = [
        'pos debit - visa check card xxxx - ',
        r'purchase authorized on \d{2}\/\d{2}',
        'pos purchase',
        'purchase',
        'pos',
        'web id',
        'terminal id',
        r'\b(id)\b',
        'withdrawal consumer debit',
        'withdrawal',
        'debit card',
        'credit card',
        'checkcard',
        'recurring payment authorized on'
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
    text = re.sub(r'\b(x+)\b', ' ', text)
    text = re.sub(r'\b(xx+)(\w)', r'\2', text)
    text = re.sub(r'\b\w+x{2,}\w+\b', ' ', text)
    return text


## Removing all the digits
def remove_digits(text):
    text = re.sub(r'\b(\d+)\b', ' ', text)
    text = re.sub(r'([^\s]*\d+)(\b)', r'\2', text)
    text = re.sub(r'\b(\d+)([a-zA-Z])', r'\2', text)
    text = re.sub(r'\b\w+\d\w+\b', ' ', text)
    return text


## Simplify repeating pattenrs for amazon and walmart
def standardize_phrase(text):
    text = re.sub(r'\b(amazon|amzn|amz)\b', 'amazon', text)
    text = re.sub(r'\b(wal\smart|wal|wm\ssupercenter|wm\ssuperc|wm)\b', 'walmart', text)
    return text


## Removing "oh" patterns
def remove_oh(text):
    text = re.sub(r'\b(oh)\b', ' ', text)
    return text


## Removing multiple spaces
def remove_multiple_spaces(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()