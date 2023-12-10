from clean import *
import pandas as pd
import time
# upload the dataset and load the data.
# this dataset contains the date information for transaction.
file1 = 'Transacation_outflows_with_date_3k_firsthalf.pqt'
data1 = pd.read_parquet(file1, engine='auto')
file2 = 'Transacation_outflows_with_date_3k_secondhalf.pqt'
data2 = pd.read_parquet(file2, engine='auto')

# create a dataframe for the dataset
df = pd.concat([data1, data2], axis=0)

# Filter the required categories and define a new dataset
# which only contains these categories.
categories_filter = ['GENERAL_MERCHANDISE', 'FOOD_AND_BEVERAGES', 'GROCERIES', 'TRAVEL', 'PETS', 'EDUCATION', 'OVERDRAFT', 'RENT', 'MORTGAGE']
df = df[df['category'].isin(categories_filter)]

## Changing memo_clean column values to all lower case first.
df['memo'] = df['memo'].str.lower()

# Applying thoese cleaning functions to the subset of the dataset
# that we choose.
dff = df.copy()
dff['memo'] = dff['memo'].apply(clean_text1)
dff['memo'] = dff['memo'].apply(remove_key_phrases)
dff['memo'] = dff['memo'].apply(remove_special_char)
dff['memo'] = dff['memo'].apply(remove_xs)
dff['memo'] = dff['memo'].apply(standardize_phrase)
dff['memo'] = dff['memo'].apply(remove_multiple_spaces)
dff['memo'] = dff['memo'].apply(remove_numbers_and_oh)

dff['new_date'] = pd.to_datetime(dff['posted_date'])

# Extract month, day, and weekday into new columns
dff['month'] = dff['new_date'].dt.month
dff['day'] = dff['new_date'].dt.day
dff['weekday'] = dff['new_date'].dt.weekday 

dff = dff.drop(['prism_consumer_id', 'prism_account_id', 'posted_date', 'new_date'], axis =1)