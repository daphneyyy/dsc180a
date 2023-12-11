# DSC180A-B18-Quarter1 Final Project-Group2
## Summary
In fall quarter 2023, in order for better measuring credit scores of customers, our group did transaction categorization and income estimation tasks. For transaction, we used two models which are Bert(LLM) and the combination of TF-IDF Tokenizer and SVC. Both two models gave us a high accuracy in classification. While Bert took a long time to train, TF-IDF is much faster with a higher accuracy score.  For income estimation, we calculated the recurrence for certain inflow categories to check whether the transaction belongs to income for a customer. If transactions of the same source were deemed recurrent enough, they were counted along as income on top of anything from the paycheck and paycheck placeholder categories. 
## Set up Environment: 

Mac SetUp
```sh
pip install virtualenv
virtualenv -p python3.9 myenv

# Activate the virtual environment
source myenv/bin/activate 
pip install -r requirements.txt

# To deactivate the virtual environment when you're done
deactivate
```

## Running Code:
- ### Bert Model for transaction categorization
  ```sh
  cd Transaction_Categorization/Bert_Model
  python3 main.py
  ```
- ### TF-IDF & Support Vector Classification for transaction categorization
  ```sh
  cd Transaction_Categorization/TF-IDF_Model
  python3 main.py
  ```
- ### Income Estimation
  ```sh
  cd Income_Estimation
  python3 run.py data processing model
  ```
  

    
