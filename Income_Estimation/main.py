#!/usr/bin/env python
import sys

from etl import get_data
from processing import process_data

from model import run_model


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'processing', 'model'. 
    
    `main` runs the targets in order of data=>processing=>model.
    '''

    if 'data' in targets:
       
        data = get_data()
        print('data complete')

    if 'processing' in targets:
       
        inflow, determined_transactions, undetermined_transactions = process_data(data)
        print('processing complete')

    if 'model' in targets:
        
        results = run_model(inflow, determined_transactions, undetermined_transactions)

    return


if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)