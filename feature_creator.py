import json
from unicodedata import category
import numpy as np
import pandas as pd

file = open('cleaned.json')
data = json.load(file)
file.close()


personal_categories = ['household', 'entertainment', 'transport',
                       'personalCare', 'education', 'feeding', 'taxes', 
                        'others', 'health', 'finances', 'travel',
                         'shopping']

periodicity_list = ['DAILY', 'WEEKLY', 'NONE', 'MONTHLY', 'YEARLY']
new_categories = []

def filter_bills(bills):
    
    new_bills = []
    for bill in bills:
        has_periodicity = False

        if bill['role'] != 'RoleType.PERSONAL':
            continue

        if bill['category'] not in new_categories:
            new_categories.append(bill['category'])

        if bill['category'] == 'rent':
            bill['category'] = 'household'

            
      

        if bill['category'] == 'car' or bill['category'] == 'fuel' or bill['category'] == 'parking' :
            bill['category'] = 'transport'

        if bill['category'] == 'health_care':
            bill['category'] = 'health'

        if bill['category'] == 'dining':
            bill['category'] = 'feeding'

        if bill['category'] == 'beauty' or bill['category'] == 'fitness':
            bill['category'] = 'personalCare'

        if bill['category'] == 'groceries' or bill['category'] == 'market':
            bill['category'] = 'feeding'

        if bill['category'] == 'clothing':
            bill['category'] = 'shopping'

        


        if bill['category'] not in personal_categories or bill['value'] == None:
            continue

        if bill['periodicity'] == None:
            continue

        for peridicity in periodicity_list:
            if peridicity in bill['periodicity']:
                has_periodicity = True

        if has_periodicity == False:
            continue

        new_bills.append(bill)
    
    return new_bills


def add_missing_values():
    data = {}
    for category in personal_categories:
        data['n_'+category] = 0
        data[category + '_total'] = 0
        # data['periodicity_NONE_' + category] = 0
        # data['periodicity_DAILY_' + category] = 0
        # data['periodicity_WEEKLY_' + category] = 0
        # data['periodicity_MONTHLY_' + category] = 0
        # data['periodicity_YEARLY_' + category] = 0
        # data['periodicity_NONE_' + category + '_value'] = 0
        # data['periodicity_DAILY_' + category + '_value'] = 0
        # data['periodicity_WEEKLY_' + category + '_value'] = 0
        # data['periodicity_MONTHLY_' + category + '_value'] = 0
        # data['periodicity_YEARLY_' + category + '_value'] = 0
    return data


def generate_bill_data(list_of_bills):
    bill_data = {}
    bill_data['n_bills'] = 0
    bill_data['total_bills'] = 0

    for bill in list_of_bills:

        bill_data['total_bills'] += bill['value']

        if(bill_data.__contains__('n_'+bill['category'])):
            bill_data['n_'+bill['category']] += 1
        else:
            bill_data['n_'+bill['category']] = 1
        bill_data['n_bills'] += 1

    # bill_data['mean_bills'] = bill_data['total_bills'] / bill_data['n_bills']
    return bill_data


def generate_bill_total(list_of_bills):
    bill_data = {}

    for bill in list_of_bills:

        if(bill_data.__contains__(bill['category'] + '_total')):
            bill_data[bill['category'] + '_total'] += bill['value']
        else:
            bill_data[bill['category'] + '_total'] = bill['value']

    return bill_data


# def add_missing_expense_category(new_data):
#   if(new_data.__contains__('n_administration') == False):
#     new_data['n_administration']=0

def generate_periodicity_data(list_of_bills):
    bill_data = {}

    for bill in list_of_bills:
        if bill['value'] is None:
            continue

        if(bill_data.__contains__('periodicity_' + bill['periodicity'].split('.')[1] + '_' + bill['category'])):
            bill_data['periodicity_' +
                      bill['periodicity'].split('.')[1] + '_' + bill['category']] += 1
        else:
            bill_data['periodicity_' +
                      bill['periodicity'].split('.')[1] + '_' + bill['category']] = 1

        if(bill_data.__contains__('periodicity_' + bill['periodicity'].split('.')[1] + '_' + bill['category'] + '_value')):
            bill_data['periodicity_' +
                      bill['periodicity'].split('.')[1] + '_' + bill['category'] + '_value'] += bill['value']
        else:
            bill_data['periodicity_' +
                      bill['periodicity'].split('.')[1] + '_' + bill['category'] + '_value'] = bill['value']

    return bill_data


new_dataset = []

for entry in data:
    new_entry = {}

    new_bills_list = filter_bills(entry['bills'])

    if len(new_bills_list) <= 4:
        continue

    new_entry.update(add_missing_values())
    new_entry.update(generate_bill_data(new_bills_list))
    new_entry.update(generate_bill_total(new_bills_list))
    # new_entry.update(generate_periodicity_data(new_bills_list))
#    new_entry.update(generate_income_data(entry['incomes']))

#   if(new_entry['total_bills'] >= 50000 or new_entry['n_bills'] < 20 or new_entry['n_bills'] > 100 or new_entry['total_bills']/new_entry['n_bills'] < 2):
#      continue

    new_dataset.append(new_entry)

print(len(new_dataset[0]))
output_file = open('dataset.json', 'w')
output_file.write(json.dumps(new_dataset))
output_file.close()



