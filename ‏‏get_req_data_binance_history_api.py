# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 22:41:10 2023

@author: b.s
"""

import requests
import csv
import time

# Last 3 years until current time less than a month
current_time = int(time.time() * 1000)
three_years_ago = current_time - (3 * 365 * 24 * 60 * 60 * 1000)
params_train = {'symbol': 'DOGEUSDT', 'interval': '1d', 'startTime': str(three_years_ago), 'endTime': str(current_time - (30 * 24 * 60 * 60 * 1000))}

# One month back starting today
one_month_ago = current_time - (30 * 24 * 60 * 60 * 1000)
params_test = {'symbol': 'DOGEUSDT', 'interval': '1d', 'startTime': str(one_month_ago), 'endTime': str(current_time)}

url = 'https://api.binance.com/api/v3/klines'
headers = {'X-MBX-APIKEY': 'YOUR_API_KEY'}

# Fetch train data
response_train = requests.get(url, headers=headers, params=params_train)
if response_train.status_code == 200:
    data_train = response_train.json()
    
    # Convert JSON data to CSV
    with open('stock_data_train.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        for row in data_train:
            # Convert Unix timestamp to datetime
            date = time.strftime('%d/%m/%y', time.gmtime(row[0] / 1000))
            writer.writerow([date, row[1], row[2], row[3], row[4], row[5]])
            
    print('Train data saved to stock_data_train.csv file.')
else:
    print('Error:', response_train.status_code, response_train.reason)

# Fetch test data
response_test = requests.get(url, headers=headers, params=params_test)
if response_test.status_code == 200:
    data_test = response_test.json()
    
    # Convert JSON data to CSV
    with open('stock_data_test.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        for row in data_test:
            # Convert Unix timestamp to datetime
            date = time.strftime('%d/%m/%y', time.gmtime(row[0] / 1000))
            writer.writerow([date, row[1], row[2], row[3], row[4], row[5]])
            
    print('Test data saved to stock_data_test.csv file.')
else:
    print('Error:', response_test.status_code, response_test.reason)
