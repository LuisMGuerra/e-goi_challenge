# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 01:58:20 2019

@author: luis
"""

import requests
import pandas as pd

df_request = pd.read_csv('request_sample.csv')
df_json = df_request.to_json(orient='records')

url = 'http://127.0.0.1:5000/'
r = requests.post(url, json=df_json)
print(r.json())

input("Press Enter to continue...")