#!/usr/bin/env python
# coding: utf-8

import requests

url='http://localhost:9696/predict'

engine={"engine_rpm": 628,
       "lub_oil_pressure":4.645944,
       "fuel_pressure": 5.287920,
       "coolant_pressure": 3.032704,
       "lub_oil_temp":76.417099,
       "coolant_temp":82.734634,
       }

response=requests.post(url, json=engine).json()

if response['churn']==True:
    print("Engine status: Healthy")
else:
    print("Engine needs servicing")