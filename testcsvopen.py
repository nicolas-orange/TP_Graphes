import numpy
import pandas as pd
import datetime
import csv
import matplotlib.pyplot as plt

df = csv.reader('./test.csv', delimiter=';')
headers = df.next()
# ['col1','point','value1','value2']

for fields in df:
    print (fields)

