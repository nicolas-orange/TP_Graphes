import numpy as np
import csv
import matplotlib.pyplot as plt

csvfile=open("./test.csv")
for i in csvfile:
    print (i)

arr=np.loadtxt("./test.csv",delimiter=";",dtype=str)
print(arr)
print("l element 2 de la ligne 7 est:") 
print(arr[7][2])

with open("./test.csv", 'r') as parse:
    sample=list(csv.reader(parse,delimiter=";"))

sample_array=np.array(sample)
print(sample_array)

