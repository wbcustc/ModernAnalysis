# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import seaborn as sns
sns.set(color_codes=True) #Used Seaborn Library

files = open("iris.data")
iris_data = []  #N*p matrix which stores the features
iris_target = []  #N*1 column vector which stores the species
for line in files:
    line = line.strip('\t\n\r')
    if line == '':
        continue
    words = line.split(',')
    count = 0
    temp = []
    for word in words:
        count = count + 1
        if count == 5:
            if word == "Iris-setosa":
                iris_target.append(0)
            if word == "Iris-versicolor":
                iris_target.append(1)
            if word == "Iris-virginica":
                iris_target.append(2)
        else:
            temp.append(float(word))
    iris_data.append(temp)
print(iris_data)
print(iris_target)

iris = sns.load_dataset('iris')  #Used Seaborn Library
g = sns.pairplot(iris,hue="species")  #Used Seaborn Library
g.savefig("pairfig.png")  #Used Seaborn Library
