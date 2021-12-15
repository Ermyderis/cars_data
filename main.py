# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 06:34:04 2021

@author: dovyd
"""
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import pickle
from tqdm import tqdm

data = pd.read_csv("car.data")
#print(data.head())
#Format data
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))


predict = "safety"  #optional

X = list(zip(buying, maint , door, persons, lug_boot, safety))
y = list(safety)  #use same as predict

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
best = 0
def train_and_save_model(best):
    #train model
    for i in tqdm (range (1000), desc="Loading..."):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
        model = KNeighborsClassifier(n_neighbors = 100)

        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        #print(i , " ", acc)
        
        #save model
        if acc > best:
            best = acc
            with open("cars.pickle", "wb") as f:
                pickle.dump(model, f)
    return best
                
print("\nBest: ", best)
def print_predictions():
    pickle_in = open("cars.pickle", "rb")
    model = pickle.load(pickle_in)
    
    predicted = model.predict(x_test)
    #names = ["unacc", "acc", "good", "vgood"]
    names = ["low", "med", "high"]
    
    good_predictions = 0
    bad_predictions = 0
    
    for x in range(len(x_test)):
        if predicted[x] == y_test[x]:
            good_predictions = good_predictions + 1
            print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
        else:
            bad_predictions = bad_predictions + 1
            print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    print("All predictions: ", x+1)
    print("Good predictions: ", good_predictions)
    print("Bad predictions: ", bad_predictions)

best = train_and_save_model(best)
print_predictions()