import pandas
import matplotlib.pyplot as plt
import numpy as np

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))

dataset_2['Total']                = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))
dataset_2['Low Temp']             = pandas.to_numeric(dataset_2['Low Temp'].replace(',','', regex=True))
dataset_2['High Temp']            = pandas.to_numeric(dataset_2['High Temp'].replace(',','', regex=True))
dataset_2['Precipitation']        = pandas.to_numeric(dataset_2['Precipitation'].replace(',','', regex=True))


# Problem 1: You want to install sensors on the bridges to estimate overall
# traffic across all the bridges. But you only have enough budget to install
# sensors on three of the four bridges. Which bridges should you install the
# sensors on to get the best prediction of overall traffic?
def problem1():
    bridges = ['Queensboro Bridge', 'Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge']

    errors = []
    total = dataset_2[bridges].to_numpy()
    avg = np.mean(total, axis=1)
    for bridge in bridges:
        data = dataset_2[bridge].to_numpy()

        error = np.sum(np.abs(data - avg) / avg) / len(avg)
        errors.append(error)
    
    output(errors)

    return()


# Prints and format all the output and plots
# Input: errors - list of percent error
# Output:none
def output(errors):
    brook = dataset_2['Brooklyn Bridge'].values
    man = dataset_2['Manhattan Bridge'].values
    queen = dataset_2['Queensboro Bridge'].values
    will = dataset_2['Williamsburg Bridge'].values
    total = dataset_2['Total'].values

    plt.scatter(brook, total, color = 'blue', label="Brookyln Bikers")
    plt.scatter(man, total, color = 'red', label = "Manhattan Bikers")
    plt.scatter(queen, total, color = 'green', label = "Queensboro Bikers")
    plt.scatter(will, total, color = 'purple', label = "Williamsburg Bikers")
    plt.xlabel("Bikers at Each Bridge")
    plt.ylabel("Total Bikers")
    plt.title("Correlation of Bikers to Bridges")
    plt.legend()
    plt.show()

    print(errors)
    return()

problem1()
