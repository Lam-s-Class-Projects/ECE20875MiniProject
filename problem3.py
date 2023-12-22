import pandas
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

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
dataset_2['Day']                  = dataset_2['Day'].replace(',','', regex=True)

# Problem 3: Can you use this data to predict what day (Monday to Sunday) is today
# based on the number of bicyclists on the bridges?
def problem3():
    # Create a new array with integer values for days of the week
    days = dataset_2[['Day']]
    total = dataset_2[['Total']]
    logicRegression(total, days)
    return()

# Does logic Regression, this would provide us with an accuracy number between 0 - 1 (0% - 100%), so we can predict the accuracy of our dataset
# Input: total - data from the column 'Total' in our dataset, days - data from the column 'Day' in our dataset
# Output: none
def logicRegression(total, days):
    x_train, x_test, y_train, y_test = train_test_split(total, days, test_size = 0.2)

    logic = LogisticRegression()
    logic.fit(x_train, y_train.values.ravel())

    y_pred = logic.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(accuracy)

    return()


problem3()