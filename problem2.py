import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

# Problem 2: The city administration is cracking down on helmet laws, and wants
# to deploy police officers on days with high traffic to hand out citations.
# Can they use the next day's weather forecast(low/high temperature and precipitation)
# to predict the total number of bicyclists that day?
def problem2():
    high = dataset_2['High Temp'].values
    low = dataset_2['Low Temp'].values
    total = dataset_2['Total'].values
    preci = dataset_2['Precipitation'].values
    ridgeRegression(high,total, "Correlation Between Temperature Highs and Number of Bikers","High Temperatures", "Total Bikers", "Bikers at Temperature High")
    ridgeRegression(low,total, "Correlation Between Temperature Low and Number of Bikers","Low Temperatures", "Total Bikers", "Bikers at Temperature Low")
    ridgeRegression(preci,total, "Correlation Between Precipitation and Number of Bikers","Precipitation", "Total Bikers", "Bikers at Precipitation")
    return()

# Does ridge regression and plots a scatter plot
# Input: x, y, graph title, labels of x and y, and scatter label
# Output: none
def ridgeRegression(x,y,graph_Title, x_Label, y_Label, scatter_Label):
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train_scaled, y_train)

    y_pred = ridge.predict(x_test_scaled)

    # plots
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(x_test, y_test, color = 'blue', label=scatter_Label)
    plt.xlabel(x_Label)
    plt.ylabel(y_Label)
    plt.title(graph_Title)
    plt.legend(fontsize=6)

    plt.subplot(1, 2, 2)
    plt.scatter(x_test, y_test, color = 'blue', label=scatter_Label)
    plt.plot(x_test, y_pred, color = 'red', linewidth=3, label="Predicted Number of Bikers")
    plt.xlabel(x_Label)
    plt.ylabel(y_Label)
    plt.title(graph_Title + "\nWith Regression Line")
    plt.legend(fontsize=10)

    plt.subplots_adjust(wspace=0.5)

    plt.show()
    return()

problem2()