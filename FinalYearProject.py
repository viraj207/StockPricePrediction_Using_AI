#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import *
import webbrowser
import csv
import yfinance as yf
import autocomplete as Autocomplete
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkcalendar import Calendar
from datetime import date
from datetime import datetime, timedelta
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from math import sqrt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import confusion_matrix, accuracy_score
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error


# In[21]:


from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.graph_objs as go
from matplotlib.figure import Figure
from plotly.offline import plot
import plotly.io as pio
from PIL import ImageTk, Image
import plotly



# In[22]:


#Creating the main page
main_screen = tk.Tk()
                                    #disabling resizing of application, to maintain streamlines GUI
main_screen.resizable(False,False)        
                                    #Getting todays date, will be used in application
today = date.today()
                                    #Setting the title of the main Tk window
main_screen.title('Stock Shark')
                                    #Setting a appropriate size for the TK window
main_screen.geometry('470x850')
                                    #Creating a function that allows me to open any URL passed in
def open_url(url):
   webbrowser.open_new_tab(url)


        


# In[23]:


#Saving user input value as ticker
#disabling resizing of application, to maintain streamlines GUI
def save_text(event):       #Function to Save , and chech user input after button is selected
    global ticker
    ticker= txt_fld_1.get()
    global integer
    integer = int(int_entry.get())  #Storing the User input as Ticker
    print(ticker)
    with open("nasdaq_screener_1664449346258TEXT.txt") as f: #Opening locally stored file containing all NYSE ticker symbold as of 19/12/2022
       text = f.readlines()             #read all lines in the csv
       text = [x.strip() for x in text] #removing any white spaces from the text
       print(integer)
       delta = end_date - start_date
       date_limit = delta.days
       #text = all of the ticker symbols as a list
       if ticker in text and start_date < end_date and start_date != end_date and date_limit >=101: #Checking Valid Dates have been entered
          print("EXIST")
          print("GOOD DATES")
          target_date= today + timedelta(days=integer)                    #target date is the date in the future the user will forecast for
          target_date_string = datetime.strftime(target_date, '%Y-%m-%d') #Converting the target date to a string
          global date_range_pred
          date_range_pred = pd.date_range(start=today,end = target_date)
          
          print(target_date)
        
          print(date_limit)
          
          open_results()
       elif start_date == end_date or start_date > end_date: #invalid input handling if user enters invalid data that could crash the application
          invalid_dates()
       elif ticker not in text:    
          print("NO EXIST")
          invalid_ticker()
       elif date_limit < 101:
          invalid_dates()
       else :
          invalid_dates()
          invalid_ticker()

start_date = 0
end_date = 0

def on_date_selected_cal1(event):                #functions needed to be created so Enter key can also be used to call the next function
    selected_date1 = event.widget.selection_get()
    print(selected_date1)
    global start_date
    start_date = selected_date1
    
def on_date_selected_cal2(event):
    selected_date2 = event.widget.selection_get()
    print(selected_date2)
    global end_date
    end_date = selected_date2
           
def save_text_2():            #done again for mouse click on button
    global ticker 
    global integer
    ticker= txt_fld_1.get()   
    integer = int(int_entry.get())
    print(ticker)
    with open("nasdaq_screener_1664449346258TEXT.txt") as f:
       text = f.readlines()
       text = [x.strip() for x in text]
       integer = int(int_entry.get())
       print(integer)
       delta = end_date - start_date
       date_limit = delta.days
       #text = all of the ticker symbols as a list
       if ticker in text and start_date < end_date and start_date != end_date and date_limit >=101: #Checking Valid Dates have been entered
          print("EXIST")
          print("GOOD Dates")
          target_date= today + timedelta(days=integer)
          target_date_string = datetime.strftime(target_date, '%Y-%m-%d')
          global date_range_pred
          date_range_pred = pd.date_range(start=today,end = target_date)
          print(date_range_pred)
          
          print(target_date)
       
          open_results()
            
       elif start_date == end_date or start_date > end_date:
          invalid_dates()
       elif ticker not in text:    
          print("NO EXIST")
          invalid_ticker()
       elif date_limit < 101:
          invalid_dates()
       else :
          invalid_dates()
          invalid_ticker()
        


# In[24]:


def invalid_ticker():     #Function for if user has inputted an invalid Ticker Symbol
   error= tk.Tk()
   error.geometry("330x250")
   error.title("Error")
   lbl_9=tk.Label(error,text="Ticker Symbol Not Found",fg='red',font=("Helvatica",15))
   lbl_9.grid(row=1, column=1, padx=5, pady=5 , sticky=tk.NSEW)
   lbl_10=tk.Label(error,text="Please ensure you have entered the ticker symbol, \n exactly as displayed on Yahoo Finance.",font=("Helvatica",10))
   lbl_10.grid(row=2, column=1, padx=5, pady=5 , sticky=tk.NSEW)
   
def invalid_dates():  #Function for if user inputs invalid dates to train the model
   d_error = tk.Tk()
   d_error.geometry("450x250")
   d_error.title("Error")
   lbl_11=tk.Label(d_error,text="Invalid Dates",fg='red',font=("Helvatica",15))
   lbl_11.grid(row=1, column=1, padx=5, pady=5 , sticky=tk.NSEW)
   lbl_12=tk.Label(d_error,text="The Dates you have entered are invalid. Please \n ensure the  START DATE is before your END DATE. \n Also ensure there is at least 101 days between your start and end date",font=("Helvatica",10))
   lbl_12.grid(row=2, column=1, padx=5, pady=5 , sticky=tk.NSEW)
    
    



# In[25]:


def open_results():
   results = tk.Tk()
   results.title(ticker + " Prediction Result")
   results.geometry("2000x1000")
   results.protocol("WM_DELETE_WINDOW", results.destroy)
    
########################################################################################    
   stock_data = yf.download(ticker, start= start_date, end= end_date) # dowloading the relevant stock data between the inputted dates from YahooFinance
   stock_data.head()
   print(ticker,"Stock price History", stock_data)
#########################################################################################

#Getting US Inlfation Data from Nasdaq Data Link

   import nasdaqdatalink #Importing nasdaqdatalink so iu can download uptodate US Inflation data
   import quandl
   #DOWNLOADING US INLFATION DATA BETWEEN THE DATES CONCERNED
   #In other iterations of this software I can change to import other country inflation data
  
   nasdaqdatalink.ApiConfig.api_key = 'e8BQLbDFvrUKmfEozNty'  # My unique API key so i can use nasdaq data link
   #Only Getting dates Relecant for training data
   USA_inflation = nasdaqdatalink.get("RATEINF/INFLATION_USA",  start_date=start_date, end_date=end_date)
   USA_CPI = nasdaqdatalink.get("RATEINF/CPI_USA",  start_date=start_date, end_date=end_date) 
   print ("US INFLATION DATA :",USA_inflation)
   print ("US CPI DATA :",USA_CPI)
   #merging data so CPI and Inflation data can be used to make prediction
   merged_data = pd.merge(stock_data,USA_inflation, on='Date')
   merged_data = pd.merge(merged_data,USA_CPI,on ='Date') 
   merged_data = merged_data.rename(columns={'Value_x': 'Inflation', 'Value_y': 'CPI'}) 
   

   print("mergeddata",merged_data.head())
    
   # both fig5 and fig5 are matplot lib charts that display the CPI and Inflation data for the relevant dates
   fig5, ax5 = plt.subplots()
   ax5.set_title('US CPI Data')
   ax5.plot(merged_data.index, merged_data['CPI'], label='CPI')
   ax5.set_xlabel('Date')
   ax5.set_ylabel('CPI Index')


   fig6, ax6 = plt.subplots()
   ax6.set_title('US Inflation Data')
   ax6.plot(merged_data.index, merged_data['Inflation'], label='Inflation')
   ax6.set_xlabel('Date')
   ax6.set_ylabel('Inflation Rate (%)')
















##################################################################################################








#LSTM ALGORITHM
   
                                                           #selecting the column of close prices from the stock_data dataframe.
   close_prices = stock_data['Close']
                                                           #converting the close prices to a numpy array.
   values = close_prices.values
                                                           #calculating the length of the training data as 90% of the total length of the data.
   training_data_len = math.ceil(len(values)* 0.9)
                                                           #MinMaxScaler to scale the data to a range of 0 to 1.
   scaler = MinMaxScaler(feature_range=(0,1))
   scaled_data = scaler.fit_transform(values.reshape(-1,1))
                                                           #selecting the training data from the scaled data
   train_data = scaled_data[0: training_data_len, :]
                                                           #creating empty lists x_train and y_train to hold the training data.
   x_train = []
   y_train = []

   for i in range(60, len(train_data)):
       x_train.append(train_data[i-60:i, 0])
       y_train.append(train_data[i, 0])
                                                            #converting x_train and y_train from lists to numpy arrays
   x_train, y_train = np.array(x_train), np.array(y_train)
                                                            #Reshaping x_train to be 3-dimensional for use in LSTM
   x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

                                                            #Selecting the test data, which is the data after the training data plus the 60-day window for input.
   test_data = scaled_data[training_data_len-60: , : ]
   x_test = []
   y_test = values[training_data_len:]



                                                           #Looping through the test data and append the 60-day window to x_test
   for i in range(60, len(test_data)):
     x_test.append(test_data[i-60:i, 0])
    
   x_test = np.array(x_test)                               #Convert test data to x_test and reshape x_test to 3D                              
   x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                                                            #Building LSTM model with 2 LSTM layers and 2 dense layers
   LSTM_model = keras.Sequential()
   LSTM_model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
   LSTM_model.add(layers.LSTM(100, return_sequences=False))
   LSTM_model.add(layers.Dense(25))
   LSTM_model.add(layers.Dense(1))
   LSTM_model.summary()
    
   
                                                             #Compile and fit LSTM model
   LSTM_model.compile(optimizer='adam', loss='mean_squared_error')
   LSTM_model.fit(x_train, y_train, batch_size= 1, epochs=1) #<---------- CHANGE BACK TO 20
    
    

                                                             #Predicting the stock price using the trained model
   predictions = LSTM_model.predict(x_test)
   predictions = scaler.inverse_transform(predictions)
    
   mse = mean_squared_error(y_test, predictions)             #Calculating the root mean squared error
   rmse = np.sqrt(mse)
   print("Root Mean Squared Error:", rmse)
   
                                                             #Extending model to forecast for future dates  
   
   new_delta = end_date - start_date
   date_range_int = new_delta.days
                                                             #Creating an array of zeros to hold the forecasted stock prices
   forecast = np.zeros((integer, 1))
                                                             #Setting inputs to the most recent 'date_range_int' number of data points. where date_int_range, is number of days between start and end
   inputs = scaled_data[-date_range_int:].reshape(1, -1, 1)
   
                                                             #Interating 'integer' times to create a forecast for the next 'integer' days.
   for i in range(integer):
      prediction = LSTM_model.predict(inputs)
      forecast[i] = prediction
      inputs = np.append(inputs[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
                                                              #Inverse scaling the forecasted stock prices
   forecast = scaler.inverse_transform(forecast)
   last_date = pd.to_datetime(stock_data.index[-1])
                                                              #Creating a date range for the forecasted dates
   date_range = pd.date_range(last_date, periods=integer+1, freq='D')[1:]

    
    #fig4 is the chart containing the forecasted prediction dates
   fig4, ax4 = plt.subplots()
   ax4.set_title('LSTM FUTURE PREDICTION')
   ax4.plot(stock_data.index, values, label='Actual')
   ax4.plot(date_range, forecast, label='Forecast')
   ax4.legend(loc='lower right')
   
    

   print(date_range_pred)

   data = stock_data.filter(['Close'])
   train = data[:training_data_len]
   validation = data[training_data_len:]
   validation['Predictions'] = predictions
   

   fig, ax = plt.subplots()
   ax.set_title(ticker + ' Stock Prices History')
   ax.plot(stock_data['Close'])
   ax.set_xlabel('Date')
   ax.set_ylabel('Prices ($ USD )')
   ax.legend(loc='lower right')
   
    


   fig2, ax2 = plt.subplots()
   #ax2 = fig2.add_subplot(122)
   ax2.set_title('Stock Price Prediction Using LSTM')
   ax2.set_xlabel('Date')
   ax2.set_ylabel('Close Price USD ($)')
   ax2.plot(train)
   ax2.plot(validation[['Close', 'Predictions']])
   ax2.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
   #fig2.show()
   
                                            #Creating 3 frames wherei canp plot my chart using the Tkinter and Matplotlib
   upper_frame = tk.Frame(results)
   upper_frame.pack(fill="both",expand=True)

   middle_frame = tk.Frame(results)
   middle_frame.pack(fill="both",expand=True)
    
   lower_frame = tk.Frame(results)
   lower_frame.pack(fill="both",expand=True)

    
    
    
   #CanvasN here im creating frames within the results TK window, so that i can plot my all my charts in the TK window
   canvas1 = FigureCanvasTkAgg(fig, upper_frame)
   canvas1.draw()
   canvas1.get_tk_widget().pack(side="left", fill="both",expand=True)

   canvas2 = FigureCanvasTkAgg(fig2, upper_frame)
   canvas2.draw()
   canvas2.get_tk_widget().pack(side="right", fill="both",expand=True)
       
   canvas4 = FigureCanvasTkAgg(fig4, lower_frame)
   canvas4.draw()
   canvas4.get_tk_widget().pack(side="right", fill="both",expand=True)
 
   canvas5 = FigureCanvasTkAgg(fig5, middle_frame)
   canvas5.draw()
   canvas5.get_tk_widget().pack(side="right", fill="both",expand=True)
    
   canvas6 = FigureCanvasTkAgg(fig6, middle_frame)
   canvas6.draw()
   canvas6.get_tk_widget().pack(side="right", fill="both",expand=True)

    #Time SereiesPrediction Algorithm
   def predictFuture_TS():
#Splitting data into training and test data
      train_data, test_data = stock_data[0:int(len(stock_data)*0.9)], stock_data[int(len(stock_data)*0.8):]
## Extracting the closing prices of the training and test sets
      testArray = test_data['Close'].values
      trainArray = train_data['Close'].values
# Using auto_arima function to automatically find the optimal values for p, d and q parameters of the ARIMA model
# by minimizing the Akaike Information Criterion (AIC) value        
      stepwise_fit = auto_arima(trainArray, start_p=0, start_q=0, max_p=5, max_q=5, m=12,seasonal=False, d=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
# Extracting the optimal values of p, d and q parameters 
      order = stepwise_fit.order
 
        
        
      history = [x for x in trainArray]
      predictions = list()
## Using ARIMA model to predict the closing prices of the test set
      for t in range(len(testArray)):
         model = ARIMA(history, order=(order))
         model_fit = model.fit()
         output = model_fit.forecast()
         yhat = output[0]
         predictions.append(yhat)
         obs = testArray[t]
         history.append(obs)
#Predicting for n days in the future
      
      last_date = test_data.index[-1]
      future_dates = pd.date_range(start=last_date, periods=integer, freq='D')
      future_predictions = []
  #Same as above bue extending to forecast 'integer' times      
      for i in range(integer):
         model_fit = ARIMA(history, order=(order)).fit() 
         output = model_fit.forecast()
         yhat = output[0]
# Adding the predicted price to the future_predictions list
         future_predictions.append(yhat)
# Adding the predicted price to the history list
         history.append(yhat)
## Creating a DataFrame with the predicted prices and dates for the future period
      future_data = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
      print(future_data)

#      rmse = sqrt(mean_squared_error(test, predictions))
#      print(rmse)
      fig3, ax3= plt.subplots()
      ax3.set_title("Stock Price Prediction unsing Time Series forecasting")
      ax3.plot(stock_data['Open'], 'green', color='blue', label='Training Data')
      ax3.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', label='Predicted Price')
      ax3.plot(future_data['Date'], future_data['Predicted Price'], color='purple', marker='o', linestyle='dashed', label='Future Predictions')
      
      

    
    
      ax3.set_xlabel('Dates')
      ax3.set_ylabel('Prices')
      ax3.legend(loc='lower right')

      
      
#      ax3.legend(loc='lower right')
#      ax3.set_xlabel('Date')
#      ax3.set_ylabel('Close Price')
#      ax3.set_title('Predicted Future Close Prices Using Time Series Forecasting')
#      plt.show()
   
      canvas3 = FigureCanvasTkAgg(fig3, lower_frame)
      canvas3.draw()
      canvas3.get_tk_widget().pack(side="left", fill="both",expand=True)
    
    
   predictFuture_TS() 

   def linear_reg_prediction():
      #Smoothing out the daily CPI and inflation data, as its a monthly value, so it returns less predictions  
      daily_CPI = USA_CPI.resample('D').ffill()
      #print (daily_CPI)
      daily_inflation = USA_inflation.resample('D').ffill()
    
#Merge daily inflation and CPI data with stock data
      merged_data2 = pd.merge(stock_data, daily_inflation, on='Date')
      merged_data2 = pd.merge(merged_data2, daily_CPI,on ='Date') 
#Rename columns as merging changed the name
      merged_data2 = merged_data2.rename(columns={'Value_x': 'Inflation', 'Value_y': 'CPI'}) 
#Resample the data to fill missing dates
      merged_data2 = merged_data2.resample('D').ffill()
      print (merged_data2)

#Sort the data by date as mereging changed the order of data for some reason

      merged_data2 = merged_data2.sort_values('Date')
#Split the data into training and testing sets
      x = merged_data2[['CPI', 'Inflation','High']]
      y = merged_data2[['Close']]
      
#Splitting the features and target variables into train and test sets with a 75/25 ratio
#x: features variable which include 'CPI', 'Inflation' and 'High'
#y: target variable which is the 'Close' price
      x_train, x_test , y_train , y_test = train_test_split (x, y, random_state = 0)
#Train a linear regression model on the training data
      reg_model = LinearRegression()
      reg_model.fit(x_train, y_train)
#Make predictions on the testing data
      prediction = reg_model.predict(x_test)
#Create a dataframe with the actual and predicted values      
      y_pred = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
      y_pred = y_pred.sort_index()
      print (y_pred)
      #rmse = sqrt(mean_squared_error(test, predictions))
      #print(rmse)
      ############################################################################################

       
    
#      #extend to forcast for n days in the future
#      last_n_days = stock_data.tail(integer)
#       
#      #next_day = reg_model.predict(last_n_days)
#    
#      forecast = []
#        
#      for i in range(integer):
#        # use the last n days and the last predicted value as input to the model to predict the next day
#         last_n_days = last_n_days[['CPI', 'Inflation','High','Low','Volume']].shift(-1).iloc[:-1]
#         next_day = reg_model.predict(last_n_days)
#    
#      # add the predicted value to the list of predictions
#      forecast.append(next_day)
#      forecasted_values = pd.Series(np.concatenate(forecast), index=pd.date_range(start=last_n_days.index[-1], periods=len(forecast)))
#        
#      print(forecasted_values)
    
      

      fig6, ax6 = plt.subplots()

      plt.scatter(y_pred.index, y_pred['Actual'], color='black', label='Actual')
      plt.scatter(y_pred.index, y_pred['Predicted'], color='blue', label='Predicted')
      ax6.plot(stock_data['Close'])
        
      ax6.set_title('Linear Regression Model')
      ax6.set_xlabel('Date')
      ax6.set_ylabel('Closing Price')
      ax6.legend(['Actual', 'Predicted'])
      plt.show()
        
        
      
      
      
   
 

        
        


        
#   linear_reg_prediction()



       
##########################################################################################


   def Close_r():
      results.destroy()
      main_screen.deiconify()
   back_button = Button(results,text="Back", command=Close_r)
   back_button.pack()


# In[ ]:


lbl_0= tk.Label(main_screen,text=("Today's date is : " + today.strftime("%Y-%m-%d")), font=("Helvetica", 10))
lbl_0.grid(row=0, column=2, padx=5, pady=5 , sticky=tk.NSEW)

            
                         
lbl_1= tk.Label(main_screen,text="Enter A Ticker symbol exactly as displayed on Yahoo Finance.", fg='red', font=("Helvetica", 10))
lbl_1.grid(row=1, column=2, padx=5, pady=5 , sticky=tk.NSEW)
main_screen.grid_rowconfigure(2,weight = 1)

# Label linking to Yahoo Finance website
lbl_3=tk.Label(main_screen,text="https://finance.yahoo.com/",fg='blue',font=("Helvatica",10))
lbl_3.bind("<Button-1>", lambda e:open_url("https://finance.yahoo.com/"))
lbl_3.grid(row=2, column=2, padx=5, pady=5 , sticky=tk.NSEW)

# Text box for user to enter Ticker symbol
txt_fld_1 = tk.Entry(main_screen,bg='black',fg='white',bd=5)
txt_fld_1.grid(row=3, column=2, padx=5, pady=5 , sticky=tk.NSEW)


lbl_4= tk.Label(main_screen,text="In order to use this application to predict the future price of your desired Asset,\n You will need to train the neural network with historical price action data.\n Please chose a Start and End date. \n There must be at least 101 days between the START and END dates \n \n Please Note the greater the date range\n the longer it will take to return a result", font=("Helvetica", 10))
lbl_4.grid(row=4 ,column=2, padx=5, pady=5 , sticky=tk.NSEW)

#Calendars to select dates to train Network on

lbl_5=tk.Label(main_screen,text="SELECT A START DATE",fg='blue',font=("Helvatica",10))
lbl_5.grid(row=5 ,column=2, padx=5, pady=5 , sticky=tk.NSEW)

#
cal1 = Calendar(main_screen, selectmode = 'day', year= 2023, month = 1,day = 2023 )
cal1.grid(row=6 ,column=2, padx=5, pady=5 , sticky=tk.NSEW)
cal1.bind("<<CalendarSelected>>",on_date_selected_cal1 )

lbl_6=tk.Label(main_screen,text="SELECT AN END DATE",fg='blue',font=("Helvatica",10))
lbl_6.grid(row=7 ,column=2, padx=5, pady=5 , sticky=tk.NSEW)

cal2 = Calendar(main_screen, selectmode = 'day', year= 2023, month = 1,day = 2023,maxdate= today )
cal2.grid(row=8 ,column=2, padx=5, pady=5 , sticky=tk.NSEW)
cal2.bind("<<CalendarSelected>>",on_date_selected_cal2 )

lbl_6 = tk.Label(main_screen,text="Enter a how many days in the future you would like to predict for.")
lbl_6.grid(row=9,column=2, padx=5,pady=5,sticky=tk.NSEW)

int_entry = Entry(main_screen, width=10)
int_entry.grid(row=10,column=2, padx=5, pady=5, sticky=tk.NSEW)



btn_1 = tk.Button(main_screen, text="Submit",fg='blue',command= save_text_2)
btn_1.grid(row=11,column=2, padx=5, pady=5, sticky=tk.NSEW)

####

main_screen.bind('<Return>',save_text) #Binding the enter key to the save_text function



# https://medium.com/the-handbook-of-coding-in-finance/stock-prices-prediction-using-long-short-term-memory-lstm-model-in-python-734dd1ed6827#:~:text=Save-,Stock%20Prices%20Prediction%20Using%20Long%20Short,Memory%20(LSTM)%20Model%20in%20Python&text=Long%20Short%2DTerm%20Memory%20(LSTM)%20is%20one%20type%20of,useful%20in%20predicting%20stock%20prices.
#https://pypi.org/project/world-bank-data/
#https://data.nasdaq.com/tools/python
main_screen.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




