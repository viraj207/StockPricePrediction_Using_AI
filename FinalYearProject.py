#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
#import wbgapi as wb
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
from fbprophet import Prophet
from sklearn.linear_model import LinearRegression


# In[14]:


get_ipython().system('pip install pandas')


# In[47]:


from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.graph_objs as go
from matplotlib.figure import Figure
from plotly.offline import plot
import plotly.io as pio
from PIL import ImageTk, Image
import plotly
from fbprophet.plot import add_changepoints_to_plot


# In[48]:


#Creating the main page
main_screen = tk.Tk()
main_screen.resizable(False,False) #disabling resizing of application, to maintain streamlines GUI

today = date.today()

main_screen.title('Stock Shark')
main_screen.geometry('470x720')

def open_url(url):
   webbrowser.open_new_tab(url)


        


# In[49]:


#Saving user input value as ticker
date_error = ""
ticker_error=""

def save_text(event):
    global ticker
    ticker= txt_fld_1.get()
    global integer
    integer = int(int_entry.get())
    print(ticker)
    with open("nasdaq_screener_1664449346258TEXT.txt") as f:
       text = f.readlines()
       text = [x.strip() for x in text]
       print(integer)
       delta = end_date - start_date
       date_limit = delta.days
       #text = all of the ticker symbols as a list
       if ticker in text and start_date < end_date and start_date != end_date and date_limit >=101: #Checking Valid Dates have been entered
          print("EXIST")
          print("GOOD DATES")
          target_date= today + timedelta(days=integer)
          target_date_string = datetime.strftime(target_date, '%Y-%m-%d')
          target_date_string = datetime.strftime(target_date, '%Y-%m-%d')
          global date_range_pred
          date_range_pred = pd.date_range(start=today,end = target_date)
          
          print(target_date)

        
          print(date_limit)
          

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

start_date = 0
end_date = 0

def on_date_selected_cal1(event):
    selected_date1 = event.widget.selection_get()
    print(selected_date1)
    global start_date
    start_date = selected_date1
    
def on_date_selected_cal2(event):
    selected_date2 = event.widget.selection_get()
    print(selected_date2)
    global end_date
    end_date = selected_date2
           
def save_text_2():
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
        


# In[50]:


def invalid_ticker():
   error= tk.Tk()
   error.geometry("330x250")
   error.title("Error")
   lbl_9=tk.Label(error,text="Ticker Symbol Not Found",fg='red',font=("Helvatica",15))
   lbl_9.grid(row=1, column=1, padx=5, pady=5 , sticky=tk.NSEW)
   lbl_10=tk.Label(error,text="Please ensure you have entered the ticker symbol, \n exactly as displayed on Yahoo Finance.",font=("Helvatica",10))
   lbl_10.grid(row=2, column=1, padx=5, pady=5 , sticky=tk.NSEW)
   
def invalid_dates():
   d_error = tk.Tk()
   d_error.geometry("450x250")
   d_error.title("Error")
   lbl_11=tk.Label(d_error,text="Invalid Dates",fg='red',font=("Helvatica",15))
   lbl_11.grid(row=1, column=1, padx=5, pady=5 , sticky=tk.NSEW)
   lbl_12=tk.Label(d_error,text="The Dates you have entered are invalid. Please \n ensure the  START DATE is before your END DATE. \n Also ensure there is at least 101 days between your start and end date",font=("Helvatica",10))
   lbl_12.grid(row=2, column=1, padx=5, pady=5 , sticky=tk.NSEW)
    
    



# In[51]:


def open_results():
   results = tk.Tk()
   results.title(ticker + " Prediction Result")
   results.geometry("2000x1000")
   results.protocol("WM_DELETE_WINDOW", results.destroy)
    
#########################################################################################
#LSTM ALGORITHM
   stock_data = yf.download(ticker, start= start_date, end= end_date)
   stock_data.head()
    

   close_prices = stock_data['Close']
   values = close_prices.values
   training_data_len = math.ceil(len(values)* 0.9)

   scaler = MinMaxScaler(feature_range=(0,1))
   scaled_data = scaler.fit_transform(values.reshape(-1,1))
   train_data = scaled_data[0: training_data_len, :]

   x_train = []
   y_train = []

   for i in range(60, len(train_data)):
       x_train.append(train_data[i-60:i, 0])
       y_train.append(train_data[i, 0])
     
   x_train, y_train = np.array(x_train), np.array(y_train)
   x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

   test_data = scaled_data[training_data_len-60: , : ]
   x_test = []
   y_test = values[training_data_len:]




   for i in range(60, len(test_data)):
     x_test.append(test_data[i-60:i, 0])

   x_test = np.array(x_test)
   x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


   model = keras.Sequential()
   model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
   model.add(layers.LSTM(100, return_sequences=False))
   model.add(layers.Dense(25))
   model.add(layers.Dense(1))
   model.summary()
    
   model.compile(optimizer='adam', loss='mean_squared_error')
   model.fit(x_train, y_train, batch_size= 1, epochs=1)
    
   predictions = model.predict(x_test)
   predictions = scaler.inverse_transform(predictions)
    
   forecast = np.zeros((integer, 1))
   inputs = scaled_data[-60:].reshape(1, -1, 1)
   

   for i in range(integer):
      prediction = model.predict(inputs)
      forecast[i] = prediction
      inputs = np.append(inputs[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
  
   forecast = scaler.inverse_transform(forecast)
   last_date = pd.to_datetime(stock_data.index[-1])
   date_range = pd.date_range(last_date, periods=integer+1, freq='D')[1:]

    
   fig4, ax4 = plt.subplots()
   ax4.set_title('RNN FUTURE PREDICTION')
   ax4.plot(stock_data.index, values, label='Actual')
   ax4.plot(date_range, forecast, label='Forecast')
   ax4.legend(loc='lower right')
   rmse = np.sqrt(np.mean(predictions - y_test)**2)
   rmse
    

   print(date_range_pred)

   data = stock_data.filter(['Close'])
   train = data[:training_data_len]
   validation = data[training_data_len:]
   validation['Predictions'] = predictions
   

   fig, ax = plt.subplots()
   #ax = fig.add_subplot(121)
   ax.set_title(ticker + ' Stock Prices History')
   ax.plot(stock_data['Close'])
   ax.set_xlabel('Date')
   ax.set_ylabel('Prices ($ USD )')
    


   fig2, ax2 = plt.subplots()
   #ax2 = fig2.add_subplot(122)
   ax2.set_title('Stock Price Prediction Using a Nerual Network')
   ax2.set_xlabel('Date')
   ax2.set_ylabel('Close Price USD ($)')
   ax2.plot(train)
   ax2.plot(validation[['Close', 'Predictions']])
   ax2.legend(['Train', 'Val', 'Predictions'], loc='lower right')
   #fig2.show()
   
   upper_frame = tk.Frame(results)
   upper_frame.pack(fill="both",expand=True)

   lower_frame = tk.Frame(results)
   lower_frame.pack(fill="both",expand=True)
    
   canvas1 = FigureCanvasTkAgg(fig, upper_frame)
   canvas1.draw()
   canvas1.get_tk_widget().pack(side="left", fill="both",expand=True)

   canvas2 = FigureCanvasTkAgg(fig2, upper_frame)
   canvas2.draw()
   canvas2.get_tk_widget().pack(side="right", fill="both",expand=True)
    
   canvas4 = FigureCanvasTkAgg(fig4, lower_frame)
   canvas4.draw()
   canvas4.get_tk_widget().pack(side="right", fill="both",expand=True)
    
    

    
   def predictFuture_TS():
      
      stock_data = data.reset_index().rename(columns={'index': 'Date'})
      ts_stockData = stock_data[["Date", "Close"]]
      ts_stockData.columns = ["ds", "y"]
      print("TS Prediction")
      print(ts_stockData)
      import plotly.graph_objs as go
        
      #TRAINING THE DATA
    
      prophet = Prophet(daily_seasonality=True)
      prophet.fit(ts_stockData)
      integer = int(int_entry.get())
      print("TEST",integer)
      future_dates = prophet.make_future_dataframe(periods = integer)
      ts_prediction = prophet.predict(future_dates)
      from fbprophet.plot import plot_plotly
      fig_ts = plot_plotly(prophet, ts_prediction)
      fig_ts.show()
      ts_forecasted = ts_prediction[-integer:]

      fig_ts2 = plot_plotly(prophet, ts_prediction)

      fig3, ax3 = plt.subplots()
      ax3.plot(fig_ts.data[0]['x'], fig_ts2.data[0]['y'])
      #ax3.plot(ts_forecasted['ds'], ts_forecasted['yhat'], color='red' if ts_forecasted['ds'].max() > ts_stockData['ds'].max() else 'green', label='Predicted')
      ax3.plot(ts_forecasted['ds'], ts_forecasted['yhat'], color='green', label='Predicted' if ts_forecasted['ds'].max() <= ts_stockData['ds'].max() else None, linewidth=2)
      
      ax3.plot(ts_stockData['ds'], ts_stockData['y'], color='blue', label='Actual')
      ax3.plot(ts_forecasted['ds'], ts_forecasted['yhat'], color='red', label='Predicted' if ts_forecasted['ds'].max() > ts_stockData['ds'].max() else None, linewidth=2)
     
    
      ax3.legend(loc='lower right')
      ax3.set_xlabel('Date')
      ax3.set_ylabel('Close Price')
      ax3.set_title('Predicted Future Close Prices Using Time Series Forecasting')
      plt.show()
   
      canvas3 = FigureCanvasTkAgg(fig3, lower_frame)
      canvas3.draw()
      canvas3.get_tk_widget().pack(side="right", fill="both",expand=True)
    
   def linear_reg_prediction():
       #Dowloading Live GDP Data relevant to dates selected by user
      import wbdata #importing data from world bank
      


   predictFuture_TS() 
   linear_reg_prediction()
       
##########################################################################################


   def Close_r():
      results.destroy()
      main_screen.deiconify()
   back_button = Button(results,text="Back", command=Close_r)
   back_button.pack()


# In[52]:


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


lbl_4= tk.Label(main_screen,text="In order to use this application to predict the future price of your desired Asset,\n You will need to train the neural network with historical price action data.\n Please chose a Start and End date. \n There must be at least 101 days between the START and END dates", font=("Helvetica", 10))
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
#Use price lookback instead



btn_1 = tk.Button(main_screen, text="Submit",fg='blue',command= save_text_2)
btn_1.grid(row=11,column=2, padx=5, pady=5, sticky=tk.NSEW)

####

main_screen.bind('<Return>',save_text)



# https://medium.com/the-handbook-of-coding-in-finance/stock-prices-prediction-using-long-short-term-memory-lstm-model-in-python-734dd1ed6827#:~:text=Save-,Stock%20Prices%20Prediction%20Using%20Long%20Short,Memory%20(LSTM)%20Model%20in%20Python&text=Long%20Short%2DTerm%20Memory%20(LSTM)%20is%20one%20type%20of,useful%20in%20predicting%20stock%20prices.
#update_tickers_listbox()
#https://pypi.org/project/world-bank-data/
main_screen.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




