#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as py
import tkinter as tk
from tkinter import *
import webbrowser
import csv
import yfinance as yf
import autocomplete as Autocomplete
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkcalendar import Calendar


# In[ ]:


#Creating the main page
main_screen = tk.Tk()
bg = PhotoImage(file="Vector_2646.png")


#my_label = Label(main_screen,image=bg)
#my_label.place(x=0,y=0,relwidth=1, relheight=1)

main_screen.title('Stock Shark')
main_screen.geometry('500x700')
#main_screen.configure(bg =bg)
#global functions
def open_url(url):
   webbrowser.open_new_tab(url)



# In[ ]:


def open_popup():
   error= tk.Tk()
   error.geometry("750x250")
   error.title("Error")
   Label(error, text= "Ticker Symbol Not Found !", font=('Mistral 18 bold')).place(x=200,y=80)
   Label(error, text= "Please ensure you have entered a ticker symbol exactly as displayed on the Yahoo Finace website" ,font=("Helvatica",10)).place (x=100,y=130)


   def Close():
      error.destroy()
   exit_button = Button(error,text="Okay",command = Close).place(x=350,y=160)


# In[ ]:


def open_results():
   results = tk.Tk()
   results.title(ticker + " Prediction Result")
   results.geometry("1000x500")
   results.protocol("WM_DELETE_WINDOW", results.destroy)
    
#   cal1 = Calendar(results, selectmode = 'day', year= 2023, month = 1,day = 2023 )
#   cal1.pack(pady=20)
    
    
   
#########################################################################################
   stockPriceData = yf.download(ticker, start='2020-01-01', end='2023-01-01')

   fig = plt.figure(figsize=(15, 8))
   plt.title(ticker + ' Stock Price History')
   plt.plot(stockPriceData['Close'])
   plt.xlabel('Date')
   plt.ylabel('Prices ($)')

   canvas = FigureCanvasTkAgg(fig, master=results)
   canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


##########################################################################################

   def Close_r():
      results.destroy()
      main_screen.deiconify()
   back_button = Button(results,text="Back", command=Close_r)
   back_button.pack()


# In[12]:


#Saving user input value as ticker
def save_text(event):
    global ticker
    ticker= txt_fld_1.get()
    print(ticker)
    with open("nasdaq_screener_1664449346258TEXT.txt") as f:
       text = f.readlines()
       text = [x.strip() for x in text]
       #print(text)
       #text = all of the ticker symbols as a list
       if ticker in text:
          print("EXIST")
          #main_screen.destroy()
          open_results()
       else:
          print("NO EXIST")
          open_popup()

def save_text_2():
    global ticker
    ticker= txt_fld_1.get()
    print(ticker)
    with open("nasdaq_screener_1664449346258TEXT.txt") as f:
       text = f.readlines()
       text = [x.strip() for x in text]
       #print(text)
       #text = all of the ticker symbols as a list
       if ticker in text:
          print("EXIST")
          #main_screen.destroy()
          open_results()
       else:
          print("NO EXIST")
          open_popup()


# In[13]:


#Functions to save the selected dates from the Calendars.
def on_date_selected_cal1(event):
    selected_date1 = event.widget.selection_get()
    print(selected_date1)
    start_date = selected_date1
def on_date_selected_cal2(event):
    selected_date2 = event.widget.selection_get()
    print(selected_date2)
    end_date = selected_date2
    


# In[ ]:


#Submit button





# txt_fld_1.bind('<Key>', capitalize_text)

# Label to instruct user to enter ticker symbol as displayed on Yahoo Finance
lbl_1= tk.Label(main_screen,text="Enter the Ticker symbol exactly as displayed on Yahoo Finance", fg='red', font=("Helvetica", 10))
lbl_1.grid(row=0, column=2, padx=5, pady=5 , sticky=tk.NSEW)


# Label linking to Yahoo Finance website
lbl_3=tk.Label(main_screen,text="https://finance.yahoo.com/",fg='blue',font=("Helvatica",10))
lbl_3.bind("<Button-1>", lambda e:open_url("https://finance.yahoo.com/"))
lbl_3.grid(row=1, column=2, padx=5, pady=5 , sticky=tk.NSEW)

# Text box for user to enter Ticker symbol
txt_fld_1 = tk.Entry(main_screen,bg='black',fg='white',bd=5)
txt_fld_1.grid(row=2, column=2, padx=5, pady=5 , sticky=tk.NSEW)


lbl_4= tk.Label(main_screen,text="In order to use this application to predict the future price of your desired Asset,\n You will need train the neural network with historical price action data.\n Please chose a Start and End date ", font=("Helvetica", 10))
lbl_4.grid(row=3, column=2, padx=5, pady=5 , sticky=tk.NSEW)

#Calendars to select dates to train Network on

lbl_5=tk.Label(main_screen,text="SELECT A START DATE",fg='blue',font=("Helvatica",10))
lbl_5.grid(row=4, column=2, padx=5, pady=5 , sticky=tk.NSEW)


cal1 = Calendar(main_screen, selectmode = 'day', year= 2023, month = 1,day = 2023 )
cal1.grid(row=5, column=2, padx=5, pady=5 , sticky=tk.NSEW)
cal1.bind("<<CalendarSelected>>",on_date_selected_cal1 )

lbl_6=tk.Label(main_screen,text="SELECT AN END DATE",fg='blue',font=("Helvatica",10))
lbl_6.grid(row=6, column=2, padx=5, pady=5 , sticky=tk.NSEW)

cal2 = Calendar(main_screen, selectmode = 'day', year= 2023, month = 1,day = 2023 )
cal2.grid(row=7, column=2, padx=5, pady=5 , sticky=tk.NSEW)
cal2.bind("<<CalendarSelected>>",on_date_selected_cal2 )


btn_1 = tk.Button(main_screen, text="Submit",fg='blue',command= save_text_2)
btn_1.grid(row=8, column=2, padx=5, pady=5, sticky=tk.NSEW)

####

main_screen.bind('<Return>',save_text)





main_screen.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




