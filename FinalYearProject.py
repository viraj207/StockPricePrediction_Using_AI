#!/usr/bin/env python
# coding: utf-8

# In[86]:


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


# In[87]:


#Creating the main page
main_screen = tk.Tk()
bg = PhotoImage(file="Vector_2646.png")


#my_label = Label(main_screen,image=bg)
#my_label.place(x=0,y=0,relwidth=1, relheight=1)

main_screen.title('Stock Shark')
main_screen.geometry('1000x500')
#main_screen.configure(bg =bg)
#global functions
def open_url(url):
   webbrowser.open_new_tab(url)



# In[88]:


def open_popup():
   error= tk.Tk()
   error.geometry("750x250")
   error.title("Error")
   Label(error, text= "Ticker Symbol Not Found !", font=('Mistral 18 bold')).place(x=200,y=80)
   Label(error, text= "Please ensure you have entered a ticker symbol exactly as displayed on the Yahoo Finace website" ,font=("Helvatica",10)).place (x=100,y=130)


   def Close():
      error.destroy()
   exit_button = Button(error,text="Okay",command = Close).place(x=350,y=160)


# In[89]:


def open_results():
   results = tk.Tk()
   results.title(ticker + " Prediction Result")
   results.geometry("1000x500")
   results.protocol("WM_DELETE_WINDOW", results.destroy)

   stockPriceData = yf.download(ticker, start='2020-01-01', end='2023-01-01')

   fig = plt.figure(figsize=(15, 8))
   plt.title(ticker + ' Stock Price History')
   plt.plot(stockPriceData['Close'])
   plt.xlabel('Date')
   plt.ylabel('Prices ($)')

   canvas = FigureCanvasTkAgg(fig, master=results)
   canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)



   def Close_r():
      results.destroy()
      main_screen.deiconify()
   back_button = Button(results,text="Back", command=Close_r)
   back_button.pack()


# In[90]:


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


# In[ ]:


btn_1 = Button(main_screen, text="Submit",fg='blue',command= save_text_2)
btn_1.place(x=500, y=250)
#Text box for user to enter Ticker symbol

txt_fld_1 = Entry(main_screen,bg='black',fg='white',bd=5)
txt_fld_1.place(x=460,y=215)
#txt_fld_1.bind('<Key>', capitalize_text)


with open("nasdaq_screener_1664449346258TEXT.txt") as f:
    text_list = f.readlines()
    text_list = [x.strip() for x in text_list]
             
lbl_1= Label(main_screen,text="Enter the Ticker symbol exactly as displayed on Yahoo Finance", fg='red', font=("Helvetica", 10))
lbl_1.place(x=350,y=170)

lbl_3=Label(main_screen,text="https://finance.yahoo.com/",fg='blue',font=("Helvatica",10))
lbl_3.bind("<Button-1>", lambda e:open_url("https://finance.yahoo.com/"))
lbl_3.place(x=450,y=190)


####

main_screen.bind('<Return>',save_text)





main_screen.mainloop()


# In[ ]:





# In[ ]:




