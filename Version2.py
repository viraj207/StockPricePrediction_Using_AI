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
