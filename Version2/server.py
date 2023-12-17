import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
from pymongo import MongoClient
import random
import pandas as pd 
from NaiveBayes import NaiveBayes  
import os 
import pickle

class GmailApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gmail App")
        self.root.geometry("800x700")
        self.model = pickle.load(open("spam.pkl","rb"))
        self.cv = pickle.load(open("vectorizer.pkl","rb"))
        self.ham_messages = []
        self.spam_messages = []

        self.message_index = 1

        self.create_gui()

        self.load_messages_from_database()
        
    def load_messages_from_database(self):
        # Kết nối đến MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['email']
        email_collection = db['new_emails']

        # Tải tin nhắn từ cơ sở dữ liệu
        messages = email_collection.find()
        for message in messages:
            sender = message.get('sender', 'Unknown Sender')
            subject = message.get('subject', 'No Subject')
            date = message.get('date', 'Unknown Date')
            body_text = message.get('body_text', 'No Body Text')

            # Phân loại tin nhắn là Ham hay Spam
            vect = self.cv.transform([body_text]).toarray()
            prediction = self.model.predict(vect)
            result = prediction[0]

            if result == 1:
                self.spam_messages.append((sender, subject, date, body_text))
                self.spam_listbox.insert(tk.END, f"{self.message_index}. {sender} - {subject}")
            else:
                self.ham_messages.append((sender, subject, date, body_text))
                self.ham_listbox.insert(tk.END, f"{self.message_index}. {sender} - {subject}")

            self.message_index += 1
    
    def create_gui(self):
        inbox_frame = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        inbox_frame.grid(row=0, column=0, sticky="nsew")

        # Ham Listbox
        ttk.Label(inbox_frame, text="Ham Inbox", font=("Helvetica", 16, "bold")).grid(row=0, column=0, sticky="w")
        self.ham_listbox = tk.Listbox(inbox_frame, selectmode=tk.SINGLE, width=60, height=10, font=("Helvetica", 12))
        self.ham_listbox.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.ham_listbox.bind('<<ListboxSelect>>', self.display_selected_message)

        # Spam Listbox
        ttk.Label(inbox_frame, text="Spam Inbox", font=("Helvetica", 16, "bold")).grid(row=0, column=1, sticky="w")
        self.spam_listbox = tk.Listbox(inbox_frame, selectmode=tk.SINGLE, width=60, height=10, font=("Helvetica", 12))
        self.spam_listbox.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.spam_listbox.bind('<<ListboxSelect>>', self.display_selected_message)

        delete_ham_button = ttk.Button(inbox_frame, text="Delete Ham", command=lambda: self.delete_selected_message('ham'))
        delete_ham_button.grid(row=2, column=0, pady=10, sticky="w")

        delete_spam_button = ttk.Button(inbox_frame, text="Delete Spam", command=lambda: self.delete_selected_message('spam'))
        delete_spam_button.grid(row=2, column=1, pady=10, sticky="w")

        # Email Frame
        email_frame = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        email_frame.grid(row=1, column=0, sticky="nsew")

        # Email Display
        email_display_label = ttk.Label(email_frame, text="Email", font=("Helvetica", 16, "bold"))
        email_display_label.grid(row=0, column=0, sticky="w")

        self.email_display_frame = ttk.Frame(email_frame)
        self.email_display_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
  
        # Grid configuration
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        inbox_frame.columnconfigure(0, weight=1)
        inbox_frame.columnconfigure(1, weight=1)
        inbox_frame.rowconfigure(1, weight=1)

        email_frame.columnconfigure(0, weight=1)
        email_frame.rowconfigure(1, weight=1)

    def display_email(self, message):
        self.email_display_text.config(state=tk.NORMAL)
        self.email_display_text.delete(1.0, tk.END)

        sender_text = f"Sender: {message[0]}\n"
        subject_text = f"Subject: {message[1]}\n"
        date_text = f"Date: {message[2]}\n"
        body_text = f"Body Text:\n{message[3]}\n"

        self.email_display_text.insert(tk.END, sender_text)
        self.email_display_text.insert(tk.END, subject_text)
        self.email_display_text.insert(tk.END, date_text)
        self.email_display_text.insert(tk.END, body_text)

        # Hiển thị avatar
