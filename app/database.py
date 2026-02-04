#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 2023

@author: lefteris

@subject: database
"""

# Module 2: Database
import sqlite3

class Database:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name, check_same_thread=False,  timeout=10)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        # Create necessary tables if not exist
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS user_data (
                                id INTEGER PRIMARY KEY,
                                name TEXT,
                                phone_number TEXT,
                                message TEXT,
                                coordinates_lat REAL,
                                coordinates_long REAL,
                                severity INTEGER DEFAULT 2,
                                dt DATETIME
                                )''')

        # Add severity column to existing tables if it doesn't exist
        try:
            self.cursor.execute("ALTER TABLE user_data ADD COLUMN severity INTEGER DEFAULT 2")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS op_pos (
                                id INTEGER PRIMARY KEY,
                                name TEXT,
                                base_type TEXT,
                                coordinates_lat REAL,
                                coordinates_long REAL,
                                capacity INTEGER,
                                dt DATETIME
                                )''')
        self.conn.commit()

    def insert_user_data(self, name, phone_number, message, coordinates_lat, coordinates_long, dt, severity=2):
        # Insert user data into database
        # severity: 1=Critical, 2=Urgent, 3=Normal
        self.cursor.execute('''INSERT INTO user_data (name, phone_number,
                                message, coordinates_lat, coordinates_long, severity, dt)
                               VALUES (?, ?, ?, ?, ?, ?, ?)''',
                               (name, phone_number, message, coordinates_lat, coordinates_long, severity, dt))
        self.conn.commit()
    
    def insert_operation_base(self, name, base_type, coordinates_lat, coordinates_long, capacity):
        # Insert user data into database
        self.cursor.execute('''INSERT INTO op_pos (name, base_type, coordinates_lat, coordinates_long, capacity) 
                               VALUES (?, ?, ?, ?, ?)''',
                               (name, base_type, coordinates_lat, coordinates_long, capacity))
        self.conn.commit()

    def fetch_user_data(self):
        # Fetch user data from database
        self.cursor.execute("SELECT * FROM user_data")
        return self.cursor.fetchall()
    
    def fetch_operation_base(self):
        # Fetch user data from database
        self.cursor.execute("SELECT * FROM op_pos")
        return self.cursor.fetchall()
    
    def delete_operation_base(self, id_):
        # Fetch user data from database
        self.cursor.execute("DELETE FROM op_pos WHERE id = ?;",(id_,))
        self.conn.commit()
        
    def delete_user_data(self, id_):
        # Fetch user data from database
        self.cursor.execute("DELETE FROM user_data WHERE id = ?;",(id_,))
        self.conn.commit()



























