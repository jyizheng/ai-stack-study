import mysql.connector
import pandas as pd
from datetime import datetime
import os

def dump_table_to_excel():
    # Database connection parameters
    config = {
        'host': 'localhost',  # or '127.0.0.1'
        'port': 3306,
        'user': 'root',        # replace with your MySQL username
        'password': 'my-secret-pw',  # replace with your MySQL password
        'database': 'mydatabase'
    }
    
    try:
        # Connect to MySQL database
        print("Connecting to MySQL database...")
        connection = mysql.connector.connect(**config)
        
        # Query to select all data from ape_data_record table
        query = "SELECT * FROM ape_data_record"
        
        # Read data into pandas DataFrame
        print("Fetching data from ape_data_record table...")
        df = pd.read_sql(query, connection)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ape_data_record_{timestamp}.xlsx"
        
        # Export to Excel
        print(f"Exporting data to {filename}...")
        df.to_excel(filename, index=False, engine='openpyxl')
        
        print(f"Successfully exported {len(df)} rows to {filename}")
        print(f"File saved in: {os.path.abspath(filename)}")
        
        # Display basic info about the exported data
        print(f"\nData summary:")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the connection
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            print("MySQL connection closed.")

if __name__ == "__main__":
    # Install required packages if not already installed
    # pip install mysql-connector-python pandas openpyxl
    
    dump_table_to_excel()

