import sqlite3 as sql
import os 
import glob
import pandas as pd 
import numpy as np 


con = sql.connect("revanstack.db")

file_pattern = r"dataset/*.csv"
file_list = glob.glob(file_pattern)

for file_path in file_list:
    base_name = os.path.basename(file_path)
    table_name = os.path.splitext(base_name)[0]

    print(f"Loading {base_name} into table '{table_name}'...")


    try :
        df = pd.read_csv(file_path)

        df.to_sql(table_name, con, if_exists="replace", index=False)
        print(f"Successfully loaded {table_name}!")

    except Exception as e:
        print(f"Failed to load {base_name}. Error: {e}")

print("\nAll files processed successfully.")



# Query the sqlite_master table to get all user-created tables
query = "SELECT name FROM sqlite_master WHERE type='table';"

#  Read the results into a Pandas DataFrame
tables_df = pd.read_sql_query(query, con)

#  Display the list of tables
print("--- Tables currently in the database ---")
print(tables_df)

#  Optional: Verify the row count of a specific table to ensure it has data
for table in tables_df['name']:
    count_df = pd.read_sql_query(f"SELECT COUNT(*) as total_rows FROM {table}", con)
    print(f"Table '{table}' has {count_df['total_rows'][0]} rows.")

con.close()