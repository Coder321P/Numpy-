# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import tkinter as tk
# from tkinter import messagebox, filedialog

# def run_script():
#     file_path = filedialog.askopenfilename(
#         title="Select CSV File",
#         filetypes=[("CSV files", "*.csv")]
#     )
    
#     try:
#         data = pd.read_csv(file_path)
#         print("Data loaded successfully.")
#         messagebox.showinfo("Success", "Data loaded successfully.")
#     except Exception as e:
#         print("Failed to load file:")
#         messagebox.showerror("Failed to load file:")

# root = tk.Tk()
# root.title("Run Script")
# root.geometry("300x100")

# btn = tk.Button(root, text="Run Data Script", command=run_script)
# btn.pack(pady=20)

# root.mainloop()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk

try:
    data = pd.read_csv('/Users/prasitgupta/Documents/GitHub/Numpy-/Real_Estate_Sales_2001-2022_GL.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("File not found. Please check the file path.")

    exit()

# Display the first 5 rows of the dataset
print(data.head(5))

