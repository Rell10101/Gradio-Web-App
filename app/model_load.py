from dotenv import load_dotenv
import os
import pandas as pd
import pickle


load_dotenv()

model_filename = os.getenv("MODEL_FILE_NAME")
with open(model_filename, "rb") as f:
        lin_reg = pickle.load(f)

train_data = pd.read_csv('../models/train_data.csv')
