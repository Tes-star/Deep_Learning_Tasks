import pandas as pd
pd.set_option('display.max_rows', None)
import numpy  as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Read in the csv data using pandas
sample = pd.read_csv('data/01_train/train.csv')
train  = pd.read_csv('data/01_train/sampleSubmission.csv',index_col=0)
test   = pd.read_csv('data/01_train/sampleTest.csv', index_col=0)
