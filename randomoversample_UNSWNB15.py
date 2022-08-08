#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 15:14:02 2021

@author: dina
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 13:11:11 2021

@author: dina
"""
import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from matplotlib import pyplot

input_path = "..../code/output"
output_path = ".../code/RANDOM_oversample"

X_train_file = 'train_norm.csv'
Y_train_file = 'y_train.csv'

# reading all the files for train test and target labels

train = pd.read_table(os.path.join(input_path, X_train_file), sep = ',',index_col = 0)
y_train = pd.read_table(os.path.join(input_path, Y_train_file),sep = ',', index_col = 0)


train = train.to_numpy()
y_train = y_train.to_numpy()
y_train = y_train.flatten()

# transform the dataset
oversample = RandomOverSampler(random_state = 0)
train, y_train = oversample.fit_resample(train, y_train)
# summarize distribution
counter = Counter(y_train)

train = pd.DataFrame(train)
y_train = pd.DataFrame(y_train)

# writing on the csv files
train.to_csv(os.path.join(output_path,'train_norm.csv'), sep=',')
y_train.to_csv(os.path.join(output_path,'y_train.csv'), sep=',')
