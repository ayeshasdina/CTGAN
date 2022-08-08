

"""
Created on Sun Sep 26 14:00:57 2021

@author: dina

--> make all the attack type balanced
--> remove duplicate from the synthetic samples created by CTGAN
--> have to maintain same training data distribution
"""
from tabgan.sampler import OriginalGenerator, GANGenerator
import pandas as pd
import numpy as np
import os

threshold_normal = 10
threshold_dos = 21073
threshold_probe = 53344
threshold_r2l = 64005
threshold_u2r = 64948

num_epochs = 80

input_path = "/home/adi252/CTGAN_GPUV6/input"
output_path = "/home/adi252/CTGAN_GPUV6/output"

X_train_file = 'trainData_onehot.csv'
X_test_file = 'testData_onehot.csv'
X_test_21_file = 'testData21_onehot.csv'

Y_train_file = 'y_train.csv'
Y_test_file = 'y_test.csv'
Y_test_21_file = 'Y_test_21.csv'

# reading all the files for train test and target

train = pd.read_table(os.path.join(input_path, X_train_file), sep = ',',index_col = 0)
test = pd.read_table(os.path.join(input_path, X_test_file), sep = ',',index_col = 0)
test_21 = pd.read_table(os.path.join(input_path, X_test_21_file), sep = ',',index_col=0)


y_train = pd.read_table(os.path.join(input_path, Y_train_file),sep = ',', index_col = 0)
# y_train =y_train.loc[:, ~y_train.columns.str.contains('^Unnamed')]

y_test = pd.read_table(os.path.join(input_path, Y_test_file),sep = ',', index_col = 0)
# y_test = y_test.loc[:, ~y_test.columns.str.contains('^Unnamed')]

y_test_21 = pd.read_table(os.path.join(input_path, Y_test_21_file),sep = ',', index_col = 0)
# y_test_21 = y_test_21.loc[:, ~y_test_21.columns.str.contains('^Unnamed')]



target = y_train

normal = 0
probe = 0
R2L = 0
U2R = 0
DoS = 0

i = 0


while normal < threshold_normal or probe < threshold_probe or DoS < threshold_dos or R2L < threshold_r2l or U2R < threshold_u2r:
    
        
        
            
       
        i = i+1
      
        
        new_train3, new_target3 = GANGenerator(gen_x_times= 10, cat_cols=None, bot_filter_quantile=0.001, ### 
                                        top_filter_quantile=0.999,
                                        is_post_process=True,
                                        pregeneration_frac=2, only_generated_data=False,
                                        epochs= num_epochs).generate_data_pipe(train, target,
                                                                      test,
                                                                      only_adversarial= True,
                                                                      use_adversarial= False,only_generated_data = True)

       
        
       
        new_target3 = pd.DataFrame(new_target3)
       
        
        
        # new_target3.to_csv(os.path.join(output_path,'synthetic_target_'+ str(i) +'.csv'), sep=',')
        
        
                
        print("count of normal, probe, r2l, u2r, and Dos in each round.")

### have to change order of the name according to attack type number, 1,2,3,4
        normal = normal + new_target3[new_target3['attack_type'] == 3].count()[0]
        probe = probe + new_target3[new_target3['attack_type'] == 4].count()[0]
        R2L = R2L + new_target3[new_target3['attack_type'] == 0].count()[0]
        U2R = U2R + new_target3[new_target3['attack_type'] == 1].count()[0]
        DoS = DoS + new_target3[new_target3['attack_type'] == 2].count()[0]
        ### this will be removed if we do the other experiments , means add all the synthetic data including the giant attcak types. to make the dataset properly balamced, need this.
        if normal > threshold_normal:
            index_drop = new_target3[new_target3['attack_type'] == 3].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop)
            
        if DoS > threshold_dos:
            index_drop = new_target3[new_target3['attack_type'] == 2].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop)
        if probe > threshold_probe:
            index_drop = new_target3[new_target3['attack_type'] == 4].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop)   
        
        if U2R > threshold_u2r:
            index_drop = new_target3[new_target3['attack_type'] == 1].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop) 
            
        if R2L > threshold_r2l:
            index_drop = new_target3[new_target3['attack_type'] == 0].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop)  
         #### merge both train target to keep track on them.

         
        print("Saved the new train and target file")
        new_train3['attack_type'] = new_target3['attack_type']
        new_train3.to_csv(os.path.join(output_path,'synthetic_train_target_V6_'+ str(i) +'.csv'), sep=',')
        
        
        count = [normal,DoS,probe,U2R,R2L]
        print(count)
        print(normal)
        print(probe)
        print(R2L)
        print(U2R)
        print(DoS)
                                                                      
    

                                                                                  
                                                                            