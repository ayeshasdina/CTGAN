

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

# threshold_normal = 10
# threshold_dos = 5 #21073
# threshold_probe = 53344
# threshold_r2l = 64005
# threshold_u2r = 64948

threshold_normal = 1
threshold_analysis = 30000
threshold_backdoor = 10000
threshold_exploit = 20000
threshold_dos = 30000
threshold_fuzzers = 25000
threshold_generic = 10000
threshold_reconnaissance  = 30000
threshold_shellcode = 30000
threshold_worms = 30000

num_epochs = 70

# input_path = "/Users/dina/Documents/Cyber_security/classification/UNSW-NB15/code/output"
# output_path = "output_CTGAN/"

input_path = "/home/adi252/CTGAN-UNV1/input"
output_path = "/home/adi252/CTGAN-UNV1/output"

X_train_file = 'trainData_onehot.csv'
X_test_file = 'testData_onehot.csv'


Y_train_file = 'y_train.csv'
Y_test_file = 'y_test.csv'


# reading all the files for train test and target

train = pd.read_table(os.path.join(input_path, X_train_file), sep = ',',index_col = 0)
test = pd.read_table(os.path.join(input_path, X_test_file), sep = ',',index_col = 0)



y_train = pd.read_table(os.path.join(input_path, Y_train_file),sep = ',', index_col = 0)
# y_train =y_train.loc[:, ~y_train.columns.str.contains('^Unnamed')]

y_test = pd.read_table(os.path.join(input_path, Y_test_file),sep = ',', index_col = 0)
# y_test = y_test.loc[:, ~y_test.columns.str.contains('^Unnamed')]



### cut off most of the data for the dummy test in the local computer.
# train = train[0:1000]
# y_train = y_train[0:1000]
# test = test[0:1000]

target = y_train

normal = 0
analysis = 0
backdoor = 0
exploit = 0
dos = 0
fuzzers = 0
generic = 0
reconnaissance  = 0
shellcode = 0
worms = 0


i = 0



while normal < threshold_normal or analysis < threshold_analysis or dos < threshold_dos or backdoor < threshold_backdoor or exploit < threshold_exploit or fuzzers < threshold_fuzzers or generic < threshold_generic or reconnaissance < threshold_reconnaissance or shellcode < threshold_shellcode or worms < threshold_worms:
    
# while (count <= 10):  
    
        # normal = threshold
        # probe = threshold
        # R2L = threshold
        # U2R = threshold
        # DoS = threshold
        # print("DONE IT")
        
        
            
       
        i = i+1
        
        
        new_train3, new_target3 = GANGenerator(gen_x_times= 4, cat_cols=None, bot_filter_quantile=0.001,
                                        top_filter_quantile=0.999,
                                        is_post_process= True, ## True
                                        pregeneration_frac=2, only_generated_data= False,
                                        epochs= num_epochs).generate_data_pipe(train, target,
                                                                      test,
                                                                      only_adversarial= True,
                                                                      use_adversarial= False,only_generated_data = False)
                                                                               
        
    
        
       
        
       
        new_target3 = pd.DataFrame(new_target3)
       
        
        
        # new_target3.to_csv(os.path.join(output_path,'synthetic_target_'+ str(i) +'.csv'), sep=',')
        
        
                

        
        
        # print(new_target3)
        analysis = analysis+ new_target3[new_target3['attack_cat'] == 0].count()[0]
        backdoor = backdoor + new_target3[new_target3['attack_cat'] == 1].count()[0]
        dos      = dos + new_target3[new_target3['attack_cat'] == 2].count()[0]
        exploit = exploit + new_target3[new_target3['attack_cat'] == 3].count()[0]
        fuzzers = fuzzers + new_target3[new_target3['attack_cat'] == 4].count()[0]
        generic = generic + new_target3[new_target3['attack_cat'] == 5].count()[0]
        normal = normal + new_target3[new_target3['attack_cat'] == 6].count()[0]
        reconnaissance = reconnaissance + new_target3[new_target3['attack_cat'] == 7].count()[0]
        shellcode = shellcode + new_target3[new_target3['attack_cat'] == 8].count()[0]
        worms = worms + new_target3[new_target3['attack_cat'] == 9].count()[0]
        
        print("count of normal, probe, r2l, u2r, and Dos in each round.")

### have to change order of the name according to attack type number, 1,2,3,4
        count = [normal,analysis,backdoor,dos,exploit,fuzzers,generic,reconnaissance,shellcode,worms]
        print(count)
        ### this will be removed if we do the other experiments , means add all the synthetic data including the giant attcak types. to make the dataset properly balamced, need this. 
        if normal > threshold_normal:
            index_drop = new_target3[new_target3['attack_cat'] == 6].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop)
            
        if dos > threshold_dos:
            index_drop = new_target3[new_target3['attack_cat'] == 2].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop)
        if analysis > threshold_analysis:
            index_drop = new_target3[new_target3['attack_cat'] == 0].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop)   
        
        if backdoor > threshold_backdoor:
            index_drop = new_target3[new_target3['attack_cat'] == 1].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop) 
            
        if exploit > threshold_exploit:
            index_drop = new_target3[new_target3['attack_cat'] == 3].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop)  
            
        if fuzzers > threshold_fuzzers:
            index_drop = new_target3[new_target3['attack_cat'] == 4].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop)   
                        
        if generic > threshold_generic:
            index_drop = new_target3[new_target3['attack_cat'] == 5].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop)  
                        
        if reconnaissance > threshold_reconnaissance :
            index_drop = new_target3[new_target3['attack_cat'] == 7].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop)  
                                    
        if shellcode > threshold_shellcode :
            index_drop = new_target3[new_target3['attack_cat'] == 8].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop)  
                                    
        if worms > threshold_worms :
            index_drop = new_target3[new_target3['attack_cat'] == 9].index
            new_train3 = new_train3.drop(index_drop)
            new_target3 = new_target3.drop(index_drop)  
            
        
            
        #### merge both train target to keep track on them.

         
        print("Saved the new train and target file")
        new_train3['attack_type'] = new_target3['attack_cat']
        new_train3.to_csv(os.path.join(output_path,'synthetic_train_target'+ str(i) +'.csv'), sep=',')
        
        count = [normal,analysis,backdoor,dos,exploit,fuzzers,generic,reconnaissance,shellcode,worms]
        
        
        print(count)

                                                                      
    

                                                                                  
                                                                            