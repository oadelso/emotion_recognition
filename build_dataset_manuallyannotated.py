# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:13:03 2018

@author: Christopher
"""

import pandas as pd
import os
import glob
import random

from PIL import Image
from tqdm import tqdm

# Parameters
SIZE = 128

def loadDict():
    """Load dictionaries -- we combine both to create one big dictionary"""
    data1 = pd.read_csv('Manually_Annotated_file_lists\\training.csv', index_col = 0)
    data2 = pd.read_csv('Manually_Annotated_file_lists\\validation.csv', index_col = 0)
    
    frames = [data1, data2]
    data = pd.concat(frames)
    
    return data

def getExpression(df, idx):
    """
    Input: panda df and index
    Returns: expression for idx in df
    """
    query = df.loc[idx]
    expression = query['expression']
    
    return expression

def getFilenames():
    """
    Get filenames
    """
    path = os.getcwd()+'\\Manually_Annotated_Images\\'

    filenames = glob.glob(os.path.join(path,"**/*.jpg"), recursive = True)
    
    return filenames

def resize_and_save(filename, file_dict, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    # Get label and save file
    idx = filename.split('\\')[-2]+'/'+filename.split('\\')[-1]
    expression = getExpression(file_dict, idx)
    image.save(os.path.join(output_dir, str(expression)+"_"+filename.split('\\')[-1]))

if __name__ == '__main__':
    
    # Load dictionary
    file_dict = loadDict()
    
    # Obtain Filenames
    filenames = getFilenames()
    
    # Sort filenames
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.9 * len(filenames))
    train_filenames = filenames[:split]
    val_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'val': val_filenames}
    
    # Iterate filenames
    num_errors = 0
    print('***SAVING FILES***')
    for split in ['train', 'val']:
        i = 0
        for filename in tqdm(filenames[split]):
            try:
                resize_and_save(filename, file_dict, "D:\\AffectNet_Database\\"+split+'_milestone')
                i += 1
                if(i == 100000 and split == 'train'):
                    break
                elif(i == 10000 and split == 'val'):
                    break
            except:
                # If label is not found or image is corrupted
                num_errors += 1                
    print('***FILES SAVED!***')
    print("Found", num_errors,"files with problems.")
    
    
    # e = getExpression(file_list, '459/81456263be241927c7a59a2646f88c2700ce4b7cba6094570ec2b10c.jpg')
    
    