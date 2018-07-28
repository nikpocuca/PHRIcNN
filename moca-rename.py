import pandas as pd
import os 
from shutil import copyfile
import sys

moca_data = pd.read_csv("full-scores-moca.csv")

def get_tuple(data):
    lexicon_id = {}
    for i in range(data.shape[0]):
        id = data.iloc[i][2]
        visit = data.iloc[i][3]
        lexicon_id[(id,visit)] = data.iloc[i][4]
    return(lexicon_id)

bio_data = get_tuple(moca_data)

input_dir = str(sys.argv[1])
raw_names = os.listdir(input_dir)

def get_raw_tuple(names):
    lexicon_image = {}
    for i in range(len(names)):
        name = names[i]
        split_names = str.split(name,"_")
        if split_names[0] == "Post":
            id = int(split_names[1])
            visit = int(split_names[2])
            lexicon_image[(id,visit)] = name
        else:
            id = int(split_names[0])
            visit = int(split_names[1])
            lexicon_image[(id,visit)] = name
    return lexicon_image

image_names = get_raw_tuple(raw_names)

for i_key in image_names.keys():
    if i_key in bio_data.keys():
        image_name = image_names[i_key]
        score = str(bio_data[i_key])
        new_name = "m" + score + "_" + image_name
        copyfile("raw-data/" + image_name,"new-names/" + new_name)
