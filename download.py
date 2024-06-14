import gdown
import os
from zipfile import ZipFile

file_url = 'https://drive.google.com/drive/folders/12bteP1gw1JwHoz3qdRQwZrBmcm-_U9_5?usp=drive_link'

output_path = 'algonauts_2023_challenge_data.zip'

gdown.download(file_url,output_path,quiet=False)

with ZipFile(output_path,'r') as zip_ref:
    zip_ref.extractall('data')
    
os.remove(output_path)