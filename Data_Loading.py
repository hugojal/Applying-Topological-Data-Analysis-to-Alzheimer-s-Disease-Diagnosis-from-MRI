(If you are running this on Google Collab):


! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download jboysen/mri-and-alzheimers

# Put on the same directory
from zipfile import ZipFile


# specifying the name of the zip file
file = "/content/archive (6).zip"


# open the zip file in read mode
with ZipFile(file, 'r') as zip:
    # list all the contents of the zip file
    zip.printdir()


    # extract all files
    print('extraction...')
    zip.extractall()
    print('Done!')

import pandas as pd

df = pd.read_csv('/content/oasis_longitudinal.csv')
