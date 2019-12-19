# https://www.kaggle.com/gpreda/visualize-ct-dicom-data
from __future__ import print_function
import os
from pathlib import Path
import numpy
import pandas
import sklearn
import keras
import skimage
import pydicom
import glob
import seaborn
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

BASE_DATA_DIR = 'D:\Halimeh\Datasets\Kaggle\siim-medical-images'
print(os.listdir(BASE_DATA_DIR))

df = pandas.read_csv(os.path.join(BASE_DATA_DIR, 'overview.csv'))
print('CT Medical images: Rows({}) - Cols({})'.format(df.shape[0], df.shape[1]))
print('Tiff images', len(os.listdir(os.path.join(BASE_DATA_DIR, 'tiff_images'))))
print('DICOM files', len(os.listdir(os.path.join(BASE_DATA_DIR, 'dicom_dir'))))
# tiff_df = pandas.DataFrame([{'path' : file_path} for file_path in glob.glob(os.path.join(BASE_DATA_DIR, 'tiff_images/*.tif'))])


def process_data(path):
    df = pandas.DataFrame([{'Path' : file_path} for file_path in glob.glob(os.path.join(BASE_DATA_DIR, path))])
    df['File'] = df['Path'].map(os.path.basename)
    df['ID'] = df['File'].map(lambda x: str(x.split('_')[1]))
    df['Age'] = df['File'].map(lambda x: int(x.split('_')[3]))
    df['Contrast'] = df['File'].map(lambda x: bool(int(x.split('_')[5])))
    df['Modality'] = df['File'].map(lambda x: str(x.split('_')[6].split('.')[-2]))
    return df


def countplot_comparision(feature):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    s1 = seaborn.countplot(df[feature], ax=ax1)
    s1.set_title('Overview data')
    s2 = seaborn.countplot(tiff_df[feature], ax=ax2)
    s2.set_title('Tiff files data')
    s3 = seaborn.countplot(dicom_df[feature], ax=ax3)
    s3.set_title('DICOM files data')
    plt.show()


def show_images(data, dim=16, imtype='TIFF'):
    img_data = list(data[:dim].T.to_dict().values())
    f, ax = plt.subplots(4, 4, figsize=(16, 20))
    for i, d in enumerate(img_data):
        if imtype == 'TIFF':
            img = skimage.io.imread(d['Path'])
            ax[i//4, i%4].matshow(img, cmap='gray')
        elif imtype == 'DICOM':
            img = pydicom.read_file(d['Path'])
            ax[i//4, i%4].imshow(img.pixel_array, cmap=plt.cm.bone)
        ax[i//4, i%4].axis('off')
        ax[i//4, i%4].set_title('Modality:{Modality} Age:{Age}\nSlice:{ID} Contrast:{Contrast}'.format(**d))
    plt.show()




tiff_df = process_data('tiff_images/*.tif')
dicom_df = process_data('dicom_dir/*.dcm')
# print(tiff_df.head(5))
# countplot_comparision('Contrast')
# countplot_comparision('Age')

'''
print(type(tiff_df[:16]))
print(tiff_df[:4].to_dict())
print(tiff_df[:4].to_dict().values())
print(list(tiff_df[:4].to_dict().values()))
print(list(tiff_df[:4].T.to_dict().values()))
'''

# show_images(tiff_df, 16, 'TIFF')
# show_images(dicom_df, 16, 'DICOM')

dicom_file = pydicom.read_file(list(dicom_df[:1].T.to_dict().values())[0]['Path'])
print(dicom_file)
print(type(dicom_file))
