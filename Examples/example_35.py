import cv2
import numpy
import SimpleITK


file_path = 'D:\Halimeh\Datasets\LK-AI\LK1_lateral.tif'
file_reader = SimpleITK.ImageFileReader()
images_io_tuple = file_reader.GetRegisteredImageIOs()
file_reader.SetFileName(file_path)
print('Images IO ', images_io_tuple)
# print('ImageFileReader', file_reader)
file_reader.ReadImageInformation()
print('Size', file_reader.GetSize())
print('Origin', file_reader.GetOrigin())
metadata_keys = file_reader.GetMetaDataKeys()
print('metadata_keys', metadata_keys)
for k in metadata_keys:
    print('Meta Data', k, file_reader.GetMetaData(k))


exit(1)



image = cv2.imread(file_path, cv2.IMREAD_COLOR)
print('image', type(image), image.shape)
cv2.imshow('image', image)

img1 = cv2.GaussianBlur(image, (3, 3), 0)
cv2.imshow('img1', img1)

norm = cv2.normalize(img1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
cv2.imshow('norm', norm)
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
print('gray', gray.shape)
cv2.imshow('gray', gray)
lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=7)
cv2.imshow('lap', lap)

cv2.waitKey(0)
cv2.destroyAllWindows()
