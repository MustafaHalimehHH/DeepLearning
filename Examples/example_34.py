import os
import cv2
import numpy
import math


def find_two_crossing_lines_by_four_points(points_x, points_y):
    max_line_length = 0.0
    min_line_length = 1e+5

    p1_idx, p2_idx = None, None  # first line as it has the max length
    p3_idx, p4_idx = None, None  # second line as it has the min length
    for i in range(len(points_x)):
        for j in range(i + 1, len(points_x)):
            print('i, j', i, j)
            line_length = math.sqrt(math.pow(points_x[i] - points_x[j], 2) + math.pow(points_y[i] - points_y[j], 2))
            print('line_length', line_length)
            if line_length > max_line_length:
                max_line_length = line_length
                p1_idx = i
                p2_idx = j
            if line_length < min_line_length:
                min_line_length = line_length
                p3_idx = i
                p4_idx = j

    return (points_x[p1_idx], points_y[p1_idx]), (points_x[p2_idx], points_y[p2_idx]), (
    points_x[p3_idx], points_y[p3_idx]), (points_x[p4_idx], points_y[p4_idx])

'''
p1, p2, p3, p4 = find_two_crossing_lines_by_four_points((0, 1, 10, 22), (33, 11, -1, 10))
print('p1', p1)
print('p2', p2)
print('p3', p3)
print('p4', p4)
exit(1)
'''

measure_path = 'D:\Halimeh\Datasets\LK-AI\ser001img00010.tif'
clean_path = 'D:\Halimeh\Datasets\LK-AI\LK1_lateral.tif'
# measure_path = 'D:\Halimeh\Datasets\LK-AI\LK1_axial_measure.tif'
# clean_path = 'D:\Halimeh\Datasets\LK-AI\LK1_axial.tif'

image = cv2.imread(measure_path)
print('image_shape', image.shape)
image = image[:, 420: 440, 1]
cv2.imshow('image', image)
image = image[180:, ...]
cv2.imshow('imagebbbb', image)
print('imagebbbb', image.shape)

# image = image[:, image.shape[1] // 2 - 16: image.shape[1] // 2 + 16, ...]
_, binary = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
cv2.imshow('binary', binary)
binary = binary[:, 1, ...]
cv2.imshow('binary', binary)
idxs = numpy.argwhere(binary == 255)
print('idxs', idxs[0], idxs[-1], idxs[-1] - idxs[0])
print('binary_shape', binary.shape)
print('binary_sum', binary.sum() / 255, numpy.count_nonzero(binary))
# print('scale', 300 / 672)
print('scale', 650 / numpy.count_nonzero(binary))
print('scale', 650 / (numpy.count_nonzero(binary) + 1))
print('scale', 400 / 780)

cv2.waitKey(0)
cv2.destroyAllWindows()
