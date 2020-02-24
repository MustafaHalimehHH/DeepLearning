import os
import cv2
import SimpleITK
import numpy
import math

'''
pixels spacing (2d)
    4cm:   0.5952380952380952
    3.5cm: 0.5208333333333334
    3cm:   0.44642857142857145
    
'''
CROP_CENTER_SHIFT = 220  #
BLACK_MARGIN = 100  # Exclude the text on the margin of the image and outside the ultrasound area
CROP_MARGIN = 20  #
PIXEL_SPACING = 0.446  # 0.6  # 0.446  # 0.522 # 0.594  # mm
SHIFT_DIFF_X = -120  # if the two scans are not in the same X axis coordinates
SHIFT_DIFF_Y = -16  # if the two scans are not in the same Y axis coordinates

# measure_path = 'D:\Halimeh\Datasets\LK-AI\LK1_lateral_measure.tif'
# clean_path = 'D:\Halimeh\Datasets\LK-AI\LK1_lateral.tif'
measure_path = 'D:\Halimeh\Datasets\LK-AI\LK1_axial_measure.tif'
clean_path = 'D:\Halimeh\Datasets\LK-AI\LK1_axial.tif'

# print(os.path.basename(measure_path))
measure_file_name = os.path.splitext(os.path.basename(measure_path))[0]
print('measure_file_name', measure_file_name)
'''
sitk = SimpleITK.ReadImage(measure_path)
print('sitk', sitk)
print('sitk_Origin', sitk.GetOrigin())
print('sitk_Spacing', sitk.GetSpacing())
print('sitk', sitk.GetSize())
arr = SimpleITK.GetArrayFromImage(sitk)
print('arr', arr.shape)
img = SimpleITK.GetImageFromArray(arr)
print('img', img.GetSize())
'''


image = cv2.imread(measure_path)
# print('image', type(image), image.shape)
# cv2.imshow('img', image[..., 2])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

temp = image[..., 1].copy()
# print('temp', temp.shape)
# temp[temp < 210] = 0
# cv2.imshow('temp', temp)
_, binary = cv2.threshold(temp, 205, 255, cv2.THRESH_BINARY)
# cv2.imshow('binary', temp)
# print('binary', temp.max(), temp.min(), temp.mean())
# binary = temp.copy()
binary[:BLACK_MARGIN, :] = 0
binary[:, :BLACK_MARGIN] = 0
binary[-BLACK_MARGIN:, :] = 0
binary[:, -BLACK_MARGIN:] = 0
# cv2.imshow('bbb', binary)
# cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
# print('cross', cross)
# binary = cv2.dilate(binary, cross, iterations=1)
# binary = cv2.erode(binary, cross, iterations=2)
# cv2.imshow('eee', binary)

element_cross = numpy.array([
    [0, 0, 255, 255, 255, 0, 0],
    [0, 0, 255, 255, 255, 0, 0],
    [255, 255, 255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255, 255, 255],
    [0, 0, 255, 255, 255, 0, 0],
    [0, 0, 255, 255, 255, 0, 0]], dtype=numpy.uint8)

cross_center = element_cross.shape[0] // 2
# print('cross_center', cross_center)

res = cv2.matchTemplate(binary, element_cross, cv2.TM_SQDIFF_NORMED)
# print('res', type(res), res.shape)
# cv2.imshow('binary', binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
min_threshold = (min_val + 1e-2)
locations = numpy.where(res <= min_threshold)
print('locations', len(locations), len(locations[0]), len(locations[1]))
# print('locations', locations)
# print('minMaxLoc', min_val, min_loc, max_val, max_loc)
if (len(locations[0]) != 4) and (len(locations[0]) != 2):
    raise RuntimeError('Measure Points (Crosses) must be either 4 or 2 due the slice (cut) [Axial, Sagittal]')

if len(locations[0]) == 4:
    print('---Lateral')
    for (x, y) in zip(locations[1], locations[0]):
        # print('x', x, 'y', y)
        # cv2.rectangle(image, (x-3, y-3), (x+22, y+22), [0, 255, 255], 2)
        cv2.circle(image, (x + cross_center, y + cross_center), 8, [0, 255, 255], 1)
    # cv2.imshow('locations', image)

    y_arr = numpy.array(locations[0]) + cross_center
    x_arr = numpy.array(locations[1]) + cross_center

    # print('y_arr', y_arr)
    # print('x_arr', x_arr)
    idx_min_y = numpy.argmin(y_arr)
    idx_max_y = numpy.argmax(y_arr)
    idx_min_x = numpy.argmin(x_arr)
    idx_max_x = numpy.argmax(x_arr)
    # print(idx_min_x, idx_max_x, idx_min_y, idx_max_y)
    cv2.line(image, (x_arr[idx_min_y], y_arr[idx_min_y]), (x_arr[idx_max_y], y_arr[idx_max_y]), [0, 0, 255], 1)
    cv2.line(image, (x_arr[idx_min_x], y_arr[idx_min_x]), (x_arr[idx_max_x], y_arr[idx_max_x]), [0, 0, 255], 1)
    # cv2.imshow('lines', image)

    diff_x_1 = x_arr[idx_max_x] - x_arr[idx_min_x]
    diff_y_1 = y_arr[idx_max_x] - y_arr[idx_min_x]
    diff_x_2 = x_arr[idx_max_y] - x_arr[idx_min_y]
    diff_y_2 = y_arr[idx_max_y] - y_arr[idx_min_y]
    slope = diff_x_1 * diff_y_2 - diff_y_1 * diff_x_2
    # print('slope', slope)
    t1 = ((x_arr[idx_min_y] - x_arr[idx_min_x]) * (diff_y_2) - (y_arr[idx_min_y] - y_arr[idx_min_x]) * (
        diff_x_2)) / slope
    # print('t1', t1)
    center_x = math.ceil(x_arr[idx_min_x] + (diff_x_1 * t1))
    center_y = math.ceil(y_arr[idx_min_x] + (diff_y_1 * t1))
    # print('c_x, c_y', c_x, c_y)
    cv2.circle(image, (center_x, center_y), 4, [255, 255, 0], 2)
    # cv2.circle(image, (center_x - CROP_CENTER_SHIFT, center_y - CROP_CENTER_SHIFT), 3, [255, 0, 255], 2)
    # cv2.circle(image, (center_x + CROP_CENTER_SHIFT, center_y + CROP_CENTER_SHIFT), 3, [255, 255, 0], 2)
    # cv2.rectangle(image, (center_x - CROP_CENTER_SHIFT, center_y - CROP_CENTER_SHIFT), (center_x + CROP_CENTER_SHIFT, center_y + CROP_CENTER_SHIFT), [255, 0, 0], 1)
    rect_crop_point_1 = (x_arr[idx_min_x] - CROP_MARGIN, y_arr[idx_min_y] - CROP_MARGIN)
    rect_crop_point_2 = (x_arr[idx_max_x] + CROP_MARGIN, y_arr[idx_max_y] + CROP_MARGIN)
    cv2.rectangle(image, rect_crop_point_1, rect_crop_point_2, [255, 0, 0], 1)

    '''
    square_half_width = max(
        abs(rect_crop_point_1[0] - center_x),
        abs(rect_crop_point_1[1] - center_y),
        abs(rect_crop_point_2[0] - center_x),
        abs(rect_crop_point_2[1] - center_y)
    ) # + CROP_MARGIN
    print(rect_crop_point_1)
    print(rect_crop_point_2)
    print(center_x, center_y)
    print('square_half_width', square_half_width)
    cv2.rectangle(image, (center_x - square_half_width, center_y - square_half_width), (center_x + square_half_width, center_y + square_half_width), [0, 0, 255], 2)
    '''

    # Lines Lengths
    line_1_length = numpy.sqrt((x_arr[2] - x_arr[1]) ** 2 + (y_arr[2] - y_arr[1]) ** 2)
    line_2_length = numpy.sqrt((x_arr[3] - x_arr[0]) ** 2 + (y_arr[3] - y_arr[0]) ** 2)
    text = 'Line 1 Length (pixel) {:4.4f}'.format(line_1_length)
    cv2.putText(image, text, (80, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255])
    text = 'Line 1 Length (cm) {:4.4f}'.format(line_1_length * PIXEL_SPACING / 100)
    cv2.putText(image, text, (80, 120), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255])
    text = 'Line 2 Length (pixel) {:4.4f}'.format(line_2_length)
    cv2.putText(image, text, (80, 140), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255])
    text = 'Line 2 Length (cm) {:4.4f}'.format(line_2_length * PIXEL_SPACING / 100)
    cv2.putText(image, text, (80, 160), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255])

    # print('-----', line_1_length * 0.446 / 100, line_2_length * 0.446 / 100)
    # print('Line Length (pixels)', ll)
    # ll_1 = ll * PIXEL_SPACING / 100
    # ll_1 = ll * 0.446 / 100

    text = 'Pixel Spacing {:4.4f} mm'.format(PIXEL_SPACING)
    cv2.putText(image, text, (80, 180), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255])
    text = 'Center (pixel {:3d}, {:3d})'.format(center_x, center_y)
    cv2.putText(image, text, (80, 200), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255])
    text = 'Margin Size (pixel {:d})'.format(CROP_MARGIN)
    cv2.putText(image, text, (80, 220), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255])

    cv2.imshow('Measure', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('locate_lines_center.tif', image)
    # print('center', c_x, c_y)
    # print('image', image.shape)
    # lymph_node_roi = image[center_y - CROP_CENTER_SHIFT: center_y + CROP_CENTER_SHIFT,
    #                   center_x - CROP_CENTER_SHIFT: center_x + CROP_CENTER_SHIFT, ...]
    lymph_node_roi = image[rect_crop_point_1[1] + SHIFT_DIFF_Y: rect_crop_point_2[1] + SHIFT_DIFF_Y,
                     rect_crop_point_1[0] + SHIFT_DIFF_X: rect_crop_point_2[0] + SHIFT_DIFF_X, ...]
    # print('lymph_node_roi', lymph_node_roi.shape)
    cv2.imshow('lymph node roi', lymph_node_roi)
    cv2.imwrite(measure_file_name + '.tif', image)

    source = cv2.imread(clean_path)
    print('source', source.shape)
    # lroi = source[center_y - CROP_CENTER_SHIFT: center_y + CROP_CENTER_SHIFT, center_x - CROP_CENTER_SHIFT: center_x + CROP_CENTER_SHIFT,
    #        ...]
    lroi = source[rect_crop_point_1[1] + SHIFT_DIFF_Y: rect_crop_point_2[1] + SHIFT_DIFF_Y,
           rect_crop_point_1[0] + SHIFT_DIFF_X: rect_crop_point_2[0] + SHIFT_DIFF_X, ...]

    print('lroi', lroi.shape)
    cv2.imwrite(measure_file_name + '[roi].tif', lroi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if len(locations[0]) == 2:
    print('---Axial')
    y_arr = numpy.array(locations[0]) + cross_center
    x_arr = numpy.array(locations[1]) + cross_center
    cv2.line(image, (x_arr[0], y_arr[0]), (x_arr[1], y_arr[1]), [0, 0, 255], 1)
    cv2.circle(image, (x_arr[0], y_arr[0]), 8, [0, 255, 255], 1)
    cv2.circle(image, (x_arr[1], y_arr[1]), 8, [0, 255, 255], 1)
    center_x = int(x_arr.mean())
    center_y = int(y_arr.mean())
    cv2.circle(image, (center_x, center_y), 4, [255, 255, 0], 2)
    # cv2.rectangle(image, (center_x - CROP_CENTER_SHIFT, center_y - CROP_CENTER_SHIFT), (center_x + CROP_CENTER_SHIFT, center_y + CROP_CENTER_SHIFT), [255, 0, 0], 1)
    square_half_width = abs(center_x - x_arr[0]) + CROP_MARGIN
    rect_crop_point_1 = (center_x - square_half_width, center_y - square_half_width)
    rect_crop_point_2 = (center_x + square_half_width, center_y + square_half_width)
    cv2.rectangle(image, rect_crop_point_1, rect_crop_point_2, [255, 0, 0], 1)
    # l_x, l_y = x_arr[1] - x_arr[0], y_arr[1] - y_arr[0]
    # print(l_x, l_y)
    # l = cv2.norm((285, 12), normType=cv2.NORM_L2)
    # print('l', type(l), l, l * PIXEL_SPACING / 100)
    ll = numpy.sqrt((x_arr[1] - x_arr[0]) ** 2 + (y_arr[1] - y_arr[0]) ** 2)
    # print('Line Length (pixels)', ll)
    # ll_1 = ll * PIXEL_SPACING / 100
    ll_1 = ll * PIXEL_SPACING / 100
    # print('Line Length (cm) ', ll_1)
    text = 'Line Length(pixels) {:4.4f}'.format(ll)
    cv2.putText(image, text, (80, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255])
    text = 'Line Length(cm) {:4.4f}'.format(ll_1)
    cv2.putText(image, text, (80, 120), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255])

    text = 'Pixel Spacing {:4.4f} mm'.format(PIXEL_SPACING)
    cv2.putText(image, text, (80, 140), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255])
    text = 'Center (pixel {:d}, {:d})'.format(center_x, center_y)
    cv2.putText(image, text, (80, 160), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255])
    text = 'Margin Size (pixel {:d})'.format(CROP_MARGIN)
    cv2.putText(image, text, (80, 180), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255])

    cv2.imshow('Measure', image)
    cv2.imwrite(measure_file_name + '.tif', image)

    # lymph_node_roi = image[center_y - CROP_CENTER_SHIFT: center_y + CROP_CENTER_SHIFT,
    #                  center_x - CROP_CENTER_SHIFT: center_x + CROP_CENTER_SHIFT, ...]
    lymph_node_roi = image[rect_crop_point_1[1] + SHIFT_DIFF_Y: rect_crop_point_2[1] + SHIFT_DIFF_Y,
                     rect_crop_point_1[0] + SHIFT_DIFF_X: rect_crop_point_2[0] + SHIFT_DIFF_X, ...]
    # print('lymph_node_roi', lymph_node_roi.shape)
    cv2.imshow('lymph node roi', lymph_node_roi)
    # cv2.imwrite(measure_file_name + '[roi].tif', lymph_node_roi)

    source = cv2.imread(clean_path)
    print('source', source.shape)
    # lroi = source[center_y - CROP_CENTER_SHIFT: center_y + CROP_CENTER_SHIFT,
    #       center_x - CROP_CENTER_SHIFT: center_x + CROP_CENTER_SHIFT, ...]
    lroi = source[rect_crop_point_1[1] + SHIFT_DIFF_Y: rect_crop_point_2[1] + SHIFT_DIFF_Y,
           rect_crop_point_1[0] + SHIFT_DIFF_X: rect_crop_point_2[0] + SHIFT_DIFF_X, ...]

    cv2.imwrite(measure_file_name + '[roi].tif', lroi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
left_crop = min(locations[0]) - CROP_MARGIN
right_crop = max(locations[0]) + CROP_MARGIN
top_crop = min(locations[1]) - CROP_MARGIN
bottom_crop = max(locations[1]) + CROP_MARGIN
crop_image = image[left_crop: right_crop, top_crop: bottom_crop, ...]
print('crop_image', crop_image.shape)
cv2.imshow('crop_image', crop_image)
'''
