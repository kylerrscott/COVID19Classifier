import csv
import numpy
import cv2


def read_resized_image(filename, new_width):
    
    #read in image
    original_image = cv2.imread(filename)
    
    #determine smallest side
    height, width, channels = original_image.shape
    smallest_side = min(height, width)
    
    #crop to that size
    top_edge = int((height - smallest_side) / 2)
    bottom_edge = int(height - (height - smallest_side) / 2)
    left_edge = int((width - smallest_side) / 2)
    right_edge = int(width - (width - smallest_side) / 2)
    cropped_image = original_image[top_edge:bottom_edge, left_edge:right_edge]
    
    #resize to new_width
    resized_image = cv2.resize(cropped_image, (new_width, new_width))
    return resized_image


#example of usage
first_line = 0
new_width = 128
with open('train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        if first_line < 1:
            first_line += 1
            continue
        
        data = row[0].split(',')
        filename = "train_images/"+data[0]
        image = read_resized_image(filename, 128)
