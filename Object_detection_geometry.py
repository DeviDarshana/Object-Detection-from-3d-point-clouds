#!/usr/bin/env python
# coding: utf-8

# # Range Image

# In[1]:


from plyfile import PlyData, PlyElement
from pyntcloud import PyntCloud

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale

import numpy as np
from numpy.linalg import inv

import os

import scipy
from scipy import ndimage
import imutils


# # Functions

# In[2]:



def transformation_matrix(points_selected_to_form_plane):
    ''' Finding the transformation matrix '''
    
    p1, p2, p3 = points_selected_to_form_plane
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    ax, ay, az = a = [x2-x1,y2-y1,z2-z1]
    bx, by, bz = b = [x3-x2,y3-y2,z3-z2]                                             
    #u = a cross b
    u = [ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx]                                               
    ux,uy,uz = u
    # v = a cross u 
    v = [ay*uz-az*uy, az*ux-ax*uz, ax*uy-ay*ux]
    new_basis = np.asarray([a,u,v])                    # the new-basis to which pointcloud array needs to be transformed
    x,y,z =([1,0,0],[0,1,0],[0,0,1])
    old_basis = np.asarray([x,y,z])
    B = new_basis
    A = old_basis
    inv_B = inv(np.matrix(B))
    trans_matrix = np.dot(A,inv_B)                                # the required transformation matrix
    return trans_matrix,B,A


def rescale_df_columns(cloud_df):
    '''Rescaling back by factor 10 '''
    
    cloud_df['x'] = cloud_df['x'].div(10)
    cloud_df['y'] = cloud_df['y'].div(10)
    cloud_df['z'] = cloud_df['z'].div(10)
    cloud_df = cloud_df.rename(columns={'scalar_Scalar_field': 'Point_cloud_seg_id'})
    cloud_df["Point_cloud_seg_id"] = cloud_df["Point_cloud_seg_id"].astype(int)
    return cloud_df


def rotate_point_cloud(cloud):
    '''Rotate the input point cloud using the
    transformation matrix from old basis to new basis'''
    
    pcloud_np = np.asarray(cloud.points)                                       # Convert point cloud data to numpy array
    points = pcloud_np
    x_col =points[:,0]
    y_col = points[:,1]
    nz = points[:,7:8]
    z_col =nz  
    points_xy = np.stack((x_col, y_col), axis=1)
    points_xyz =  np.hstack((points_xy,z_col))
    trans_matrix,new_basis,old_basis = transformation_matrix(points_sel)
    P = points_xyz                                                             # point cloud converted to numpy format passed to variable P
    P_newbasis_trans = (np.dot(P,trans_matrix))                                # to get the points transformed into the new basis
    P_newbasis_t = np.asarray(P_newbasis_trans)                                # converting to array form
    return P_newbasis_t,points_xyz


def scale_data (transformed_point_cloud):
    ''' Scaling the point cloud numpy array '''
    
    x_col1 = preprocessing.minmax_scale(transformed_point_cloud[:,0],feature_range=(0,1649))
    y_col1 = preprocessing.minmax_scale(transformed_point_cloud[:,1], feature_range=(0,1449))
    z_col1 =transformed_point_cloud[:,2:3]
    P_newbasis_new_norm_xy = np.stack((x_col1, y_col1), axis=1)
    P_newbasis_new_norm_xy.shape
    P_newbasis_row=points_xyz.shape[0]
    P_newbasis_new_norm_xy_new = P_newbasis_new_norm_xy.reshape(P_newbasis_row, 2)
    P_newbasis_new_norm_xyz = np.hstack((P_newbasis_new_norm_xy_new,z_col1))
    return P_newbasis_new_norm_xyz


def rangeImage(scaled_point_cloud):
    '''Projecting points into range image; 
    storing the pixel indices of the laser 
    points in the dataframe; having z=normal 
    vector along z-direction; passing only the max z value'''
    
    img = np.zeros((1650,1450,3))                                            #initialize empty numpy array of size(1650,1450,3)
    points_norm =scaled_point_cloud                                     #pass pcloud_np_xyz values to points variable  
    footprintSize = [5,5]                                                    # 5,5 for a 11x11 template. trial and error to choose which is suitable
    point_indices_dict = dict()
    count = 0
    for p in range(len(points_norm)):
        u0= int(points_norm[p][0])
        v0= int (points_norm[p][1])
        z= (points_norm[p][2])                                               # this z-value contains information about the possible borderlines between boxes
        temp=0
        for u in range(u0-footprintSize[0], u0+footprintSize[0], 1):
            for v in range(v0-footprintSize[1], v0+footprintSize[1], 1):        
                if((u>=0)and(u<1650) and (v>=0)and(v<1450)):
                    img[u,v] = max(img[u,v])+z
        new_cloud_df.loc[count, 'pixel_x'] = u0
        new_cloud_df.loc[count, 'pixel_y'] = v0
        count = count+1
    return img,new_cloud_df


def newColumnsDf(cloud_df):
    ''' A new dataframe that can store point indices, 
    pixel indices, image and point cloud segmentation ids'''
    
    new_cloud_df = cloud_df[['x', 'y', 'z','scalar_Nz','laser_point_index','Point_cloud_seg_id']].copy()
    new_cloud_df['pixel_x'] = np.nan
    new_cloud_df['pixel_y'] = np.nan
    new_cloud_df['image_segment_id']=np.nan
    new_cloud_df['center_x']= np.nan
    new_cloud_df['center_y']= np.nan
    new_cloud_df['center_z']= np.nan
    new_cloud_df['segment_width'] = np.nan
    new_cloud_df['segment_height'] = np.nan
    new_cloud_df['angle'] = np.nan
    return new_cloud_df


def grayscaleImage(img):
    ''' Visualising the range image as grayscale image'''
    range_image= img
    rgb_weights = [0.2125, 0.7154, 0.0721]
    grayscale_image = np.dot(img[...,:3], rgb_weights)
    return grayscale_image


# In[3]:


# Reads all point cloud file and projects them onto range images

DIR=r"D:\Thesis\dump\neww\dataset_a"

for i in os.listdir(DIR):
    if i.endswith(".ply"):
        cloud = PyntCloud.from_file(os.path.join(DIR,i)) 
        cloud.points['laser_point_index'] = np.arange(len(cloud.points))                  # Create a new column 'index' that has the row number of each row
        cloud_df = cloud.points
        cloud_df = rescale_df_columns(cloud_df)                                           # Rescale x,y,z columns by factor 10
        points_sel = [[2.379062,0.375020,0.226943],
                     [2.397269,-1.782023,0.182852],
                     [2.305973,-0.595197,-1.865193]]                                              # three points selected manually from the corners of the box pile
        trans_matrix,new_basis,old_basis = transformation_matrix(points_sel)
        P_newbasis_t,points_xyz = rotate_point_cloud(cloud)                               # point cloud after transformation
        scaled_point_cloud = scale_data(P_newbasis_t)                                     # the transformed point cloud dtaa is scled to the size of image
        new_cloud_df = newColumnsDf(cloud_df)
        img,new_cloud_df = rangeImage(scaled_point_cloud)
        i = i.rstrip('.ply')
        new_cloud_df.to_csv('D:/Thesis/dump/neww/dataset_a/textfile_'+i+'.txt')
        range_image= grayscaleImage(img)
        plt.imsave('D:/Thesis/dump/neww/dataset_a/image_'+i+'.jpg',range_image,cmap='gray') 
          


# In[ ]:





# # Segmentation

# In[4]:


import numpy as np
import cv2 as cv
import imutils
import math

import skimage
from skimage import segmentation
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from scipy import ndimage
from scipy.spatial import distance as dist

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import shapely
from shapely.geometry import Polygon

from collections import OrderedDict

import operator
import itertools

import pandas as pd


# # Functions

# In[9]:


def watershedSegment(img):  
    '''Performs watershed segmentation and returns 
    labels, along with image read by opencv'''
    
    blur = cv.GaussianBlur(img,(3,3),0)
    kernel = np.array(([1/9,0.3,1/9],[0.3,1,0.3],[1/9,0.3,1/9]), dtype="float32")
    dst = cv.filter2D(blur,-1,kernel)
    image_grey = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret,th1 = cv.threshold(image_grey,0,256,cv.THRESH_BINARY + cv.THRESH_OTSU)         # thresholding
    kernel1 = np.ones((3,3),np.float32)
    dilate = cv.dilate(th1,kernel,iterations=2)
    erosion = cv.erode(dilate,kernel1,iterations = 3)
    D = ndimage.distance_transform_edt(erosion)                                        # distance transform
    localMax = peak_local_max(D, indices=False, min_distance=70,labels=erosion)        # markers
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=erosion)                                      # apply watershed
    number_segments = (len(np.unique(labels))-1 )
    print("[INFO] {} unique segments found".format(len(np.unique(labels))-1,'in',i )) 
    return labels,image_grey
                        
def contoursUniqueSegment(img,labels,image_grey):  
    ''' Draws contours over each segmented region'''
    
    i=0                                                                                # initialising i=0 for printing the segment numbers on each segments
    areas = []                                                                         # list that contains areas of all segments
    segments ={}                                                                       # dict that contains all segment numbers and contour pixels
    segments_area ={}                                                                  # dict that contains all segment numbers and segment areas
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(image_grey.shape, dtype="uint8")                               # draw label on the mask
        mask[labels == label] = 255
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)   # detect contours in the mask and grab the largest one
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)
        area = cv.contourArea(c)
        areas.append(int(area))
        for c in cnts:                                                                 # cnts contains all contours
            ((x, y), _) = cv.minEnclosingCircle(c)
            cv.putText(img, "{}".format(i + 1), (int(x) - 10, int(y)),cv.FONT_HERSHEY_SIMPLEX, 1.7, (255,255,255), 3)
            cv.drawContours(img, [c], -1, (256, 256,256), 2)
            i=i+1
            segments[i] = c
            segments_area[i] = [int(area),c]
    plot_contour = plt.imshow(img,cmap='gray')  
    return plot_contour,areas,segments_area


def histogram_from_data(areas_list,hist_bins):                       
    '''Plots histogram and returns the maximum occuring 
    areas of segments'''
    
    n, bins=np.histogram(areas_list,bins= hist_bins)
    ind = np.searchsorted(bins, areas_list, side='right') 
    sort = areas.sort()
    sections = np.split(areas_list, np.cumsum(n[:-1]))
    high_freq_list = max(sections, key=len)                                                   # the list that has highest number of segments
    return high_freq_list


def threshold(list1):     
    ''' Takes the maximum frequency segments areas list 
    and identifies a minimum and a maximum threshold to 
    split detected segments into ideal and smaller segments '''
    
    median = np.median(list1)                                                                 # calculate median at this step
    std_dev = np.std(list1)                                                                   # calculate std. dev. at this step
    maxx= int(median + (5*std_dev))
    minn  = int(median - (3*std_dev))
    thresh = minn,maxx
    return thresh

def dist(p1, p2):
    ''' Calculate distance between
    two points'''
    
    (x1, y1), (x2, y2) = p1, p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def EuclideanDist(A,B):  
    ''' Calculate euclidean distance between
    two points'''
    
    D = dist.euclidean((A[0],A[1]), ( B[0],B[1]))
    return D


def split_small_large_contours(segments_area,thresh): 
    ''' Splitting the segments into small and ideal
    ones based on the defined threshold'''
    
    small_cnts = []
    large_cnts =[]
    mainLarge_cnts = []
    mainSmall_cnts = []
    dict1 ={}
    dict2={}
    for i in segments_area:                                                                 # dict that contains segment number and area   
        if (segments_area[i][0]  in range (thresh[0],thresh[1] )) or (segments_area[i][0] > (thresh[1])) :
            dict1[i] = segments_area[i]  
            large_cnts.append(dict1)
            mainLarge_cnts.append(large_cnts)
        else :
            dict2[i] = segments_area[i]
            small_cnts.append(dict2)
            mainSmall_cnts.append(small_cnts)
        small_cnts=[]
        large_cnts = []
    return mainLarge_cnts, mainSmall_cnts


def draw_contour_large(image,contour_list): 
    '''Draw bounding box from contours  with segment 
    numbers marked'''
    
    for i,j in contour_list[0][0].items():
        x,y,w,h = cv.boundingRect(j[1])
        cv.rectangle(image,(x,y),(x+w,y+h),(123,255,255,0),2)
        ((x, y), _) = cv.minEnclosingCircle(j[1])
        cv.putText(image, "#{}".format(i), (int(x) - 10, int(y)),cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
    plt.imshow(image)

def draw_contour_small(image,contour_list):                          # method to draw small bbox 
    for i,j in contour_list[0][0].items():
        x,y,w,h = cv.boundingRect(j[1])
        cv.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
        ((x, y), _) = cv.minEnclosingCircle(j[1])
        cv.putText(image, "#{}".format(i), (int(x) - 10, int(y)),cv.FONT_HERSHEY_SIMPLEX, 1.2, (255,255, 255), 4)
    plt.imshow(image)

    
def get_seg_id_bbox(small_bbox_list,large_bbox_list,img):
    ''' Get only segment id and bbox from the split 
    large and small contours '''
    
    bbox_coords_small = []
    bbox_coords_large = []
    for i,j in small_bbox_list[0][0].items():
        x,y,w,h = cv.boundingRect(j[1])
        bbox_coords_small.append([i,list((x,y,w,h))])                
        rect = cv.minAreaRect(j[1])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(img,[box],0,(255,0,255),2)
        ((x, y), _) = cv.minEnclosingCircle(j[1])
        cv.putText(img, "#{}".format(i), (int(x) - 10, int(y)),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for i,j in large_bbox_list[0][0].items():
        x,y,w,h = cv.boundingRect(j[1])
        bbox_coords_large.append([i,list((x,y,w,h))])                
        rect = cv.minAreaRect(j[1])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(img,[box],0,(255,0,255),2)
        ((x, y), _) = cv.minEnclosingCircle(j[1])
        cv.putText(img, "#{}".format(i), (int(x) - 10, int(y)),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return bbox_coords_small,bbox_coords_large    
    


def draw_bbox(image,bbox_list):
    '''Draw bounding box when a list with segment ids
    and bbox coordinates are input'''
    
    for i in bbox_list:
        start_point =  (i[1][0],i[1][1])
        end_point = (i[1][0]+i[1][2],i[1][1]+i[1][3])
        image = cv.rectangle(image, start_point, end_point,(255,0,0),5)
    plt.imshow(image)
    
    
def merge_two_bbox(small_bbox):
    '''Returns a list containing segments that are
    candidates for a merge operation'''
    
    merged_small_bbox=[]
    for i in small_bbox:
        a_id = small_bbox[0][0]                                  # seg id of box 1
        b_id = small_bbox[1][0]                                  # seg id of box 2
        a = small_bbox[0][1]                                     # fetch bbox of box 1
        b = small_bbox[1][1]                                     # fetch bbox of box 2
        left= min(a[0], b[0])                                    # min x  of box 1,box2
        top = min(a[1], b[1])                                    # min y of box 1,box2
        right = max(a[0]+a[2], b[0]+b[2])                        # max x of box 1, box 2
        bottom = max(a[1]+a[3], b[1]+b[3])                       # max y of box 1, box 2
        result = [left, top, right - left,bottom-top]
        merged_small_bbox.append(result)
    return merged_small_bbox

def Average(lst):
    ''' Returns average of a list'''
    
    return sum(lst) / len(lst)

def getXFromRect(item):
    ''' Orders the list based on x-coordinate'''
    
    return item[1][0]

def average_area(small_bbox):
    ''' Defines the average area from a given list'''
    
    small_box_areas=[]
    for i in small_bbox: 
        areas = int(i[1][2]*i[1][3])
        small_box_areas.append(areas)
        avg_area = Average(small_box_areas)
        avg_area = int(avg_area)
    return avg_area

def min_max_thresh(single_boxes,avg_area):
    ''' Defines a threshold to get identify ideal segments'''
    
    final_single_boxes=[]
    for j in single_boxes:
        areas = int(j[1][2]*j[1][3])
        min_thresh = int((avg_area) - (0.40*avg_area))
        max_thresh = int((avg_area) + (0.70*avg_area))
        if (areas in range(min_thresh,max_thresh)):
            final_single_boxes.append(j)
    return final_single_boxes

def min_dist_thresh(small_bbox):
    ''' Defines a distance threshold to identify 
    neighboring segments'''
    
    dist_list = [] 
    dist_avg =[]
    dist_main_avg = []
    for i in small_bbox:
        for j in small_bbox:
            if (i[1] == j[1]):
                continue
            p1 = i[1][0],i[1][1]
            p2 = j[1][0],j[1][1]
            distance = int(dist(p1,p2))
            dist_list.append(distance)
        dist_average = Average(dist_list)
        dist_average = int(dist_average)
        dist_avg.append(dist_average)
    dist_main_average = Average(dist_avg)
    dist_main_avg.append(dist_main_average)
    dist_main_avg = int (dist_main_avg[0])
    min_dthresh = ((dist_main_avg)- 0.80*dist_main_avg)
    return min_dthresh


def merge_dist_condition(small_bbox):
    ''' Condition to merge two candidate segments'''
    
    merged_small_bboxes_id =[]
    merged_small_bboxes = []
    single_boxes =[]
    bool_arr =[]
    min_dthresh = min_dist_thresh(small_bbox)
    for i in small_bbox:
        for j in small_bbox:
            if (i[1] == j[1]):
                continue
            p1 = i[1][0],i[1][1]
            p2 = j[1][0],j[1][1]
            distance = int(dist(p1,p2))
            if(distance < min_dthresh):
                merged = merge_two_bbox([i,j])
                merged_small_bboxes_id.append([[i[0],j[0]],merged[0]])                                # appends the seg id of the two boxes that are merged
                merged_small_bboxes.append(merged[0])
                bool_arr.append(1)
        if(len(bool_arr) == 0):
            single_boxes.append([i[0],i[1]])
        bool_arr = []
    return merged_small_bboxes_id ,single_boxes


def bb_intersection_over_union(boxA, boxB):
    ''' Find IoU between two segments'''
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def intersect_bbox(box_1,box_2):
    ''' Threshold to identify intersecting segments'''
    
    flag = False
    bbox_1 = box_1[1]
    bbox_2 = box_2[1]
    bottom_left_x1 = bbox_1[0]
    top_right_x1 = bbox_1[0] +bbox_1[2]
    bottom_left_y1 = bbox_1[1]
    top_right_y1 = bbox_1[1] +bbox_1[3]
    bottom_left_x2 = bbox_2[0]
    top_right_x2 = bbox_2[0] +bbox_2[2]
    bottom_left_y2 = bbox_2[1]
    top_right_y2 = bbox_2[1] +bbox_2[3]
    iou = bb_intersection_over_union([bottom_left_x1,bottom_left_y1,top_right_x1,top_right_y1],
                                     [bottom_left_x2,bottom_left_y2,top_right_x2,top_right_y2])
    if(iou >= 0.02):
        flag = True
    else:
        flag = False
    
    return flag


def draw_bbox_with_seg_id(image,bbox_list):  
    ''' Draw bounding box when passed as a list'''
    
    for i in bbox_list:
        start_point =  (i[1][0],i[1][1])
        end_point = (i[1][0]+i[1][2],i[1][1]+i[1][3])
        image = cv2.rectangle(image, start_point, end_point,(0,255,0),7)
        x,y = i[1][0],i[1][1]
        image = cv2.putText(image, "#{}".format(i[0]), (int(x)+80 , int(y)+80),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    plt.imshow(image)


def draw_bbox_from_dict(image,dict_all_boxes):  
    ''' Draw bounding box when passed as a dictionary'''
    
    for i,j in dict_all_boxes.items():
        start_point =  (j[0],j[1])
        end_point = (j[0]+j[2],j[1]+j[3])
        image = cv.rectangle(image, start_point, end_point,(0,128,0),7)
        x,y = j[0],j[1]
        image = cv.putText(image, "{}".format(i), (int(x)+80 , int(y)+80),cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    plt.imshow(image)   
    
def midpoint(p1, p2):
    ''' Returns midpoint'''
    
    point = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
    return point

def centroid (list_of_xyz_points):
    ''' Returns centroid from a list of 3d points'''
    
    points = list_of_xyz_points
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    centroid = [sum(x) / len(points), sum(y) / len(points),sum(z) / len(points)]
    return centroid

def polygon(box1,box2):
    ''' Returns a polygon that is formed by the overlap
    of two segments'''
    
    bbox_1 = box1 [1]
    bbox_2 = box2[1]
    p = Polygon([(bbox_1[0],bbox_1[1]),((bbox_1[0]),(bbox_1[1]+bbox_1[3])),
                 ((bbox_1[0]+bbox_1[2]),(bbox_1[1]+bbox_1[3])),((bbox_1[0]+bbox_1[2]),bbox_1[1])])
    q = Polygon([(bbox_2[0],bbox_2[1]),((bbox_2[0]),(bbox_2[1]+bbox_2[3])),
                 ((bbox_2[0]+bbox_2[2]),(bbox_2[1]+bbox_2[3])),((bbox_2[0]+bbox_2[2]),bbox_2[1])])
    return p,q

def remove_duplicate (m_first,large_bbox,final_single_boxes):
    ''' Remove duplicate segments'''
    
    m_first_final = []                                                                                #list without duplicates
    from collections import OrderedDict
    [m_first_final.append(x[1]) for x in m_first if x[1] not in m_first_final]  
    for i in range(len(large_bbox)):
        m_first_final.append(large_bbox[i][1])
    for i in range(len(final_single_boxes)):
        m_first_final.append(final_single_boxes[i][1])
    return m_first_final

    
def all_box_dict(m_first_final):
    ''' All segments after merge function'''
    
    all_boxes=[]
    all_boxes_to_dict=[]
    for i in range(len(m_first_final)):
        all_boxes.append( [i,m_first_final[i]])
        all_boxes_to_dict.append( m_first_final[i])
    dict_all_boxes = { i : all_boxes_to_dict[i] for i in range(0, len(all_boxes_to_dict) ) }
    return dict_all_boxes,all_boxes


def check_overlap(all_boxes):
    ''' Checks which two segments overlap'''
    
    all_bbox_neighbours=[]
    all_bbox_neighbours_str=[]
    for i in all_boxes:
        for j in all_boxes:
            if (i[1] == j[1]):
                continue
            intersection_check = intersect_bbox(i,j)
            if(intersection_check == True):
                if (  str(j)+str(i) not in all_bbox_neighbours_str ):                      # checks if reversed string is in list and appends         
                    all_bbox_neighbours.append([i,j])
                    all_bbox_neighbours_str.append(str(i)+str(j))
    return all_bbox_neighbours                                                                    # bbox pairs that has significant overlap  


def remove_overlap(all_bbox_neighbours):
    ''' Removes the overlap between two given segments'''
    
    boxes_overlap_removed=[]
    boxes_overlap_removed_dict={}
    for i in all_bbox_neighbours:
        box_1 = i[0]
        box_2 = i[1]
        bbox_1=box_1[1]
        bbox_2=box_2[1]
        if (box_1[0] in boxes_overlap_removed_dict.keys()):
            bbox_1 = boxes_overlap_removed_dict.get(box_1[0])
            box_1 = [box_1[0],bbox_1]
        if (box_2[0] in boxes_overlap_removed_dict.keys()):
            bbox_2 = boxes_overlap_removed_dict.get(box_2[0])
            box_2 = [box_2[0],bbox_2]
        bbox_1_x_min = bbox_1[0]
        bbox_2_x_min = bbox_2[0]
        bbox_1_y_min = bbox_1[1]
        bbox_2_y_min = bbox_2[1]
        p,q = polygon(box_1,box_2)
        if (bbox_1_x_min>bbox_2_x_min):
            overlap_region = p.intersection(q)
        else:
            overlap_region = q.intersection(p)
        if (overlap_region.is_empty != True):
            x, y = overlap_region.exterior.coords.xy                                 # gets the coordinates of the polygon
            tl,tr,br,bl = [[x[0],y[0]],[x[1],y[1]],[x[2],y[2]],[x[3],y[3]]]
            width = dist(br,bl)
            height = dist(bl,tl)
            width_to_be_reduced =width/2
            height_to_be_reduced = height/2
            if (width>height):
                if (bbox_1_y_min<bbox_2_y_min):
                    update_box_1 = [box_1[0],[bbox_1_x_min,bbox_1_y_min,bbox_1[2],int(bbox_1[3]-height_to_be_reduced)]]  # the new bbox_1 to be appended
                    update_box_2 = [box_2[0],[bbox_2_x_min,int(bbox_2_y_min+height_to_be_reduced),bbox_2[2],int(bbox_2[3]-height_to_be_reduced)]]
                else:
                    update_box_1 = [box_2[0],[bbox_2_x_min,bbox_2_y_min,bbox_2[2],int(bbox_2[3]-height_to_be_reduced)]]  # the new bbox_1 to be appended
                    update_box_2 = [box_1[0],[bbox_1_x_min,int(bbox_1_y_min+height_to_be_reduced),bbox_1[2],int(bbox_1[3]-height_to_be_reduced)]]
                boxes_overlap_removed_dict[update_box_1[0]] = update_box_1[1]
                boxes_overlap_removed_dict[update_box_2[0]] = update_box_2[1]
                boxes_overlap_removed.append([update_box_1,update_box_2])
            else:
                if (bbox_1_x_min<bbox_2_x_min):
                    update_box_1 = [box_1[0],[bbox_1_x_min,bbox_1_y_min,int(bbox_1[2]-width_to_be_reduced),bbox_1[3]]]  # the new bbox_1 to be appended
                    update_box_2 = [box_2[0],[int(bbox_2_x_min+width_to_be_reduced),bbox_2_y_min,int(bbox_2[2]-width_to_be_reduced),bbox_2[3]]]
                else:
                    update_box_1 = [box_2[0],[bbox_2_x_min,bbox_2_y_min,int(bbox_2[2]-width_to_be_reduced),bbox_2[3]]]  # the new bbox_1 to be appended
                    update_box_2 = [box_1[0],[int(bbox_1_x_min+width_to_be_reduced),bbox_1_y_min,int(bbox_1[2]-width_to_be_reduced),bbox_1[3]]]
                boxes_overlap_removed_dict[update_box_1[0]] = update_box_1[1]
                boxes_overlap_removed_dict[update_box_2[0]] = update_box_2[1]
                boxes_overlap_removed.append([update_box_1,update_box_2])
    return boxes_overlap_removed_dict
    
def overlap_removed_box_dict(dict_all_boxes,boxes_overlap_removed_dict):
    ''' Final segments free of overlap'''
    
    non_common_pairs = dict()
    for key in dict_all_boxes:
        if (key not in boxes_overlap_removed_dict):
            continue
        else:
            dict_all_boxes[key] = boxes_overlap_removed_dict[key]
    return dict_all_boxes

def reprojection( dict_all_boxes,new_cloud_df):
    ''' Reprojection of image segmentation results
    to the pointcloud dataset'''
    
    bounding_box = dict_all_boxes
    count = 0 
    row_in_df = new_cloud_df.shape[0]
    for i in range(0,row_in_df):
        pix_x = new_cloud_df.iloc[i]["pixel_x"]
        pix_y = new_cloud_df.iloc[i]["pixel_y"]
        for seg_id,bbox in bounding_box.items():
            y_min  = bbox[0]                                                           # y to x and x to y(right an left hand coordinate systems of 3d and 2d)
            x_min = bbox[1]
            y_max = bbox[0]+bbox[2]
            x_max = bbox[1]+bbox[3]
            if((pix_x>=x_min)and (pix_x<x_max) and(pix_y>=y_min) and (pix_y<y_max)):   # equal signs not on both conditions
                new_cloud_df.loc[i,"image_segment_id"] = int(seg_id)
    return new_cloud_df


# # Implementation                
# 

# In[12]:


#Reads all image files; performs segmentation and reprojects the results to its respective point cloud dataset

DIR=r"D:\Thesis\dump\neww\dataset_a"

for i in os.listdir(DIR):
    if i.endswith(".jpg"):
        img = cv.imread((os.path.join(DIR,i))) 
        labels,image_grey = watershedSegment(img)
        plot_contour,areas,segments_area = contoursUniqueSegment(img,labels,image_grey) 
        img = cv.imread((os.path.join(DIR,i))) 
        high_freq_list = histogram_from_data(areas,10)                                             # plot histogram to separate the small boxes from large boxes
        thresh         = threshold(high_freq_list)
        large,small = split_small_large_contours(segments_area,thresh)
#         large_cnt = draw_contour_large(img,large)                                                # visualize the small and large boxes
#         small_cnt = draw_contour_small(img,small)
        small_bbox,large_bbox = get_seg_id_bbox(small,large,img)                                   # get the bounding box coords and the segment id 
        small_bbox.sort(key = getXFromRect)                                                        # sort the small bounding boxes list based on the x-coord
        m_first,single_boxes = merge_dist_condition(small_bbox)                                    # merge the neighbouring small bounding boxes
        avg_area = average_area(small_bbox)                                                        # find the average area of the small bounding boes to find the min and max threshold 
                                                                                                   # based on which the merging of two boxes should take place
        final_single_boxes = min_max_thresh(single_boxes,avg_area)
        m_first_final = remove_duplicate(m_first,large_bbox,final_single_boxes)                    # Remove duplicate bounding box
        dict_all_boxes,all_boxes = all_box_dict(m_first_final)
        img = cv.imread((os.path.join(DIR,i))) 
        draw_bbox_from_dict(img,dict_all_boxes)
        all_bbox_neighbours=check_overlap(all_boxes)                                               # Check which two pairs of bounding boxes are adjacent and check if the pair has overlap
                                                                                                   # between them is over the set threshold based on the iou percentag
        boxes_overlap_removed_dict = remove_overlap(all_bbox_neighbours)                           # remove the overlap between adjacent bounding boxes
        dict_all_boxes_new = overlap_removed_box_dict(dict_all_boxes,boxes_overlap_removed_dict)   

        image = cv.imread((os.path.join(DIR,i)))                                                   # Read the grayscale imge and draw the final refined bounding boxes
        result = draw_bbox_from_dict(image,dict_all_boxes_new)
        plt.imshow(image,cmap='gray')
        
        plt.imsave('D:/Thesis/dump/neww//dataset_a/segmentation_'+i,image,cmap='gray')             # save the final image with the final bounding boxes    
#         print(i)
        extension = i.rstrip('.jpg')
        extension = extension.strip('image_')
#         print(extension)        
        
        for t in os.listdir(DIR):
            if ( ( t.endswith(".txt")) and (extension in t)): 
#                 print(t)
#                 print("------------")
                textfile = pd.read_csv((os.path.join(DIR,t))) 
                reproject_segments = reprojection(dict_all_boxes_new,textfile)
#                 print(reproject_segments)
                i = i.rstrip('.jpg')
                reproject_segments.to_csv(r'D:\Thesis\dump\neww\dataset_a\reprojected_pointcloud\reprojected_'+i+'.txt')           #saving the new point cloud df to csv and an import into cloud compare to visulaise based on seg id


# # Pose and Dimensions

# In[13]:


import open3d as o3d

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from numpy import *

import math
from scipy.spatial.transform import Rotation as R


# # Functions

# In[14]:


def globalNormBasis (point_cloud):
    ''' Defining the global basis using the x,y 
    and z coordinte values'''
    
    cloud_df1 = np.asarray(point_cloud.points)
    x_col =cloud_df1[:,0]
    y_col = cloud_df1[:,1]
    z = cloud_df1[:,2:3]
    points_xy1 = np.stack((x_col, y_col), axis=1)
    points_xyz1 =  np.hstack((points_xy1,z))
    points_sel = [[2.379062,0.375020,0.226943],
                 [2.397269,-1.782023,0.182852],
                 [2.305973,-0.595197,-1.865193]]                                              # three points selected manually from the corners of the box pile
    trans_matrix,new_basis,old_basis = transformation_matrix(points_sel)
    P_newbasis_trans1 = (np.dot(points_xyz1,trans_matrix)) 
    P_newbasis_t1 = np.asarray(P_newbasis_trans1) 
    return P_newbasis_t1


def fitPLaneLTSQ(XYZ):
    
    '''Fits a plane to a point cloud, 
    Where Z = aX + bY + c        ----Eqn #1
    Rearanging Eqn1: aX + bY -Z +c =0
    
    Gives normal (a,b,-1)
    Normal = (a,b,-1)'''
    
    [rows,cols] = XYZ.shape
    G = np.ones((rows,3))
    G[:,0] = XYZ[:,0]  #X
    G[:,1] = XYZ[:,1]  #Y
    Z = XYZ[:,2]
    (a,b,c),resid,rank,s = np.linalg.lstsq(G,Z) 
    normal = (a,b,-1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return normal

def mag(x): 
    ''' Defines the magnitude'''
    
    return math.sqrt(sum(i**2 for i in x))

def rotation_matrix_from_vectors(vec1, vec2):
    '''Find the rotation matrix that aligns vec1 to vec2
    param vec1: A 3d "source" vector
    param vec2: A 3d "destination" vector
    return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    ''' 
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def rot2eul(R):
    ''' Convert rotation matrix to
    Euler angles'''
    
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))


def segmentGeometry(unique_ids_list,new_cloud_df):
    ''' Extract 3d pose, dimensions and
    orientation of detected segments'''
    
    centroid_dict ={}
    dimension_dict ={}
    angles_dict ={}
    eul_angles_dict = {}
    row_in_df = new_cloud_df.shape[0]
    for i in unique_ids_list:
        if(str(i) != 'nan'):
            '''segments centroid'''
            
            seg_grouped= new_cloud_df.loc[new_cloud_df['image_segment_id'] == i] 
            seg_grouped_xyz = seg_grouped[['x','y','z']]
            seg_grouped_xyz_list = seg_grouped_xyz.values.tolist()
            seg_centroid = centroid(seg_grouped_xyz_list)
            '''segments_dimensions'''
            
            seg_grouped_xy       = seg_grouped[['x','y']]
            seg_grouped_xy_list  = seg_grouped_xy.values.tolist()
            x_max                = seg_grouped_xy['x'].max()
            x_min                = seg_grouped_xy['x'].min()
            y_max                = seg_grouped_xy['y'].max()
            y_min                = seg_grouped_xy['y'].min()
            width                = ((x_max - x_min)*100)
            height               = ((y_max - y_min)* 100)
            '''segments_orientation'''
            
            dataset=np.asarray(seg_grouped_xyz)
            local_norm = fitPLaneLTSQ(dataset)
            global_norm = fitPLaneLTSQ(P_newbasis_t1)
            dot=np.dot(global_norm,local_norm )
            mag_a= mag(global_norm)
            mag_b = mag(local_norm )
            cos_theta = (dot/(mag_a*mag_b))
            theta = math.acos(cos_theta)
            degrees = np.rad2deg(theta)
            angles_dict[i] = [degrees]
            rot_vec = rotation_matrix_from_vectors(global_norm,local_norm)
            rot = R.from_matrix(rot_vec)
            eul = rot2eul(rot_vec)
            eul_deg = rad2deg(eul)
            eul_angles_dict[i] = [eul_deg]
            '''Populating values'''
            
            dimension_dict[i] = [width,height]
            centroid_dict[i] = seg_centroid
        '''Populating values in datafarme columns and exporting
        final csv file'''
        
    for j in range(1,row_in_df):
        seg_id = new_cloud_df.loc[j,"image_segment_id"]
        if(str(seg_id) != 'nan'):
            centroid_list = centroid_dict[seg_id] 
            new_cloud_df.loc[j,"center_x"]= centroid_list[0]
            new_cloud_df.loc[j,"center_y"]= centroid_list[1]
            new_cloud_df.loc[j,"center_z"]= centroid_list[2]
            dimension_list = dimension_dict [seg_id] 
            angles_list = angles_dict [seg_id]
            eul_angles_list = eul_angles_dict[seg_id]
            new_cloud_df.loc[j,"segment_width"]= dimension_list[0]
            new_cloud_df.loc[j,"segment_height"]= dimension_list[1]
            new_cloud_df.loc[j,"angle"]= angles_list[0]
            new_cloud_df.loc[j,"euler_angles_x"] = eul_angles_list[0][0]
            new_cloud_df.loc[j,"euler_angles_y"] = eul_angles_list[0][1]
            new_cloud_df.loc[j,"euler_angles_z"] = eul_angles_list[0][2]
    return new_cloud_df


# In[17]:


#Extracts 3d pose, dimensions and orientation of each detected object

DIR=r"D:\Thesis\dump\neww\dataset_a\reprojected_pointcloud"

for pc in os.listdir(DIR):
    if ( pc.endswith(".ply")): 
        print(pc)
        point_cloud = PyntCloud.from_file(pc)
        P_newbasis_t1 = globalNormBasis(point_cloud)
        extension = pc.rstrip('.ply')     
        for r in os.listdir(DIR):
            if ((r.endswith(".txt")) and (extension in r)):
                print(r)
                file = pd.read_csv((os.path.join(DIR,r))) 
                unique_ids = file.image_segment_id.unique()                           # to get unique values of a particular column
                unique_ids_list = unique_ids.tolist()                                         # convert numpy array to a list
                segment_pose = segmentGeometry(unique_ids_list,file)
                segment_pose.to_csv(r'D:\Thesis\dump\neww\dataset_a\reprojected_pointcloud\final\final_'+r)

