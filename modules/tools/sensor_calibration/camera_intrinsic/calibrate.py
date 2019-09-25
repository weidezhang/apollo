import numpy as np
import cv2
import cv2.aruco as aruco
import json 
import os

# params
markersX = 3;                # Number of markers in X direction
markersY = 5;                # Number of markers in Y direction
markerLength = 2*72;           # Marker side length (in pixels)
markerSeparation = 1*72;       # Separation between two consecutive markers in the grid (in pixels)
dictionaryId = '6x6_250';    # dictionary id
margins = markerSeparation;  # Margins size (in pixels)
borderBits = 1;              # Number of bits in marker borders
inch2meter = 0.0254 #inch to meter multiplier
imageSize = np.array([markersX, markersY]) * (markerLength + markerSeparation) - markerSeparation + 2 * margins;
print(imageSize)
dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.GridBoard_create(markersX, markersY, 2*inch2meter, 1*inch2meter, dictionary)
  

def calibrate_camera(allCorners, allIds, counter, imsize):
    print('start to calibrate cameras')
    distCoeff = np.zeros((5,1))
    #print(allIds)
    allIds = np.array(allIds)
    counter = np.array(counter)
    flags = (cv2.CALIB_RATIONAL_MODEL)
    cameraMatrixInit = np.array([[ 2000.,    0., imsize[0]/2.],
                                [    0., 2000., imsize[1]/2.],
                                [    0.,    0.,           1.]])

    (ret, camera_matrix, distortion_coefficients0,rotation_vectors, translation_vectors) = cv2.aruco.calibrateCameraAruco(corners = allCorners, ids = allIds, counter = counter, board=board, imageSize=imsize, cameraMatrix = cameraMatrixInit, distCoeffs=distCoeff, flags=flags, criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    print(camera_matrix)
    print(distortion_coefficients0)
    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


def validate_calibration(mtx, dist, test_img):
    frame = cv2.imread(test_img)
    img_undistort = cv2.undistort(frame, mtx, dist, None)
    cv2.imwrite("/tmp/undist.jpg", img_undistort)

def read_dir(data_dir):
    datadir = "./data/6mm/"
    images = np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".jpg") ])
    return images

def detect_corner(images):
    global dictionary
    allCorners = []
    allIds = []
    counter = []
    for im in images:
        print("process image %s"%im)
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        imsize = gray.shape
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary)
        if corners is not None and len(corners) >0:
            allCorners.extend(corners)
            allIds.extend(ids)
            counter.append(len(corners))
            print("total corner got is %d"%len(corners))
    return (allCorners, allIds, counter, imsize)


def main():
    images = read_dir("./data")
    (allCorners, allIds, counter, imsize) = detect_corner(images)
    (a,b,c,d,e) = calibrate_camera(allCorners, allIds, counter, imsize)
    validate_calibration(b,c,"/home/weide/dev/arucomarker/data/6mm/test2.jpg") 

if __name__=="__main__":
    main()

