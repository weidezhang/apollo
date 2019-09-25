import numpy as np
import cv2
import cv2.aruco as aruco
import json 
  
'''
drawMarker(...)
drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
'''

# params
markersX = 3;                # Number of markers in X direction
markersY = 5;                # Number of markers in Y direction
markerLength = 2*72;           # Marker side length (in pixels)
markerSeparation = 1*72;       # Separation between two consecutive markers in the grid (in pixels)
dictionaryId = '6x6_250';    # dictionary id
margins = markerSeparation;  # Margins size (in pixels)
borderBits = 1;              # Number of bits in marker borders

imageSize = np.array([markersX, markersY]) * (markerLength + markerSeparation) - markerSeparation + 2 * margins;
print(imageSize)
dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.GridBoard_create(markersX, markersY, 2*72, 72, dictionary)

# show created board
boardImage = aruco.drawPlanarBoard(board, tuple(imageSize), marginSize=margins)

# save image
cv2.imwrite('GridBoard.png', boardImage)
