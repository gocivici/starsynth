import cv2
import numpy
from numpy import asarray
from numpy import savetxt
from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse

#osc initialize
osc_startup()
osc_udp_client("127.0.0.1", 57120, "local")
note_C = oscbuildparse.OSCMessage("/note_C", None, ["text"])
note_Db = oscbuildparse.OSCMessage("/note_Db", None, ["text"])
note_D = oscbuildparse.OSCMessage("/note_D", None, ["text"])
note_Eb = oscbuildparse.OSCMessage("/note_Eb", None, ["text"])
note_E = oscbuildparse.OSCMessage("/note_E", None, ["text"])
note_F = oscbuildparse.OSCMessage("/note_F", None, ["text"])
note_Gb = oscbuildparse.OSCMessage("/note_Gb", None, ["text"])
note_G = oscbuildparse.OSCMessage("/note_G", None, ["text"])
note_Ab = oscbuildparse.OSCMessage("/note_Ab", None, ["text"])
note_A = oscbuildparse.OSCMessage("/note_A", None, ["text"])
note_Bb = oscbuildparse.OSCMessage("/note_Bb", None, ["text"])
note_B = oscbuildparse.OSCMessage("/note_B", None, ["text"])

#import imutils
CONNECTIVITY = 4
DRAW_CIRCLE_RADIUS = 5
imageorg = cv2.imread('testimage.jpg')
imageorginal = cv2.imread('testimage.jpg')
gray = cv2.cvtColor(imageorg, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]
height, width, channels = imageorg.shape

# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
# thresh = cv2.erode(thresh, None, iterations=2)
# thresh = cv2.dilate(thresh, None, iterations=4)
components = cv2.connectedComponentsWithStats(thresh, CONNECTIVITY, cv2.CV_32S)
centers = components[3]
savetxt('data.csv', centers, delimiter=',')

def calculateDistance(x1,y1,x2,y2):
     dist = numpy.sqrt((x2 - x1)**2 + (y2 - y1)**2)
     return dist
#print calculateDistance(x1, y1, x2, y2)

print(centers)
print(height)
print(width)
centerDistance = numpy.zeros((int(centers.shape[0]),4))
#print(centers.shape[0])
print(centerDistance)


count=0
for center in centers:
    cv2.circle(imageorg, (int(center[0]), int(center[1])), DRAW_CIRCLE_RADIUS, (255), thickness=1)
    #numpy.append(centers[0], calculateDistance(int(width / 2), int(height / 2),int(center[0]), int(center[1])))
    numpy.put(centerDistance[count], [0,1, 2], [center[0],center[1], calculateDistance(int(width / 2), int(height / 2),int(center[0]), int(center[1]))])
    count+=1
print(centers)
print(centerDistance)

line_thickness = 1
cv2.line(imageorg, (int(width / 2), 0), (int(width / 2), int(height)), (0, 0, 255), thickness=line_thickness)
cv2.line(imageorg, (int(width / 2 - height / 2), int(height / 2)), (int(width/2 + height / 2), int(height / 2)), (0, 0, 255), thickness=line_thickness)
cv2.line(imageorg, (int((width / 2) - numpy.sqrt(3)*int(height / 2)/2), int(height / 2 - int(height / 2) / 2)), (int((width / 2) + numpy.sqrt(3)*int(height / 2)/2), int(height / 2 + int(height / 2) / 2)), (0, 0, 255), thickness=line_thickness)
cv2.line(imageorg, (int((width / 2) - int(height / 2) / 2), int(height / 2 - numpy.sqrt(3)*int(height / 2)/2)), (int((width / 2) + int(height / 2) / 2), int(height / 2 + numpy.sqrt(3)*int(height / 2)/2)), (0, 0, 255), thickness=line_thickness)
cv2.line(imageorg, (int((width / 2) - numpy.sqrt(3)*int(height / 2)/2), int(height / 2 + int(height / 2) / 2)), (int((width / 2) + numpy.sqrt(3)*int(height / 2)/2), int(height / 2 - int(height / 2) / 2)), (0, 0, 255), thickness=line_thickness)
cv2.line(imageorg, (int((width / 2) - int(height / 2) / 2), int(height / 2 + numpy.sqrt(3)*int(height / 2)/2)), (int((width / 2) + int(height / 2) / 2), int(height / 2 - numpy.sqrt(3)*int(height / 2)/2)), (0, 0, 255), thickness=line_thickness)

radiusCount=0
# for x in centerDistance:
#     drawingImage = imageorg.copy()
#     cv2.circle(drawingImage, (int(width / 2), int(height / 2)), radiusCount, (0, 255, 0), thickness=2)
#     cv2.imshow('Overlay',drawingImage)
#     cv2.waitKey(20)
#     radiusCount+=1
#     print(count)

def inArea(xp,yp,x1,y1,x2,y2,x3,y3):
    # d1 = (px1-x2)*(y1-y2) - (x1-x2)*(py1-y2)
    # d2 = (px1-x3)*(y2-y3) - (x2-x3)*(py1-y3)
    # d3 = (px1-x3)*(y3-y1) - (x3-x1)*(py1-y1)
    c1 = (x2-x1)*(yp-y1)-(y2-y1)*(xp-x1)
    c2 = (x3-x2)*(yp-y2)-(y3-y2)*(xp-x2)
    c3 = (x1-x3)*(yp-y3)-(y1-y3)*(xp-x3)

    if (c1<0 and c2<0 and c3<0) or (c1>0 and c2>0 and c3>0):
        print("The star (" + str(xp) + "," + str(yp) + ") is in the region," + "(" + str(x1) + "," + str(y1) + ")" + "(" + str(x2) + "," + str(y2) + ")" + "(" + str(x3) + "," + str(y3) + ")")
        return True
    else:
        #print("The point is outside the triangle.")
        return False

testCount = 0;
for x in range(int(height / 2)):
    drawingImage = imageorg.copy()
    cv2.circle(drawingImage, (int(width / 2), int(height / 2)), x, (0, 255, 0), thickness=1)
    for distance in centerDistance:
        if (distance[3]==0) and (distance[2] <= x) and inArea(int(distance[0]),int(distance[1]),int(width/2),int(height/2),int(width/2),0,int((width / 2) + int(height / 2) / 2), int(height / 2 - numpy.sqrt(3)*int(height / 2)/2)):
            cv2.circle(imageorg, (int(distance[0]),int(distance[1])), DRAW_CIRCLE_RADIUS, (0, 255, 0), thickness=1)
            testCount+=1
            osc_send(note_C, "local")
            osc_process()
            #print(testCount)
            distance[3] = 1
        # elif (distance[3]==0) and (distance[2] <= x) and inArea(int(distance[0]),int(distance[1]),int(width/2),int(height/2),int((width / 2) + int(height / 2) / 2), int(height / 2 - numpy.sqrt(3)*int(height / 2)/2),int((width / 2) + numpy.sqrt(3)*int(height / 2)/2), int(height / 2 - int(height / 2) / 2)):
        #     cv2.circle(imageorg, (int(distance[0]),int(distance[1])), DRAW_CIRCLE_RADIUS, (200, 120, 200), thickness=1)
        #     osc_send(note_Db, "local")
        #     osc_process()
        #     distance[3] = 1
        elif (distance[3]==0) and (distance[2] <= x) and inArea(int(distance[0]),int(distance[1]),int(width/2),int(height/2),int((width / 2) + numpy.sqrt(3)*int(height / 2)/2), int(height / 2 - int(height / 2) / 2),int(width/2 + height / 2), int(height / 2)):
            cv2.circle(imageorg, (int(distance[0]),int(distance[1])), DRAW_CIRCLE_RADIUS, (50, 150, 20), thickness=1)
            osc_send(note_D, "local")
            osc_process()
            distance[3] = 1
        # elif (distance[3]==0) and (distance[2] <= x) and inArea(int(distance[0]),int(distance[1]),int(width/2),int(height/2),int(width/2 + height / 2), int(height / 2),int((width / 2) + numpy.sqrt(3)*int(height / 2)/2), int(height / 2 + int(height / 2) / 2)):
        #     cv2.circle(imageorg, (int(distance[0]),int(distance[1])), DRAW_CIRCLE_RADIUS, (200, 100, 50), thickness=1)
        #     osc_send(note_Eb, "local")
        #     osc_process()
        #     distance[3] = 1
        elif (distance[3]==0) and (distance[2] <= x) and inArea(int(distance[0]),int(distance[1]),int(width/2),int(height/2),int((width / 2) + numpy.sqrt(3)*int(height / 2)/2), int(height / 2 + int(height / 2) / 2),int((width / 2) + int(height / 2) / 2), int(height / 2 + numpy.sqrt(3)*int(height / 2)/2)):
            cv2.circle(imageorg, (int(distance[0]),int(distance[1])), DRAW_CIRCLE_RADIUS, (200, 200, 50), thickness=1)
            osc_send(note_E, "local")
            osc_process()
            distance[3] = 1
        elif (distance[3]==0) and (distance[2] <= x) and inArea(int(distance[0]),int(distance[1]),int(width/2),int(height/2),int((width / 2) + int(height / 2) / 2), int(height / 2 + numpy.sqrt(3)*int(height / 2)/2),int(width / 2),int(height)):
            cv2.circle(imageorg, (int(distance[0]),int(distance[1])), DRAW_CIRCLE_RADIUS, (200, 200, 200), thickness=1)
            osc_send(note_F, "local")
            osc_process()
            distance[3] = 1
        # elif (distance[3]==0) and (distance[2] <= x) and inArea(int(distance[0]),int(distance[1]),int(width/2),int(height/2),int(width / 2),int(height),int((width / 2) - int(height / 2) / 2), int(height / 2 + numpy.sqrt(3)*int(height / 2)/2)):
        #     cv2.circle(imageorg, (int(distance[0]),int(distance[1])), DRAW_CIRCLE_RADIUS, (0, 200, 200), thickness=1)
        #     osc_send(note_Gb, "local")
        #     osc_process()
        #     distance[3] = 1
        elif (distance[3]==0) and (distance[2] <= x) and inArea(int(distance[0]),int(distance[1]),int(width/2),int(height/2),int((width / 2) - int(height / 2) / 2), int(height / 2 + numpy.sqrt(3)*int(height / 2)/2),int((width / 2) - numpy.sqrt(3)*int(height / 2)/2), int(height / 2 + int(height / 2) / 2)):
            cv2.circle(imageorg, (int(distance[0]),int(distance[1])), DRAW_CIRCLE_RADIUS, (0, 250, 200), thickness=1)
            osc_send(note_G, "local")
            osc_process()
            distance[3] = 1
        # elif (distance[3]==0) and (distance[2] <= x) and inArea(int(distance[0]),int(distance[1]),int(width/2),int(height/2),int((width / 2) - numpy.sqrt(3)*int(height / 2)/2), int(height / 2 + int(height / 2) / 2),int((width / 2) - (height / 2)),int(height / 2) ):
        #     cv2.circle(imageorg, (int(distance[0]),int(distance[1])), DRAW_CIRCLE_RADIUS, (0,50, 250), thickness=1)
        #     osc_send(note_Ab, "local")
        #     osc_process()
        #     distance[3] = 1
        elif (distance[3]==0) and (distance[2] <= x) and inArea(int(distance[0]),int(distance[1]),int(width/2),int(height/2),int((width / 2) - (height / 2)),int(height / 2),int((width / 2) - numpy.sqrt(3)*int(height / 2)/2), int(height / 2 - int(height / 2) / 2) ):
            cv2.circle(imageorg, (int(distance[0]),int(distance[1])), DRAW_CIRCLE_RADIUS, (0,150, 250), thickness=1)
            osc_send(note_A, "local")
            osc_process()
            distance[3] = 1
        # elif (distance[3]==0) and (distance[2] <= x) and inArea(int(distance[0]),int(distance[1]),int(width/2),int(height/2),int((width / 2) - numpy.sqrt(3)*int(height / 2)/2), int(height / 2 - int(height / 2) / 2),int((width / 2) - int(height / 2) / 2), int(height / 2 - numpy.sqrt(3)*int(height / 2)/2) ):
        #     cv2.circle(imageorg, (int(distance[0]),int(distance[1])), DRAW_CIRCLE_RADIUS, (0,200, 250), thickness=1)
        #     osc_send(note_Bb, "local")
        #     osc_process()
        #     distance[3] = 1
        elif (distance[3]==0) and (distance[2] <= x) and inArea(int(distance[0]),int(distance[1]),int(width/2),int(height/2),int((width / 2) - int(height / 2) / 2), int(height / 2 - numpy.sqrt(3)*int(height / 2)/2),int(width / 2),0 ):
            cv2.circle(imageorg, (int(distance[0]),int(distance[1])), DRAW_CIRCLE_RADIUS, (50,250, 250), thickness=1)
            osc_send(note_B, "local")
            osc_process()
            distance[3] = 1
    #cv2.circle(imageorg, (389,33), DRAW_CIRCLE_RADIUS, (0, 255, 0), thickness=1)

    cv2.imshow('Overlay',drawingImage)
    cv2.waitKey(100)

#returns true if point is in triangle ABC
#input is (point,A,B,C)




# line_thickness = 1
# cv2.line(imageorg, (int(width / 2), 0), (int(width / 2), int(height)), (0, 0, 255), thickness=line_thickness)
# cv2.line(imageorg, (int(width / 2 - height / 2), int(height / 2)), (int(width/2 + height / 2), int(height / 2)), (0, 0, 255), thickness=line_thickness)
# cv2.line(imageorg, (int((width / 2) - numpy.sqrt(3)*int(height / 2)/2), int(height / 2 - int(height / 2) / 2)), (int((width / 2) + numpy.sqrt(3)*int(height / 2)/2), int(height / 2 + int(height / 2) / 2)), (0, 0, 255), thickness=line_thickness)
# cv2.line(imageorg, (int((width / 2) - int(height / 2) / 2), int(height / 2 - numpy.sqrt(3)*int(height / 2)/2)), (int((width / 2) + int(height / 2) / 2), int(height / 2 + numpy.sqrt(3)*int(height / 2)/2)), (0, 0, 255), thickness=line_thickness)
# cv2.line(imageorg, (int((width / 2) - numpy.sqrt(3)*int(height / 2)/2), int(height / 2 + int(height / 2) / 2)), (int((width / 2) + numpy.sqrt(3)*int(height / 2)/2), int(height / 2 - int(height / 2) / 2)), (0, 0, 255), thickness=line_thickness)
# cv2.line(imageorg, (int((width / 2) - int(height / 2) / 2), int(height / 2 + numpy.sqrt(3)*int(height / 2)/2)), (int((width / 2) + int(height / 2) / 2), int(height / 2 - numpy.sqrt(3)*int(height / 2)/2)), (0, 0, 255), thickness=line_thickness)
#cv2.line(imageorg, (0, int(height)), (int(width), 0), (0, 0, 255), thickness=line_thickness)
#cv2.line(imageorg, (0, 225), (700, 225), (0, 0, 255), thickness=line_thickness)


#cv2.imshow('Original image',imageorginal)
#cv2.imshow('Overlay',imageorg)
#cv2.imshow('Gray image', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
osc_terminate()
