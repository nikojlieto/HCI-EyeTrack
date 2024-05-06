import cv2
import dlib
import numpy as np
import pyttsx3
import time
tts = pyttsx3.init()

#global variables here
sentence = ""
#switch between word sets 
masterList = {"conj": ["the", "a", "one", "that", sentence, "this", "their", "BACK", "FINISH"],
              "adj": ["red", "big", "sad", "quick", sentence, "loud", "tired", "BACK", "FINISH"],
              "noun": ["pig", "grape", "apple", "shrimp", sentence, "chair", "fox", "BACK", "FINISH"],
              "verb": ["jumped", "cried", "ran", "sat", sentence, "thought", "fled", "BACK", "FINISH"],
              "adv": ["sadly", "slowly", "calmly", "nicely", sentence, "loudly", "slyly", "BACK", "FINISH"]}
switchMasterList = list(masterList.keys())
switchMasterCurrent = switchMasterList[0]
#words to be displayed on the screen
wordList = masterList["conj"]
#all text is colored red for easy visiblity against backgrounds
red = [(0, 0, 255)]
#where the words are located, broken into x and y coordinates
xPositions = [200, 300, 400]
yPositions = [175, 250, 325]
#the screen is divided up into 9 quadrants numbered (0-8)
currentQuadrant = 4
done = False
#the assumed edges of the user's vision
parameters = [100, 500, 100, 400]
#used to calculate how long the gaze has been on a word
manualTimer = 0 

#speak is used at the end, copied from earlier homework
def speak(tts, text):
    tts.say(text)
    tts.runAndWait()

#below functions were directly copied from example code
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass

#img is the canvas you want to draw on, or omit for a new window
def draw_word(texts, colors, positions, img=None):
    if not (img is None):  
        canvas = img
    else:
        # Create a blank canvas (image) of size 800x600 with 3 channels (RGB), filled with black (0)
        canvas = np.zeros((600, 800, 3), dtype="uint8")
 
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    font_scale = 1.5  # Font scale (size)
    thickness = 3  # Thickness of the text
 
    # Color and location
    ##text = "Hello world"
    ##color = (0, 255, 0)  # Text color in BGR (green)
    ##position = (50, 300)  # Bottom-left corner of the text in the canvas
 
    for i in range(len(texts)):
        # Draw
        cv2.putText(canvas, texts[i], positions[i], font, font_scale, colors[i], thickness)
 
    # Show
    if img is None:
        cv2.imshow("Canvas with Text", canvas)

#master function which checks how the sentence should change
def check_words(shape, left, right):
    global currentQuadrant
    global elapse
    global done
    global sentence
    global manualTimer
    global switchMasterList
    global switchMasterCurrent
    global wordList
    global masterList

    #passes the parameters directly to this function
    eye_center = get_eye_center(shape, left, right)
    #check which quadrant the gaze is currently in
    newQuadrant = check_quadrant(eye_center)

    #if the gaze has not moved since the next check
    if currentQuadrant == newQuadrant:
        manualTimer+= 1
        print(manualTimer)
        #if enough time has elapsed
        if manualTimer > 20:
            #if the period is selected, say the sentence and exit the loop
            if newQuadrant == 8:
                done = True
                speak(tts, sentence)
                #middle quadrant is just the sentence display, doesn't change anything
            elif newQuadrant == 6:
                #get rid of part after last space
                sentence.rsplit(' ', 1)[0]
                #same process as below but easier
                smlIndex = switchMasterList.index(switchMasterCurrent)
                if smlIndex > 0:
                    newKey = switchMasterList[(smlIndex-1)]
                else:
                    newKey = switchMasterList[4]                  
                switchMasterCurrent = switchMasterList[((smlIndex+1)%5)]
                wordList = masterList[newKey]
                wordList[4] = sentence
                manualTime = 0
            elif newQuadrant != 4:
                #add to the sentence and reset the timer
                sentence = sentence + " " + (wordList[currentQuadrant])
                #i'd like to calculate new words and change them, but that's a bit much for now.
                #words could be generated and replaced with the exception of quadrants 4 and 8
                #for now a convoluted way to load in new dictionaries
                #check the position of the current key in the master list
                smlIndex = switchMasterList.index(switchMasterCurrent)
                #increment both the current key and position in the master list by one
                #looping around after reaching 4
                newKey = switchMasterList[((smlIndex+1)%5)]
                switchMasterCurrent = switchMasterList[((smlIndex+1)%5)]
                wordList = masterList[newKey]
                #update the new sentence
                wordList[4] = sentence
                manualTimer = 0
    else:
        #reset the timer with the new quadrant
        currentQuadrant = newQuadrant
        manualTimer = 0

def get_eye_center(shape, left, right):
    # Calculate the center of the left eye
    left_eye_center = np.mean(shape[left], axis=0).astype(int)
    # Calculate the center of the right eye
    right_eye_center = np.mean(shape[right], axis=0).astype(int)
    # Calculate the midpoint between the two eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2, 
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    

    return eye_center # [0] is x coordinate, [1] is y coordinate

#checks the quadrant the gaze is currently in
def check_quadrant(eye_center):
    global xPositions
    global yPositions

    x = eye_center[0]
    y = eye_center[1]

    #if the gaze is outside the current parameters where text is displayed/interacted with
    #dynamically expand the parameters
    if x < parameters[0]:
        parameters[0] = x
    if x > parameters[1]:
        parameters[1] = x
    if y < parameters[2]:
        parameters[2] = y
    if y > parameters[3]:
        parameters[3] = y

    #math to determine where quadrant boundaries are
    x1 = int((parameters[0] + parameters[1])/3)
    x2 = x1*2
    y1 = int((parameters[2]+parameters[3])/3)
    y2 = y1*2

    #math to determine where words will be placed in boundaries
    #the word drawing parameters must be integers
    x0 = (parameters[0])
    x00 = (parameters[1])
    y0 = (parameters[2])
    y00 = (parameters[3])
    xsix = int((x00-x0)/6)
    ysix = int((y00-y0)/6)

    #place 1/6, 1/2, and 5/6 of the way along the space
    xPositions = [x0+xsix, x0+xsix*3, x0+xsix*5]
    yPositions = [y0+ysix, y0+ysix*3, y0+ysix*5]

    #nested if statements to see where the gaze currently lies
    quadrant = 5
    if y < y1:
        if x < x1:
            quadrant = 0
        if x < x2:
            quadrant = 1
        else:
            quadrant = 2
    elif y < y2:
        if x < x1:
            quadrant = 3
        if x < x2:
            quadrant = 4
        else:
            quadrant = 5
    else:
        if x < x1:
            quadrant = 6
        if x < x2:
            quadrant = 7
        else:
            quadrant = 8
    #return the current quadrant gaze is in
    return quadrant

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

def gettime():
    global elapse
    elapse = time.time()

while(not done):
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    #print all the words to the screen
    currentWord = 0
    #iterating through all x and y positions
    for p in xPositions:
        for q in yPositions:
            #make sure the word is in the center of the quadrant
            cwLen = int(len(wordList[currentWord])/2)
            #all words are red/green for contrast
            if currentWord == currentQuadrant:
                draw_word([wordList[currentWord]], [(0, 0, 255)], [((p-cwLen), (q-cwLen))], img)
            else:
                draw_word([wordList[currentWord]], [(0, 255, 0)], [((p-cwLen), (q-cwLen))], img)
            currentWord+= 1

    for rect in rects:

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        thresh = cv2.bitwise_not(thresh)
        contouring(thresh[:, 0:mid], mid, img)
        contouring(thresh[:, mid:], mid, img, True)
        #and call my function here
        check_words(shape, left, right)
        # for (x, y) in shape[36:48]:
        #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    # show the image with the face detections + facial landmarks
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

