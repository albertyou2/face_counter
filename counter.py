import cv2
import numpy as np
import datetime as dt

# constant
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
OPENCV_METHODS = {
    "Correlation": 0,
    "Chi-Squared": 1,
    "Intersection": 2,
    "Hellinger": 3}
hist_limit = 0.6
ttl = 1 * 60
q_limit = 3

# init variables
total_count = 0
prev_count = 0
total_delta = 0
stm = {}
q = []
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
video_capture = cv2.VideoCapture(0)

while True:
    for t in list(stm):  # short term memory
        if (dt.datetime.now() - t).seconds > ttl:
            stm.pop(t, None)

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    count = len(faces)
    if len(q) >= q_limit: del q[0]
    q.append(count)

    isSame = True
    for c in q:  # Protect from fluctuation
        if c != count: isSame = False
    if isSame is False: continue

    max_hist = 0
    total_delta = 0
    for (x, y, w, h) in faces:
        # Draw a rectangle around the faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if count == prev_count: continue

        # set up the ROI
        face = frame[y: y + h, x: x + w]
        hsv_roi = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(face, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        face_hist = cv2.calcHist([face], [0], mask, [180], [0, 180])
        cv2.normalize(face_hist, face_hist, 0, 255, cv2.NORM_MINMAX)

        isFound = False
        for t in stm:
            hist_compare = cv2.compareHist(stm[t], face_hist, OPENCV_METHODS["Correlation"])
            if hist_compare > max_hist: max_hist = hist_compare
            if hist_compare >= hist_limit: isFound = True

        if (len(stm) == 0) or (isFound is False and max_hist > 0):
            total_delta += 1
        stm[dt.datetime.now()] = face_hist

    if prev_count != count:
        total_count += total_delta
        print("", count, " > ", total_count)
        prev_count = count

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
