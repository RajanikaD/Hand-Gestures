{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "043956db",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-f7f99fa3d58c>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     29\u001b[0m \u001b[1;32mwhile\u001b[0m \u001b[1;32mTrue\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     30\u001b[0m     \u001b[1;31m# Read each frame from the webcam\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 31\u001b[1;33m     \u001b[0m_\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mframe\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcap\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mread\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     32\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     33\u001b[0m     \u001b[0mx\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mc\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mframe\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "import cv2\n",
    "import numpy as np\n",
    "import mediapipe as mp\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "# initialize mediapipe\n",
    "mpHands = mp.solutions.hands\n",
    "hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)\n",
    "mpDraw = mp.solutions.drawing_utils\n",
    "\n",
    "# Load the gesture recognizer model\n",
    "model = load_model('mp_hand_gesture')\n",
    "\n",
    "# Load class names\n",
    "f = open('gesture.names', 'r')\n",
    "classNames = f.read().split('\\n')\n",
    "f.close()\n",
    "print(classNames)\n",
    "\n",
    "\n",
    "# Initialize the webcam\n",
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "while True:\n",
    "    # Read each frame from the webcam\n",
    "    _, frame = cap.read()\n",
    "\n",
    "    x, y, c = frame.shape\n",
    "\n",
    "    # Flip the frame vertically\n",
    "    frame = cv2.flip(frame, 1)\n",
    "    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "    # Get hand landmark prediction\n",
    "    result = hands.process(framergb)\n",
    "\n",
    "    # print(result)\n",
    "    \n",
    "    className = ''\n",
    "\n",
    "    # post process the result\n",
    "    if result.multi_hand_landmarks:\n",
    "        landmarks = []\n",
    "        for handslms in result.multi_hand_landmarks:\n",
    "            for lm in handslms.landmark:\n",
    "                # print(id, lm)\n",
    "                lmx = int(lm.x * x)\n",
    "                lmy = int(lm.y * y)\n",
    "\n",
    "                landmarks.append([lmx, lmy])\n",
    "\n",
    "            # Drawing landmarks on frames\n",
    "            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)\n",
    "\n",
    "            # Predict gesture\n",
    "            prediction = model.predict([landmarks])\n",
    "            # print(prediction)\n",
    "            classID = np.argmax(prediction)\n",
    "            className = classNames[classID]\n",
    "\n",
    "    # show the prediction on the frame\n",
    "    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, \n",
    "                   1, (0,0,255), 2, cv2.LINE_AA)\n",
    "\n",
    "    # Show the final output\n",
    "    cv2.imshow(\"Output\", frame) \n",
    "\n",
    "    if cv2.waitKey(1) == ord('q'):\n",
    "        break\n",
    "\n",
    "# release the webcam and destroy all active windows\n",
    "cap.release()\n",
    "\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6de14ad7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
