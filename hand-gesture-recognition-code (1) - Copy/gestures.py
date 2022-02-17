{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "043956db",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import mediapipe as mp\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import load_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6de14ad7",
   "metadata": {},
   "outputs": [],
   "source": [
    "mpHands = mp.solutions.hands\n",
    "hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)\n",
    "mpDraw = mp.solutions.drawing_utils"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b8c70e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = load_model('mp_hand_gesture')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0773eb48",
   "metadata": {},
   "outputs": [],
   "source": [
    "f = open('gesture.names', 'r')\n",
    "classNames = f.read().split('\\n')\n",
    "f.close()\n",
    "print(classNames)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "49b24e32",
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "while True:\n",
    "    _, frame = cap.read()\n",
    "\n",
    "    x, y, c = frame.shape\n",
    "\n",
    "    frame = cv2.flip(frame, 1)\n",
    "    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "    result = hands.process(framergb)\n",
    "\n",
    "    \n",
    "    className = ''\n",
    "\n",
    "    if result.multi_hand_landmarks:\n",
    "        landmarks = []\n",
    "        for handslms in result.multi_hand_landmarks:\n",
    "            for lm in handslms.landmark:\n",
    "                lmx = int(lm.x * x)\n",
    "                lmy = int(lm.y * y)\n",
    "\n",
    "                landmarks.append([lmx, lmy])\n",
    "\n",
    "            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)\n",
    "\n",
    "            prediction = model.predict([landmarks])\n",
    "            classID = np.argmax(prediction)\n",
    "            className = classNames[classID]\n",
    "\n",
    "    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, \n",
    "                   1, (0,0,255), 2, cv2.LINE_AA)\n",
    "\n",
    "    cv2.imshow(\"Output\", frame) \n",
    "\n",
    "    if cv2.waitKey(1) == ord('q'):\n",
    "        break\n",
    "\n",
    "cap.release()\n",
    "\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a63b223a",
   "metadata": {},
   "outputs": [],
   "source": [
    "framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "  result = hands.process(framergb)\n",
    "  className = ''\n",
    "  if result.multi_hand_landmarks:\n",
    "      landmarks = []\n",
    "      for handslms in result.multi_hand_landmarks:\n",
    "          for lm in handslms.landmark:\n",
    "              lmx = int(lm.x * x)\n",
    "              lmy = int(lm.y * y)\n",
    "              landmarks.append([lmx, lmy])\n",
    "          mpDraw.draw_landmarks(frame, handslms, \n",
    "mpHands.HAND_CONNECTIONS)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b6eda7b",
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
