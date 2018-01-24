from argparse import _ActionsContainer

import pyforms
from   pyforms          import BaseWidget
from   pyforms.Controls import ControlText
from   pyforms.Controls import ControlButton
from   pyforms.Controls import ControlFile
from   pyforms.Controls import ControlSlider
from   pyforms.Controls import ControlPlayer
from   pyforms.Controls import ControlCheckBoxList

import cv2

class ComputerVisionAlgorithm(BaseWidget):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    def __init__(self):
        super(ComputerVisionAlgorithm, self).__init__('Eyetracking object detection with Deep Neural Networks')

        # Definition of the forms fields
        self._videofile = ControlFile('Video')
        self._outputfile = ControlFile('Results output file')
        self._objectclasses = ControlCheckBoxList('Select the classes you expect in the video')
        self._player = ControlPlayer('Player')
        self._loadbutton = ControlButton('Load Video')
        self._runbutton = ControlButton('Run')

        # Define the function that will be called when a file is selected
        # TODO: NOT WORKING!!!
        self._videofile.changed = self.__videoFileSelectionEvent

        self._loadbutton.value = self.__loadEvent

        # Define the event that will be called when the run button is processed
        self._runbutton.value = self.__runEvent
        # Define the event called before showing the image in the player
        self._player.processFrame = self.__processFrame

        self._player.click_event = self.__returnOpenCVBox

        # Define the organization of the Form Controls
        self._formset = [
            ('_videofile', '_outputfile'),
            ('_loadbutton', '_runbutton'),
            ('_objectclasses', '_player')
        ]

        # Adding all Classes to the selection pane
        for val in self.CLASSES:
            self._objectclasses += (val, True)

    def __videoFileSelectionEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        """ self._player.value = self._videofile.value """
        self._player.value = cv2.VideoCapture(self._videofile.value)
        print (self._player.video_index)
        print (self._player.max)
        print (self._player.fps)


    def __processFrame(self, frame):
        """
        Do some processing to the frame and return the result frame
        """
        return frame

    def __loadEvent(self):
        self._player.value = cv2.VideoCapture(self._videofile.value)
        print (self._player.video_index)
        print (self._player.max)
        print (self._player.fps)

    def __runEvent(self):
        """
        After setting the best parameters run the full algorithm
        """
        self._player.play()
        pass

    def __returnOpenCVBox(self, event, x, y):
        """
        This is where OpenCV is called to open a selection window in that particular frame
        :return:
        """
        print (event)
        print (x)
        print (y)
        pass

#Execute the application
if __name__ == "__main__":   pyforms.start_app( ComputerVisionAlgorithm )