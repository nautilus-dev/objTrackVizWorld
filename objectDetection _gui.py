import pyforms
from   pyforms          import BaseWidget
from   pyforms.Controls import ControlText
from   pyforms.Controls import ControlButton
from   pyforms.Controls import ControlFile
from   pyforms.Controls import ControlSlider
from   pyforms.Controls import ControlPlayer

import cv2

class ComputerVisionAlgorithm(BaseWidget):

    def __init__(self):
        super(ComputerVisionAlgorithm, self).__init__('Computer vision algorithm example')

        # Definition of the forms fields
        self._videofile = ControlFile('Video')
        self._outputfile = ControlText('Results output file')
        self._threshold = ControlSlider('Threshold', 114, 0, 255)
        self._blobsize = ControlSlider('Minimum blob size', 100, 100, 2000)
        self._player = ControlPlayer('Player')
        self._runbutton = ControlButton('Run')

        # Define the function that will be called when a file is selected
        # TODO: NOT WORKING!!!
        self._videofile.changed = self.__videoFileSelectionEvent
        # Define the event that will be called when the run button is processed
        self._runbutton.value = self.__runEvent
        # Define the event called before showing the image in the player
        self._player.processFrame = self.__processFrame

        # Define the organization of the Form Controls
        self._formset = [
            ('_videofile', '_outputfile'),
            '_threshold',
            ('_blobsize', '_runbutton'),
            '_player'
        ]

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

    def __runEvent(self):
        """
        After setting the best parameters run the full algorithm
        """
        self._player.value = cv2.VideoCapture(self._videofile.value)
        print (self._player.video_index)
        print (self._player.max)
        print (self._player.fps)
        self._player.play()
        pass

#Execute the application
if __name__ == "__main__":   pyforms.start_app( ComputerVisionAlgorithm )