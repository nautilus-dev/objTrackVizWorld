from pyforms 			import BaseWidget
from pyforms.Controls 	import *
import pyforms

import cv2
import iA
import detection
import os
import copy


class objectDetectionGUI(BaseWidget):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    definedClasses = []
    annotationText = "Set Bounding Box for item from Class: "
    bb = (0, 0, 0, 0)
    bb_selected = False
    bb_frame = None
    bb_selection_window_X = 0
    bb_selection_window_Y = 0


    def __init__(self):
        super(objectDetectionGUI, self).__init__('Eyetracking object detection with Deep Neural Networks')

        # Definition of the forms fields
        self._videofile = ControlFile('Video')
        self._outputfile = ControlFile('Results output file')
        self._player = ControlPlayer('Player')
        self._loadbutton = ControlButton('Load Video')
        self._annotatebutton = ControlButton('Mark Objects')
        self._runbutton = ControlButton('Run')

        self._monitorX = ControlNumber('Resolution of Monitor X', default=1920, minimum=640, maximum=4000)
        self._monitorY = ControlNumber('Resolution of Monitor Y', default=1080, minimum=480, maximum=4000)

        # Classes
        self._classBackground = ControlButton('background')
        self._classAeroplane = ControlButton('aeroplane')
        self._classBicycle = ControlButton('bicycle')
        self._classBird = ControlButton('bird')
        self._classBoat = ControlButton('boat')
        self._classBottle = ControlButton('bottle')
        self._classBus = ControlButton('bus')
        self._classCar = ControlButton('car')
        self._classCat = ControlButton('cat')
        self._classChair = ControlButton('chair')
        self._classCow = ControlButton('cow')
        self._classDiningTable = ControlButton('diningtable')
        self._classDog = ControlButton('dog')
        self._classHorse = ControlButton('horse')
        self._classMotorbike = ControlButton('motorbike')
        self._classPerson = ControlButton('person')
        self._classPottedPlant = ControlButton('pottedplant')
        self._classSheep = ControlButton('sheep')
        self._classSofa = ControlButton('sofa')
        self._classTrain = ControlButton('train')
        self._classTvmonitor = ControlButton('tvmonitor')

        self._classBackground.value = self.__getClassValuesBackground
        self._classAeroplane.value = self.__getClassValuesAeroplane
        self._classBicycle.value = self.__getClassValuesBicycle
        self._classBird.value = self.__getClassValuesBird
        self._classBoat.value = self.__getClassValuesBoat
        self._classBottle.value = self.__getClassValuesBottle
        self._classBus.value = self.__getClassValuesBus
        self._classCar.value = self.__getClassValuesCar
        self._classCat.value = self.__getClassValuesCat
        self._classChair.value = self.__getClassValuesChair
        self._classCow.value = self.__getClassValuesCow
        self._classDiningTable.value = self.__getClassValuesDiningtable
        self._classDog.value = self.__getClassValuesDog
        self._classHorse.value = self.__getClassValuesHorse
        self._classMotorbike.value = self.__getClassValuesMotorbike
        self._classPerson.value = self.__getClassValuesPerson
        self._classPottedPlant.value = self.__getClassValuesPottedplant
        self._classSheep.value = self.__getClassValuesSheep
        self._classSofa.value = self.__getClassValuesSofa
        self._classTrain.value = self.__getClassValuesTrain
        self._classTvmonitor.value = self.__getClassValuesTvmonitor

        # Define the function that will be called when a file is selected ==> Nothing happening for windows here...
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
            ('_loadbutton', '_runbutton', '_monitorX', '_monitorY',),
            ([' ', '_classBackground', '_classAeroplane', '_classBicycle', '_classBird', '_classBoat', '_classBottle',
              '_classBus', '_classCar', '_classCat', '_classChair', '_classCow', '_classDiningTable', '_classDog',
              '_classMotorbike', '_classPerson', '_classPottedPlant', '_classSheep', '_classSofa', '_classTrain',
              '_classTvmonitor', ' ', ' '], '_player' )
        ]

        # Adding all Classes to the selection pane
        # for val in self.CLASSES:
        #    self._objectclasses += (val, False)

    def __getClassValuesBackground(self):
        self.__getClassValues('background')

    def __getClassValuesAeroplane(self):
        self.__getClassValues('aeroplane')

    def __getClassValuesBicycle(self):
        self.__getClassValues('bicycle')

    def __getClassValuesBird(self):
        self.__getClassValues('bird')

    def __getClassValuesBoat(self):
        self.__getClassValues('boat')

    def __getClassValuesBottle(self):
        self.__getClassValues('bottle')

    def __getClassValuesBus(self):
        self.__getClassValues('bus')

    def __getClassValuesCar(self):
        self.__getClassValues('car')

    def __getClassValuesCat(self):
        self.__getClassValues('cat')

    def __getClassValuesChair(self):
        self.__getClassValues('chair')

    def __getClassValuesCow(self):
        self.__getClassValues('cow')

    def __getClassValuesDiningtable(self):
        self.__getClassValues('diningtable')

    def __getClassValuesDog(self):
        self.__getClassValues('dog')

    def __getClassValuesHorse(self):
        self.__getClassValues('horse')

    def __getClassValuesMotorbike(self):
        self.__getClassValues('motorbike')

    def __getClassValuesPerson(self):
        self.__getClassValues('person')

    def __getClassValuesPottedplant(self):
        self.__getClassValues('pottedplant')

    def __getClassValuesSheep(self):
        self.__getClassValues('sheep')

    def __getClassValuesSofa(self):
        self.__getClassValues('sofa')

    def __getClassValuesTrain(self):
        self.__getClassValues('train')

    def __getClassValuesTvmonitor(self):
        self.__getClassValues('tvmonitor')



    def __videoFileSelectionEvent(self):
        """
        When the videofile is selected instantiate the video in the player
        """
        """ self._player.value = self._videofile.value """
        self._player.value = cv2.VideoCapture(self._videofile.value)
        print(self._player.video_index)
        print(self._player.max)
        print(self._player.fps)


    def __processFrame(self, frame):
        """
        Do some processing to the frame and return the result frame
        """

        return frame

    def __loadEvent(self):
        self._player.value = cv2.VideoCapture(self._videofile.value)
        print(self._player.video_index)
        print(self._player.max)
        print(self._player.fps)

    def __getClassValues(self, classname):
        # car3 = iA(900, [[40, 125], [73, 155]], "car")
        # definedClasses
        if self.bb_selected :
            print("adding an instance of : " + classname)
            self.definedClasses.append(iA.iA(
                            self._player.video_index, [[self.bb[0], self.bb[1]], [self.bb[2], self.bb[3]]], classname))
        self.bb_selected = False
        self.bb_frame = None
        pass


    def __runEvent(self):
        """
        After setting the parameters run the full algorithm
        """
        # TODO: Check that all values are set!!!
        # self._player.play()
        detection.xmax = int(self._monitorX.value)
        detection.ymax = int(self._monitorY.value)
        detection.input_path = self._videofile.value
        detection.output_path = self._outputfile.value if (self._outputfile.value != "") else os.getcwd()
        detection.detect(detection.input_path, detection.output_path)
        pass

    def __openBBBox(self):
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        self.bb = cv2.selectROI("Select ROI", self._player.frame)
        cv2.destroyWindow("Select ROI")
        self.bb_selected = True
        pass

    def __returnOpenCVBox(self, event, x, y):
        """
        :return:
        """
        # TODO: Check wheter the video is loaded otherwise the app crashes

        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        self.bb = cv2.selectROI("Select ROI", self._player.frame)
        cv2.destroyWindow("Select ROI")
        self.bb_selected = True
        self.bb_selection_window_X = cv2
        self.bb_frame = cv2.rectangle(self._player.frame, (self.bb[0], self.bb[1]), (self.bb[2], self.bb[3]),
                                      (255, 0, 0), 2)
        self._player.frame = self.bb_frame
        pass

#Execute the application
if __name__ == "__main__":   pyforms.start_app(objectDetectionGUI)

