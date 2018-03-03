#!/usr/bin/env python3
# -*- coding: utf-8 -*-

number_of_iA = 0
instances = {}

class iA:
    def __init__(self, trackframe=0, box=[[0,0],[0,0]], label= ""):
        """
        A - ist the areas-value of the bb
        p - is the centerpoint
        rect - are the cooordinates of the actual bb
        detected - gives informations wether the instances was
        already detected in this frame
        number - is just for the writer (each instance has a number)
        """
        self.A = 0
        self.Amax = 0
        self.p = [0, 0]
        self.rect = box
        self.c = label
        self.calibrate()
        self.detected = 0
        self.counter = 0
        self.tf = trackframe
        global number_of_iA
        number_of_iA += 1
        self.number = number_of_iA
        if self.c != "":
                 insert_inst(self.c, self)

    def calA(self):
        s1 =  max(self.rect[0][0], self.rect[1][0]) - min(self.rect[0][0], self.rect[1][0])
        s2 =  max(self.rect[0][1], self.rect[1][1]) - min(self.rect[0][1], self.rect[1][1])
        self.A = s1 * s2

    def calp(self):
        v1 = min(self.rect[0][0], self.rect[1][0])
        v2 = min(self.rect[0][1], self.rect[1][1])
        s1 =  max(self.rect[0][0], self.rect[1][0]) - v1
        s2 =  max(self.rect[0][1], self.rect[1][1]) - v2
        self.p = [v1 + 0.5 * s1, v2 + 0.5 * s2]

    def calibrate(self):
        self.calA()
        if self.A > self.Amax:
            self.Amax = self.A
        self.calp()

    def writer(self, string):
        writedata("interestarea_" + str(self.number), string)


def writedata(name, string):
    """ For windwos
    if not os.path.exists(output_path + "\data"):
        os.makedirs(output_path + "\data")
    if not os.path.exists(output_path + "\data" + "\data_" + d + "_" +t):
        os.makedirs(output_path + "\data" + "\data_" + d + "_" +t)
    obj = open(output_path + "\data" + "\data_" + d + "_" +t + "\" + name + ".txt", "a")
    obj.write(string + "\n")
    obj.close()
    """
    if not os.path.exists(output_path + "/data"):
        os.makedirs(output_path + "/data")
    if not os.path.exists(output_path + "/data" + "/data_" + d + "_" +t):
        os.makedirs(output_path + "/data" + "/data_" + d + "_" +t)
    obj = open(output_path + "/data" + "/data_" + d + "_" +t + "/" + name + ".txt", "a")
    obj.write(string + "\n")
    obj.close()

def insert_inst(klasse, objekt):
    if klasse in instances:
        instances[klasse].append(objekt)
    else:
        instances[klasse] = [objekt]

