import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPainter
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
import numpy as np 



class ColorElement:
    def __init__(self, center, radius, color):
        self.x = center[0]
        self.y = center[1]
        self.radius = radius
        self.color = color
        self.pressed = False

    def drawColor(self, painter):
        painter.setPen(QtGui.QColor(self.color[0], self.color[1], self.color[2]))
        for x in range(self.x - self.radius, self.x + self.radius):
            for y in range(self.y - self.radius, self.y + self.radius):
                if self.inCircle((x, y)):
                    painter.drawPoint(x, y)

    def inCircle(self, point):
        center = np.array([self.x, self.y])
        test_pt = np.array([point[0], point[1]])
        return np.linalg.norm(center - test_pt) <= self.radius

    def eventWithinShape(self, event):
        return self.inCircle((event.pos().x(), event.pos().y()))


class ColorRegion:
    def __init__(self):
        self.blob1 = ColorElement((45, 45), 10, (255,255,255)) 
        self.blob2 = ColorElement((45, 95), 10, (245, 173, 66))
        self.blob3 = ColorElement((75, 30), 20, (1.0, 1.0, 1.0))
        self.blob_lst = [self.blob1, self.blob2, self.blob3]
        self.bbox = self.calcBBox()

    def drawRegion(self, painter):
        self.bbox = self.calcBBox()
        blob_centers = [np.array([c.x, c.y]) for c in self.blob_lst] #ASSUME 2 Centers for now 

        for x in range(self.bbox[0], self.bbox[2]):
            for y in range(self.bbox[1], self.bbox[3]):
                currPt = np.array([x, y])
                distLst = [np.linalg.norm(currPt - blob_center) for blob_center in blob_centers]
                colorLst = [np.array(blob.color) for blob in self.blob_lst]
                color_res, toRender = self.calcColorAdvanced(distLst, colorLst)
                if toRender:
                    painter.setPen(QtGui.QColor(color_res[0], color_res[1], color_res[0]))
                    painter.drawPoint(x, y)


    def calcColor(self, distList, colorList):
        assert len(distList) == len(colorList)
        res_color = np.array([0.0, 0.0, 0.0])
        for i in range(len(distList)):
            res_color += colorList[i] * distList[i]
        res_color /= np.sum(distList)
        return res_color

    def calcColorAdvanced(self, distList, colorList):
        b = 40
        assert len(distList) == len(colorList)
        res_color = np.array([0.0, 0.0, 0.0])
        color = np.array([0.0, 0.0, 0.0])
        influence_sum = 0
        for i in range(len(distList)):
            d = distList[i]
            if d <= b:
                influence = 1 - 4 * pow(d, 6) / (9 * pow(b, 6)) + 17 * pow(d, 4) / (9 * pow(b, 4)) - 22 * pow(d, 2) / (9 * pow(b,2))
            else:
                influence = 0
            color += influence * colorList[i]
            influence_sum += influence

        render = False 
        if influence_sum > 0.4: 
            res_color = color / influence_sum
            render = True 
        return res_color, render 


    def calcBBox(self):
        xCoords = [b.x - b.radius for b in self.blob_lst] + [b.x + b.radius for b in self.blob_lst]
        yCoords = [b.y - b.radius for b in self.blob_lst] + [b.y + b.radius for b in self.blob_lst]
        minX = int(min(xCoords)* 0.5)
        minY = int(min(yCoords)* 0.5)
        maxX = int(max(xCoords) * 1.5)
        maxY = int(max(yCoords) * 1.5)
        return (minX, minY, maxX, maxY)


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(30,30,300,200)
        self.region = ColorRegion()
        self.show()


    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        self.region.drawRegion(qp)


    def mousePressEvent(self, event):
        for c in self.region.blob_lst:
            if c.eventWithinShape(event):
                c.pressed = True 
        self.update()

    def mouseMoveEvent(self, event):
        for c in self.region.blob_lst:
            if c.eventWithinShape(event) and c.pressed:
                c.x = event.pos().x()
                c.y = event.pos().y()
        self.update()

    def mouseReleaseEvent(self, event):
        for c in self.region.blob_lst:
            c.pressed = False
        self.update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWidget()
    window.show()

    app.aboutToQuit.connect(app.deleteLater)
    sys.exit(app.exec_())