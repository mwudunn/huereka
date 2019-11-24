from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5 import QtCore, QtGui, QtWidgets

class ExampleQGLWidget(QtWidgets.QOpenGLWidget):

    def buildShaders(self):
        self.shaderProgram = QtGui.QOpenGLShaderProgram()

        vertex = """
        void main(void)
        {
            gl_Position = ftransform();
        }
        """

        fragment = """
        void main(void)
        {
            gl_FragColor = vec4(1.0,0.0,0.0,1.0);
        }
        """

        self.shaderProgram.addShaderFromSourceCode(QtGui.QOpenGLShader.Vertex, vertex)
        self.shaderProgram.addShaderFromSourceCode(QtGui.QOpenGLShader.Fragment, fragment)
        self.shaderProgram.link()


    def __init__(self, parent):
        QtWidgets.QOpenGLWidget.__init__(self, parent)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.shaderProgram.bind()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

    def initializeGL(self):
        glViewport(0,0, 640, 480)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        self.buildShaders()

class TestContainer(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        widget = ExampleQGLWidget(self)
        self.setCentralWidget(widget)

if __name__ == '__main__':
    app = QtWidgets.QApplication(['Shader Example'])
    window = TestContainer()
    window.show()
    app.exec_()