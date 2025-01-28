import qt, ctk, vtk, slicer

from .PedicleScrewSimulatorStep import *
from .Helper import *
import PythonQt
import math
import os
import time
import logging

class ScrewStep(PedicleScrewSimulatorStep):


    def __init__( self, stepid ):
      self.initialize( stepid )
      self.setName( '5. Place Screws' )
      self.setDescription( 'Load screw models and change orientation using sliders' )
      self.screwPath = None
      self.screwName = None
      self.coords = [0,0,0]
      self.matrix1 = vtk.vtkMatrix3x3()
      self.matrix2 = vtk.vtkMatrix3x3()
      self.matrix3 = vtk.vtkMatrix3x3()
      self.matrixScrew = vtk.vtkMatrix4x4()
      self.fiduciallist = []
      self.screwSummary = []

      self.approach = None

      self.screwList = []
      self.currentFidIndex = 0
      self.currentFidLabel = None

      self.fidNode = slicer.vtkMRMLMarkupsFiducialNode()

      self.valueTemp1 = 0
      self.valueTemp2 = 0
      self.driveTemp = 0

      self.__loadScrewButton = None
      self.__parent = super( ScrewStep, self )

      self.timer = qt.QTimer()
      self.timer.setInterval(2)
      self.timer.connect('timeout()', self.driveScrew)
      self.timer2 = qt.QTimer()
      self.timer2.setInterval(2)
      self.timer2.connect('timeout()', self.reverseScrew)
      self.screwInsert = 0.0


    def killButton(self):
      # hide useless button
      bl = slicer.util.findChildren(text='Final')
      if len(bl):
        bl[0].hide()

    def createUserInterface( self ):

      self.__layout = self.__parent.createUserInterface()

      self.fiducial = ctk.ctkComboBox()
      self.fiducial.toolTip = "Select an insertion site."
      self.fiducial.addItems(self.fiduciallist)
      self.connect(self.fiducial, PythonQt.QtCore.SIGNAL('activated(QString)'), self.fiducial_chosen)

      #self.screwGridLayout.addWidget(self.fiducial,0,0)

      self.__layout.addRow("Insertion Site:", self.fiducial)
      self.__fiducial = ''
      measuredText1 = qt.QLabel("     Measured:")
      measuredText2 = qt.QLabel("     Measured:")
      lengthText = qt.QLabel("Screw Length:   ")
      widthText = qt.QLabel("Screw Width:    ")
      self.length = ctk.ctkComboBox()
      self.length.toolTip = "Select a screw to insert."
      screwList = ['Select a length (mm)','47.5', '55.0','62.5','70.0']
      self.length.addItems(screwList)
      self.connect(self.length, PythonQt.QtCore.SIGNAL('activated(QString)'), self.length_chosen)
      self.lengthMeasure = qt.QLineEdit()
      self.__length = ''

      self.QHBox1 = qt.QHBoxLayout()
      self.QHBox1.addWidget(lengthText)
      self.QHBox1.addWidget(self.length)
      self.QHBox1.addWidget(measuredText1)
      self.QHBox1.addWidget(self.lengthMeasure)
      self.__layout.addRow(self.QHBox1)

      self.diameter = ctk.ctkComboBox()
      self.diameter.toolTip = "Select a screw to insert."
      screwList = ['Select a diameter (mm)','3.0', '3.5', '4.5', '5.0']
      self.diameter.addItems(screwList)
      self.widthMeasure = qt.QLineEdit()
      self.connect(self.diameter, PythonQt.QtCore.SIGNAL('activated(QString)'), self.diameter_chosen)
      self.__diameter = ''

      self.QHBox2 = qt.QHBoxLayout()
      self.QHBox2.addWidget(widthText)
      self.QHBox2.addWidget(self.diameter)
      self.QHBox2.addWidget(measuredText2)
      self.QHBox2.addWidget(self.widthMeasure)
      self.__layout.addRow(self.QHBox2)

      # Load Screw Button
      self.__loadScrewButton = qt.QPushButton("Load Screw")
      self.__loadScrewButton.enabled = False
      self.__loadScrewButton.setStyleSheet("background-color: green;")
      #self.__layout.addWidget(self.__loadScrewButton)
      self.__loadScrewButton.connect('clicked(bool)', self.loadScrew)

      # Delete Screw Button
      self.__delScrewButton = qt.QPushButton("Delete Screw")
      self.__delScrewButton.enabled = True
      self.__delScrewButton.setStyleSheet("background-color: red;")
      #self.__layout.addWidget(self.__delScrewButton)
      self.__delScrewButton.connect('clicked(bool)', self.delScrew)

      self.QHBox3 = qt.QHBoxLayout()
      self.QHBox3.addWidget(self.__loadScrewButton)
      self.QHBox3.addWidget(self.__delScrewButton)
      self.__layout.addRow(self.QHBox3)

      # Input model node selector
      self.modelNodeSelector = slicer.qMRMLNodeComboBox()
      self.modelNodeSelector.toolTip = "."
      self.modelNodeSelector.nodeTypes = ["vtkMRMLModelNode"]
      self.modelNodeSelector.addEnabled = False
      self.modelNodeSelector.removeEnabled = False
      self.modelNodeSelector.setMRMLScene(slicer.mrmlScene)

      self.transformGrid = qt.QGridLayout()
      vText = qt.QLabel("Vertical Adjustment:")
      iText = qt.QLabel("Horizontal Adjustment:")
      self.transformGrid.addWidget(vText, 0,0)
      self.transformGrid.addWidget(iText, 0,2)

      self.b = ctk.ctkDoubleSpinBox()
      self.b.minimum = -45
      self.b.maximum = 45

      self.transformGrid.addWidget(self.b, 1,0)

      # Transform Sliders
      self.transformSlider1 = ctk.ctkDoubleSlider()
      self.transformSlider1.minimum = -45
      self.transformSlider1.maximum = 45
      self.transformSlider1.connect('valueChanged(double)', self.transformSlider1ValueChanged)
      self.transformSlider1.connect('valueChanged(double)', self.b.setValue)
      self.transformSlider1.setMinimumHeight(120)
      #self.__layout.addRow("Rotate IS", self.transformSlider1)
      self.transformGrid.addWidget(self.transformSlider1, 1,1)

      self.b.connect('valueChanged(double)', self.transformSlider1.setValue)

      # Transform Sliders
      self.transformSlider2 = ctk.ctkSliderWidget()
      self.transformSlider2.minimum = -45
      self.transformSlider2.maximum = 45
      self.transformSlider2.connect('valueChanged(double)', self.transformSlider2ValueChanged)
      self.transformSlider2.setMinimumHeight(120)
      #self.__layout.addRow("Rotate LR", self.transformSlider2)
      self.transformGrid.addWidget(self.transformSlider2, 1,2)
      self.__layout.addRow(self.transformGrid)

      # Insert Screw Button
      self.insertScrewButton = qt.QPushButton("Insert Screw")
      self.insertScrewButton.enabled = True
      self.insertScrewButton.setStyleSheet("background-color: green;")
      #self.__layout.addWidget(self.__loadScrewButton)
      self.insertScrewButton.connect('clicked(bool)', self.insertScrew)

      # Backout Screw Button
      self.backoutScrewButton = qt.QPushButton("Backout Screw")
      self.backoutScrewButton.enabled = False
      self.backoutScrewButton.setStyleSheet("background-color: red;")
      #self.__layout.addWidget(self.__delScrewButton)
      self.backoutScrewButton.connect('clicked(bool)', self.backoutScrew)

      # Reset Screw Button
      self.resetScrewButton = qt.QPushButton("Reset Screw")
      self.resetScrewButton.enabled = True
      self.resetScrewButton.setStyleSheet("background-color: blue;")
      #self.__layout.addWidget(self.__delScrewButton)
      self.resetScrewButton.connect('clicked(bool)', self.resetScrew)

      self.QHBox4 = qt.QHBoxLayout()
      self.QHBox4.addWidget(self.insertScrewButton)
      self.QHBox4.addWidget(self.backoutScrewButton)
      self.QHBox4.addWidget(self.resetScrewButton)
      self.__layout.addRow(self.QHBox4)

      # Hide ROI Details
      qt.QTimer.singleShot(0, self.killButton)
      self.currentFidIndex = self.fiducial.currentIndex
      self.currentFidLabel = self.fiducial.currentText
      self.fidNode.GetNthControlPointPosition(self.currentFidIndex,self.coords)
      logging.debug("Coords: {0}".format(self.coords))
      self.updateMeasurements()
      self.cameraFocus(self.coords)

    def insertScrew(self):
      logging.debug("insert")
      self.timer.start()
      self.backoutScrewButton.enabled = True
      self.insertScrewButton.enabled = False
      self.b.enabled = False
      self.transformSlider1.enabled = False
      self.transformSlider2.enabled = False

      temp = self.screwList[self.currentFidIndex]
      temp[3] = "1"
      self.screwList[self.currentFidIndex] = temp
      logging.debug("Screw list: {0}".format(self.screwList))

    def backoutScrew(self):
      logging.debug("backout")
      self.timer2.start()
      self.backoutScrewButton.enabled = False
      self.insertScrewButton.enabled = True
      self.b.enabled = True
      self.transformSlider1.enabled = True
      self.transformSlider2.enabled = True

      temp = self.screwList[self.currentFidIndex]
      temp[3] = "0"
      self.screwList[self.currentFidIndex] = temp
      logging.debug("Screw list: {0}".format(self.screwList))

    def resetScrew(self):
      logging.debug("reset")
      self.resetOrientation()

      temp = self.screwList[self.currentFidIndex]
      temp[3] = "0"
      self.screwList[self.currentFidIndex] = temp
      logging.debug("Screw list: {0}".format(self.screwList))

    def updateMeasurements(self):
      pedicleLength = slicer.modules.BART_PlanningWidget.measurementsStep.angleTable.cellWidget(self.currentFidIndex,3).currentText
      pedicleWidth = slicer.modules.BART_PlanningWidget.measurementsStep.angleTable.cellWidget(self.currentFidIndex,4).currentText
      self.lengthMeasure.setText(pedicleLength + " mm")
      self.widthMeasure.setText(pedicleWidth + " mm")
      logging.debug("Pedicle length: {0}".format(pedicleLength))

    def screwLandmarks(self):
      self.fiducial = self.fiducialNode()
      self.fidNumber = self.fiducial.GetNumberOfControlPoints()
      self.fidLabels = []
      self.fidLevels = []
      self.fidSides = []
      self.fidLevelSide = []

      for i in range(0,self.fidNumber):
          self.fidLabels.append(slicer.modules.BART_PlanningWidget.measurementsStep.angleTable.item(i,0).text())
          self.fidLevels.append(slicer.modules.BART_PlanningWidget.measurementsStep.angleTable.cellWidget(i,1).currentText)
          self.fidSides.append(slicer.modules.BART_PlanningWidget.measurementsStep.angleTable.cellWidget(i,2).currentText)
          #self.fidLevelSide.append(self.fidLevels[i] + " " + self.fidSides[i])

      logging.debug("Fid level side: {0}".format(self.fidLevelSide))

    def sliceChange(self):
        pos = [0,0,0]
        if self.fidNode != None:
          self.fidNode.GetNthControlPointPosition(self.currentFidIndex,pos)

          lm = slicer.app.layoutManager()
          redWidget = lm.sliceWidget('Red')
          redController = redWidget.sliceController()

          yellowWidget = lm.sliceWidget('Yellow')
          yellowController = yellowWidget.sliceController()

          greenWidget = lm.sliceWidget('Green')
          greenController = greenWidget.sliceController()

          yellowController.setSliceOffsetValue(pos[0])
          greenController.setSliceOffsetValue(pos[1])
          redController.setSliceOffsetValue(pos[2])
          logging.debug("Position: {0}".format(pos))
          self.fidNode.UpdateScene(slicer.mrmlScene)

        else:
            return

    def fidChanged(self, fid):

        self.fid = fid
        self.valueTemp1 = 0
        self.valueTemp2 = 0
        self.driveTemp = 0

        #self.transformSlider3.reset()

        screwCheck = slicer.mrmlScene.GetFirstNodeByName('Screw at point %s' % self.currentFidLabel)

        if screwCheck == None:
            self.transformSlider1.setValue(0)
            self.transformSlider2.reset()
        else:
            temp = self.screwList[self.currentFidIndex]
            vertOrt = float(temp[4])
            horzOrt = float(temp[5])
            self.resetScrew()
            self.transformSlider1.setValue(vertOrt)
            self.transformSlider2.setValue(horzOrt)

        logging.debug("Length: {0}".format(self.__length))
        self.sliceChange()
        self.updateMeasurements()
        self.cameraFocus(self.coords)

        self.backoutScrewButton.enabled = False
        self.insertScrewButton.enabled = True
        self.b.enabled = True
        self.transformSlider1.enabled = True
        self.transformSlider2.enabled = True

    def fiducial_chosen(self, text):
        if text != "Select an insertion landmark":
            self.__fiducial = text
            self.currentFidIndex = self.fiducial.currentIndex
            self.currentFidLabel = self.fiducial.currentText
            self.fidNode.GetNthControlPointPosition(self.currentFidIndex,self.coords)
            logging.debug("Current fid index = {0}, label = {1}, coords = {2}".format(
              self.currentFidIndex, self.currentFidLabel, self.coords))
            self.updateMeasurements()
            self.combo_chosen()


    def length_chosen(self, text):
        if text != "Select a length (mm)":
            self.__length = text
            self.combo_chosen()

    def diameter_chosen(self, text):
        if text != "Select a diameter (mm)":
            self.__diameter = text
            self.combo_chosen()

    def combo_chosen(self):
        if self.__length != "Select a length (mm)" and self.__diameter != "Select a diameter (mm)":
            # Remove any '.' in length and diameter
            sanitized_length = self.__length.replace('.', '')
            sanitized_diameter = self.__diameter.replace('.', '')

            self.screwPath = os.path.join(
                os.path.dirname(slicer.modules.bart_planning.path),
                f'Resources/ScrewModels/scaled_{sanitized_length}x{sanitized_diameter}.vtk'
            )
            self.screwPath = self.screwPath.replace("\\", "/")
            logging.debug("Screw file path: {0}".format(self.screwPath))
            self.__loadScrewButton.enabled = True
            # self.screwName = 'scaled_' + text
            # self.transformSlider3.maximum = int(self.__diameter)

    def loadScrew(self):
        logging.debug("load screw button")

        screwCheck = slicer.mrmlScene.GetFirstNodeByName('Screw at point %s' % self.currentFidLabel)
        if screwCheck != None:
            # screw already loaded
            return

        screwDescrip = ["0","0","0","0","0","0"]
        screwModel = slicer.modules.models.logic().AddModel(self.screwPath)
        if screwModel is None:
            logging.error("Failed to load screw model: "+self.screwPath)
            return

        matrix = vtk.vtkMatrix4x4()
        matrix.DeepCopy((1, 0, 0, self.coords[0],
                       0, -1, 0, self.coords[1],
                       0, 0, -1, self.coords[2],
                       0, 0, 0, 1))

        transformScrewTemp = slicer.vtkMRMLLinearTransformNode()
        transformScrewTemp.SetName("Transform-%s" % self.currentFidLabel)
        slicer.mrmlScene.AddNode(transformScrewTemp)
        transformScrewTemp.ApplyTransformMatrix(matrix)

        screwModel.SetName('Screw at point %s' % self.currentFidLabel)
        screwModel.SetAndObserveTransformNodeID(transformScrewTemp.GetID())

        self.addLineAlignedWithScrew(screwModel)

        modelDisplay = screwModel.GetDisplayNode()
        modelDisplay.SetColor(0.12,0.73,0.91)
        modelDisplay.SetDiffuse(0.90)
        modelDisplay.SetAmbient(0.10)
        modelDisplay.SetSpecular(0.20)
        modelDisplay.SetPower(10.0)
        modelDisplay.SetVisibility2D(True)
        screwModel.SetAndObserveDisplayNodeID(modelDisplay.GetID())

        screwDescrip[0] = self.currentFidLabel
        screwDescrip[1] = self.__length
        screwDescrip[2] = self.__diameter

        self.screwList.append(screwDescrip)

        self.insertScrewButton.enabled = True
        self.backoutScrewButton.enabled = False
        self.b.enabled = True
        self.transformSlider1.enabled = True
        self.transformSlider2.enabled = True

    def delScrew(self):
        #fidName = self.inputFiducialsNodeSelector.currentNode().GetName()

        transformFid = slicer.mrmlScene.GetFirstNodeByName('Transform-%s' % self.currentFidLabel)
        screwModel = slicer.mrmlScene.GetFirstNodeByName('Screw at point %s' % self.currentFidLabel)

        if screwModel != None:
            slicer.mrmlScene.RemoveNode(transformFid)
            slicer.mrmlScene.RemoveNode(screwModel)
        else:
            return

    def fidMove(self, observer, event):

        screwCheck = slicer.mrmlScene.GetFirstNodeByName('Screw at point %s' % observer.GetName())

        if screwCheck != None:
          coords = [0,0,0]
          observer.GetFiducialCoordinates(coords)

          matrixScrew = vtk.vtkMatrix4x4()
          transformFid = slicer.mrmlScene.GetFirstNodeByName('Transform-%s' % observer.GetName())

          matrixScrew = transformFid.GetMatrixTransformToParent(matrixScrew)
          matrixScrew.SetElement(0,3,coords[0])
          matrixScrew.SetElement(1,3,coords[1])
          matrixScrew.SetElement(2,3,coords[2])
          transformFid.SetMatrixTransformToParent(matrixScrew)

          transformFid.UpdateScene(slicer.mrmlScene)
          self.sliceChange()

          screwModel = slicer.mrmlScene.GetFirstNodeByName(f"Screw at point {self.currentFidLabel}")
          if screwModel:
              self.addLineAlignedWithScrew(screwModel)
        else:
          return

    def addLineAlignedWithScrew(self, screwNode):
        """
        Adds a line that aligns with the screw node.

        Parameters:
            screwNode: vtkMRMLModelNode
                The screw model node loaded in the scene.
        """

        # Remove old line node if present
        existingLineNode = slicer.mrmlScene.GetFirstNodeByName("Screw Alignment Line")
        if existingLineNode:
            slicer.mrmlScene.RemoveNode(existingLineNode)

        # Get screw transform to world
        screwTransformMatrix = vtk.vtkMatrix4x4()
        screwTransformNode = screwNode.GetParentTransformNode()
        if screwTransformNode:
            screwTransformNode.GetMatrixTransformToWorld(screwTransformMatrix)
        else:
            screwTransformMatrix.Identity()

        # Extract position and direction
        screwPosition = [screwTransformMatrix.GetElement(i, 3) for i in range(3)]
        screwDirection = [screwTransformMatrix.GetElement(i, 1) for i in range(3)]

        # Define line endpoints
        lineLength = 100.0
        startPoint = [screwPosition[i] - lineLength * screwDirection[i] for i in range(3)]
        endPoint = [screwPosition[i] + lineLength * screwDirection[i] for i in range(3)]

        # Create a line source
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(startPoint)
        lineSource.SetPoint2(endPoint)
        lineSource.Update()

        # Create model node
        lineModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Screw Alignment Line")
        lineModelNode.SetAndObservePolyData(lineSource.GetOutput())

        # Create display node
        lineDisplayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        lineDisplayNode.SetColor(1, 0, 0)  # red
        lineDisplayNode.SetLineWidth(3)
        lineDisplayNode.SetVisibility2D(True)
        # Key step: show full line in slice
        lineDisplayNode.SetSliceDisplayModeToProjection()

        # Link display node to model node
        lineModelNode.SetAndObserveDisplayNodeID(lineDisplayNode.GetID())

    def cameraFocus(self, position):
      camera = slicer.mrmlScene.GetNodeByID('vtkMRMLCameraNode1')

      if self.approach == 'Posterior':

          camera.SetFocalPoint(*position)
          camera.SetPosition(position[0],-400,position[2])
          camera.SetViewUp([0,0,1])

      elif self.approach == 'Anterior':

          camera.SetFocalPoint(*position)
          camera.SetPosition(position[0],400,position[2])
          camera.SetViewUp([0,0,1])

      elif self.approach == 'Left':

          camera.SetFocalPoint(*position)
          camera.SetPosition(-400,position[1],position[2])
          camera.SetViewUp([0,0,1])

      elif self.approach == 'Right':

          camera.SetFocalPoint(*position)
          camera.SetPosition(400,position[1],position[2])
          camera.SetViewUp([0,0,1])

      camera.ResetClippingRange()

    def transformSlider1ValueChanged(self, value):
        logging.debug("Transform slider 1 changed: {0}".format(value))

        newValue = value - self.valueTemp1

        angle1 = math.pi / 180.0 * newValue * -1 # Match screw direction

        matrix1 = vtk.vtkMatrix3x3()
        matrix1.DeepCopy([ 1, 0, 0,
                          0, math.cos(angle1), -math.sin(angle1),
                          0, math.sin(angle1), math.cos(angle1)])

        self.transformScrewComposite(matrix1)

        self.valueTemp1 = value

        temp = self.screwList[self.currentFidIndex]
        temp[4] = str(value)
        self.screwList[self.currentFidIndex] = temp
        logging.debug("Screw list: {0}".format(self.screwList))

    def transformSlider2ValueChanged(self, value):
        logging.debug("Transform slider 2 changed: {0}".format(value))

        newValue = value - self.valueTemp2

        angle2 = math.pi / 180.0 * newValue * -1 # Match screw direction

        matrix2 = vtk.vtkMatrix3x3()
        matrix2.DeepCopy([ math.cos(angle2), -math.sin(angle2), 0,
                          math.sin(angle2), math.cos(angle2), 0,
                          0, 0, 1])

        self.transformScrewComposite(matrix2)

        self.valueTemp2 = value

        temp = self.screwList[self.currentFidIndex]
        temp[5] = str(value)
        self.screwList[self.currentFidIndex] = temp
        logging.debug("Screw list: {0}".format(self.screwList))

    def delayDisplay(self,action, msec=1000):
      """This utility method displays a small dialog and waits.
      This does two things: 1) it lets the event loop catch up
      to the state of the test so that rendering and widget updates
      have all taken place before the test continues and 2) it
      shows the user/developer/tester the state of the test
      so that we'll know when it breaks.
      """
      #logging.info(message)
      #self.info = qt.QDialog()
      #self.infoLayout = qt.QVBoxLayout()
      #self.info.setLayout(self.infoLayout)
      #self.label = qt.QLabel(message,self.info)
      #self.infoLayout.addWidget(self.label)
      #qt.QTimer.singleShot(msec, action)
      #self.info.exec_()
      pass


    def driveScrew(self):
        sanitized_length = self.__length.replace('.', '')
        sanitized_diameter = self.__diameter.replace('.', '')
        if self.screwInsert < int(sanitized_diameter):

            value = self.screwInsert
            # attempt to rotate with driving

            angle3 = math.radians(72) #((360/2.5)*self.screwInsert)

            matrix3 = vtk.vtkMatrix3x3()
            matrix3.DeepCopy([ math.cos(angle3), 0, -math.sin(angle3),
                          0, 1, 0,
                          math.sin(angle3), 0, math.cos(angle3)])

            self.transformScrewComposite(matrix3)

            # Reverse direction for translation
            value = -value
            transformFid = slicer.mrmlScene.GetFirstNodeByName(f'Transform-{self.currentFidLabel}')
            if transformFid is not None:
                # Get the current transformation matrix
                matrixScrew = vtk.vtkMatrix4x4()
                transformFid.GetMatrixTransformToParent(matrixScrew)

                # Calculate new translation
                newVal = value - self.driveTemp
                drive1 = matrixScrew.GetElement(0, 1)
                drive2 = matrixScrew.GetElement(1, 1)
                drive3 = matrixScrew.GetElement(2, 1)

                coord1 = drive1 * newVal + matrixScrew.GetElement(0, 3)
                coord2 = drive2 * newVal + matrixScrew.GetElement(1, 3)
                coord3 = drive3 * newVal + matrixScrew.GetElement(2, 3)

                # Update translation in the matrix
                matrixScrew.SetElement(0, 3, coord1)
                matrixScrew.SetElement(1, 3, coord2)
                matrixScrew.SetElement(2, 3, coord3)

                # Apply the updated transformation matrix
                transformFid.SetMatrixTransformToParent(matrixScrew)

                # Notify Slicer of changes
                slicer.mrmlScene.Modified()

                # Update state variables
                self.driveTemp = value
                self.screwInsert += 1
            else:
                slicer.util.errorDisplay(f"Transform node 'Transform-{self.currentFidLabel}' not found.")
        else:
            # Stop the timer and reset variables
            self.timer.stop()
            self.screwInsert = 0.0
            self.driveTemp = 0

    def reverseScrew(self):
        sanitized_length = self.__length.replace('.', '')
        sanitized_diameter = self.__diameter.replace('.', '')
        if self.screwInsert < int(sanitized_diameter):

            value = self.screwInsert
            # Calculate the reverse rotation angle (convert degrees to radians, negative for reverse)
            angle3 = math.radians(-72)  # Adjust rotation angle as needed

            # Create a 4x4 rotation matrix
            matrix3 = vtk.vtkMatrix4x4()
            matrix3.Identity()  # Initialize to identity matrix
            matrix3.SetElement(0, 0, math.cos(angle3))
            matrix3.SetElement(0, 2, -math.sin(angle3))
            matrix3.SetElement(2, 0, math.sin(angle3))
            matrix3.SetElement(2, 2, math.cos(angle3))

            # Apply reverse rotation transformation
            self.transformScrewComposite(matrix3)

            # Get the transform node for the screw
            transformFid = slicer.mrmlScene.GetFirstNodeByName(f'Transform-{self.currentFidLabel}')
            if transformFid is not None:
                # Retrieve the current transformation matrix
                matrixScrew = vtk.vtkMatrix4x4()
                transformFid.GetMatrixTransformToParent(matrixScrew)

                # Calculate translation in reverse direction
                # Calculate translation in reverse direction
                newVal = value - self.driveTemp

                drive1 = matrixScrew.GetElement(0, 1)
                drive2 = matrixScrew.GetElement(1, 1)
                drive3 = matrixScrew.GetElement(2, 1)

                # Log intermediate values for debugging
                logging.debug(f"newVal: {newVal}, value: {value}, driveTemp: {self.driveTemp}")

                # Normalize the direction vector
                direction_length = math.sqrt(drive1 ** 2 + drive2 ** 2 + drive3 ** 2)
                if direction_length > 0:
                    drive1 /= direction_length
                    drive2 /= direction_length
                    drive3 /= direction_length

                # Compute new translation coordinates
                coord1 = drive1 * newVal + matrixScrew.GetElement(0, 3)
                coord2 = drive2 * newVal + matrixScrew.GetElement(1, 3)
                coord3 = drive3 * newVal + matrixScrew.GetElement(2, 3)

                # Update the matrix
                matrixScrew.SetElement(0, 3, coord1)
                matrixScrew.SetElement(1, 3, coord2)
                matrixScrew.SetElement(2, 3, coord3)

                # Apply the updated transformation matrix
                transformFid.SetMatrixTransformToParent(matrixScrew)

                # Notify Slicer of changes
                slicer.mrmlScene.Modified()

                # Update state variables
                self.driveTemp = value
                self.screwInsert += 1
        else:
            # Stop the reverse timer and reset variables
            self.timer2.stop()
            self.screwInsert = 0.0
            self.driveTemp = 0

    def resetOrientation(self):
        # Reset sliders to initial values
        self.transformSlider1.setValue(0)
        self.transformSlider2.reset()

        # Retrieve the transform node
        transformFid = slicer.mrmlScene.GetFirstNodeByName(f'Transform-{self.currentFidLabel}')
        if transformFid is not None:
            # Initialize a new transformation matrix
            matrixScrew = vtk.vtkMatrix4x4()

            # Reset the rotation matrix to identity (or desired orientation)
            matrixScrew.Identity()
            matrixScrew.SetElement(0, 0, 1)
            matrixScrew.SetElement(0, 1, 0)
            matrixScrew.SetElement(0, 2, 0)

            matrixScrew.SetElement(1, 0, 0)
            matrixScrew.SetElement(1, 1, -1)  # Adjusted for desired orientation
            matrixScrew.SetElement(1, 2, 0)

            matrixScrew.SetElement(2, 0, 0)
            matrixScrew.SetElement(2, 1, 0)
            matrixScrew.SetElement(2, 2, -1)

            # Set the translation components to stored coordinates
            matrixScrew.SetElement(0, 3, self.coords[0])
            matrixScrew.SetElement(1, 3, self.coords[1])
            matrixScrew.SetElement(2, 3, self.coords[2])

            # Apply the transformation matrix
            transformFid.SetMatrixTransformToParent(matrixScrew)

            # Notify Slicer of scene changes
            slicer.mrmlScene.Modified()

        # Update UI elements
        self.backoutScrewButton.enabled = False
        self.insertScrewButton.enabled = True

        self.transformSlider1.enabled = True
        self.transformSlider2.enabled = True
        self.b.enabled = True
        # Uncomment and reset the third slider if applicable
        # self.transformSlider3.reset()

    def transformScrewComposite(self, inputMatrix):
        # Retrieve the transform node
        transformFid = slicer.mrmlScene.GetFirstNodeByName(f'Transform-{self.currentFidLabel}')
        if transformFid is not None:
            # Get the current transformation matrix
            matrixScrew = vtk.vtkMatrix4x4()
            transformFid.GetMatrixTransformToParent(matrixScrew)
            screwModel = slicer.mrmlScene.GetFirstNodeByName(f"Screw at point {self.currentFidLabel}")
            if screwModel:
                self.addLineAlignedWithScrew(screwModel)

            # Extract the 3x3 rotation matrix from the 4x4 matrix
            currentRotation = vtk.vtkMatrix3x3()
            for i in range(3):
                for j in range(3):
                    currentRotation.SetElement(i, j, matrixScrew.GetElement(i, j))

            # Extract the 3x3 part of the inputMatrix
            inputRotation = vtk.vtkMatrix3x3()
            for i in range(3):
                for j in range(3):
                    inputRotation.SetElement(i, j, inputMatrix.GetElement(i, j))

            # Prepare the output 3x3 matrix for the resulting rotation
            outputRotation = vtk.vtkMatrix3x3()

            # Perform the multiplication of the rotation matrices
            vtk.vtkMatrix3x3.Multiply3x3(currentRotation, inputRotation, outputRotation)

            # Update the 4x4 transformation matrix with the resulting 3x3 rotation
            for i in range(3):
                for j in range(3):
                    matrixScrew.SetElement(i, j, outputRotation.GetElement(i, j))

            # Ensure the translation components remain unchanged
            # (They are already preserved as part of `matrixScrew`.)

            # Ensure the homogeneous coordinate row remains consistent
            matrixScrew.SetElement(3, 0, 0)
            matrixScrew.SetElement(3, 1, 0)
            matrixScrew.SetElement(3, 2, 0)
            matrixScrew.SetElement(3, 3, 1)

            # Apply the updated matrix to the transform node
            transformFid.SetMatrixTransformToParent(matrixScrew)

            # Notify Slicer of changes
            slicer.mrmlScene.Modified()
        else:
            slicer.util.errorDisplay(f"Transform node 'Transform-{self.currentFidLabel}' not found.")

    def validate( self, desiredBranchId ):

      self.__parent.validate( desiredBranchId )
      self.__parent.validationSucceeded(desiredBranchId)


    def onEntry(self, comingFrom, transitionType):

      self.fidNode = self.fiducialNode()
      self.fidNodeObserver = self.fidNode.AddObserver(vtk.vtkCommand.ModifiedEvent,self.fidMove)

      logging.debug("Fiducial node: {0}".format(self.fidNode))

      self.fidNode.SetLocked(1)
      slicer.modules.models.logic().SetAllModelsVisibility(1)

      for x in range (0,self.fidNode.GetNumberOfControlPoints()):
        label = self.fidNode.GetNthControlPointLabel(x)
        level = slicer.modules.BART_PlanningWidget.landmarksStep.table2.cellWidget(x,1).currentText
        side = slicer.modules.BART_PlanningWidget.landmarksStep.table2.cellWidget(x,2).currentText
        self.fiduciallist.append(label + " / " + level + " / " + side)
        #modelX = slicer.mrmlScene.GetNodeByID('vtkMRMLModelDisplayNode' + str(x + 4))
        #modelX.SetSliceIntersectionVisibility(1)

      logging.debug("Fiducial list: {0}".format(self.fiduciallist))

      #self.fiducial.clear()
      #self.fiducial.addItem("Select an insertion site")
      #self.fiducial.addItems(self.fiduciallist)

      super(ScrewStep, self).onEntry(comingFrom, transitionType)

      lm = slicer.app.layoutManager()
      if lm == None:
        return
      lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

      pNode = self.parameterNode()
      pNode.SetParameter('currentStep', self.stepid)
      logging.debug("Current step: {0}".format(pNode.GetParameter('currentStep')))
      self.approach = str(pNode.GetParameter('approach'))

      qt.QTimer.singleShot(0, self.killButton)


    def onExit(self, goingTo, transitionType):

      self.fidNode.RemoveObserver(self.fidNodeObserver)

      if goingTo.id() != 'Grade' and goingTo.id() != 'Measurements':
          return

      if goingTo.id() == 'Measurements':
          '''
          fiducialNode = self.fiducialNode()
          fidCount = fiducialNode.GetNumberOfFiducials()
          for i in range(fidCount):
            fidName = fiducialNode.GetNthFiducialLabel(i)
            screwModel = slicer.mrmlScene.GetFirstNodeByName('Screw at point %s' % fidName)
            slicer.mrmlScene.RemoveNode(screwModel)

          fiducialNode.RemoveAllMarkups()
          '''
          slicer.modules.models.logic().SetAllModelsVisibility(0)

          # modelDisplayNodes = slicer.util.getNodesByClass('vtkMRMLModelDisplayNode')
          # for x in range(0,self.fidNode.GetNumberOfFiducials()):
          #    modelX = slicer.mrmlScene.GetNodeByID( + str(x + 4))
          #    modelX.SetSliceIntersectionVisibility(0)

          self.fidNode.SetLocked(0)

      if goingTo.id() == 'Grade':
        lineModelNode = slicer.util.getNode("Screw Alignment Line*")
        if lineModelNode:
            lineDisplayNode = lineModelNode.GetDisplayNode()
            if lineDisplayNode:
                lineDisplayNode.SetVisibility(False)

        self.doStepProcessing()

      super(ScrewStep, self).onExit(goingTo, transitionType)


    def doStepProcessing(self):

        logging.debug('Done')
