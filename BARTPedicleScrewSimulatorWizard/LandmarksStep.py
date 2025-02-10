import qt, ctk, vtk, slicer

from .PedicleScrewSimulatorStep import *
from .Helper import *
import PythonQt
import os

class LandmarksStep( PedicleScrewSimulatorStep ):

    def __init__( self, stepid ):
      self.initialize( stepid )
      self.setName( '3. Identify Insertion Landmarks' )
      self.setDescription( 'Place at least one fiducial on the spine to mark a screw insertion point.' )

      self.__parent = super( LandmarksStep, self )
      qt.QTimer.singleShot(0, self.killButton)
      self.levels = ("L1", "L2", "L3", "L4", "L5")
      self.startCount = 0
      self.addCount = 0
      self.fiducialNodeObservations = []
      self.fidMoveObserverTag = None


    def killButton(self):
      # hide useless button
      bl = slicer.util.findChildren(text='Final')
      if len(bl):
        bl[0].hide()

    def begin(self):
      # TODO: we could prepare placement mode here
      pass

    def stop(self):
      self.startMeasurements.placeModeEnabled = False

    def cameraFocus(self, position):
      camera = slicer.mrmlScene.GetNodeByID('vtkMRMLCameraNode1')

      if self.approach == 'Posterior':

          camera.SetFocalPoint(*position)
          camera.SetPosition(position[0],-200,position[2])
          camera.SetViewUp([0,0,1])

      elif self.approach == 'Anterior':

          camera.SetFocalPoint(*position)
          camera.SetPosition(position[0],200,position[2])
          camera.SetViewUp([0,0,1])

      elif self.approach == 'Left':

          camera.SetFocalPoint(*position)
          camera.SetPosition(-200,position[1],position[2])
          camera.SetViewUp([0,0,1])

      elif self.approach == 'Right':

          camera.SetFocalPoint(*position)
          camera.SetPosition(200,position[1],position[2])
          camera.SetViewUp([0,0,1])

      camera.ResetClippingRange()

    def onTableCellClicked(self):
      if self.table2.currentColumn() == 0:
          logging.debug(self.table2.currentRow())
          currentFid = self.table2.currentRow()
          position = [0,0,0]
          self.fiducial = self.fiducialNode()
          self.fiducial.GetNthControlPointPosition(currentFid,position)
          logging.debug(position)
          # self.cameraFocus(position)


    def updateTable(self):
        self.fiducial = self.fiducialNode()
        self.fidNumber = self.fiducial.GetNumberOfDefinedControlPoints()
        self.table2.setRowCount(self.fidNumber)

        for i in range(self.fidNumber):
            fidLabel = self.fiducial.GetNthControlPointLabel(i)

            # Create the label in column 0
            labelItem = qt.QTableWidgetItem(fidLabel)
            self.table2.setItem(i, 0, labelItem)

            # Create combo for Level
            comboLevel = qt.QComboBox()
            comboLevel.addItems(self.levelselection)
            attrLevel = self.fiducial.GetAttribute(f"{fidLabel}_Level")
            if attrLevel in self.levelselection:
                comboLevel.setCurrentText(attrLevel)
            # Connect
            comboLevel.connect("currentIndexChanged(int)", lambda idx, fi=i: self.onLevelChanged(idx, fi))

            # Create combo for Side
            comboSide = qt.QComboBox()
            comboSide.addItems(["Left", "Right"])
            attrSide = self.fiducial.GetAttribute(f"{fidLabel}_Side")
            if attrSide in ["Left", "Right"]:
                comboSide.setCurrentText(attrSide)
            # Connect
            comboSide.connect("currentIndexChanged(int)", lambda idx, fi=i: self.onSideChanged(idx, fi))

            # Place them in the table
            self.table2.setCellWidget(i, 1, comboLevel)
            self.table2.setCellWidget(i, 2, comboSide)

    def onLevelChanged(self, index, fidIndex):
        fidLabel = self.fiducial.GetNthControlPointLabel(fidIndex)
        chosenText = self.table2.cellWidget(fidIndex, 1).currentText
        self.fiducial.SetAttribute(f"{fidLabel}_Level", chosenText)

    def onSideChanged(self, index, fidIndex):
        fidLabel = self.fiducial.GetNthControlPointLabel(fidIndex)
        chosenText = self.table2.cellWidget(fidIndex, 2).currentText
        self.fiducial.SetAttribute(f"{fidLabel}_Side", chosenText)

    def deleteFiducial(self):
        if self.table2.currentColumn() == 0:
            item = self.table2.currentItem()
            if not item:
                return  # Exit if no item is selected

            self.fidNumber = self.fiducial.GetNumberOfDefinedControlPoints()
            self.fiducial = self.fiducialNode()

            deleteIndex = -1
            for i in range(self.fidNumber):
                label = self.fiducial.GetNthControlPointLabel(i)
                if label and item.text() == label:
                    deleteIndex = i
                    break

            if deleteIndex != -1:
                # Remove the corresponding fiducial
                self.fiducial.RemoveNthControlPoint(deleteIndex)

            # Remove the corresponding row from the table
            row = self.table2.currentRow()
            if row != -1:
                self.table2.removeRow(row)

    def lockFiducials(self):
      fidNode = self.fiducialNode()
      slicer.modules.markups.logic().SetAllControlPointsLocked(fidNode,True)

    def addFiducials(self):
      pass

    def addFiducialToTable(self, observer, event):
      self.updateTable()

    def createUserInterface( self ):
      markup = slicer.modules.markups.logic()
      markup.AddNewFiducialNode()
      # Create the main layout
      self.__layout = self.__parent.createUserInterface()

      # Create the markups place widget
      self.startMeasurements = slicer.qSlicerMarkupsPlaceWidget()
      self.startMeasurements.setButtonsVisible(False)
      self.startMeasurements.placeButton().show()
      self.startMeasurements.setMRMLScene(slicer.mrmlScene)
      self.startMeasurements.placeMultipleMarkups = slicer.qSlicerMarkupsPlaceWidget.ForcePlaceMultipleMarkups

      # Style the button with dynamic colors based on active mode
      def updateButtonStyle(isActive):
          if isActive:
              # Active = Green
              self.startMeasurements.placeButton().setStyleSheet(
                  "background-color: green;"
              )
          else:
              # Inactive = Red
              self.startMeasurements.placeButton().setStyleSheet(
                  "background-color: red;"
              )

      # Initialize the button style to inactive (red)
      updateButtonStyle(False)

      # Connect to the same signal that indicates a change in the active placing mode
      self.startMeasurements.connect('activeMarkupsFiducialPlaceModeChanged(bool)', updateButtonStyle)
      self.startMeasurements.connect('activeMarkupsFiducialPlaceModeChanged(bool)', self.addFiducials)

      buttonLayout = qt.QHBoxLayout()
      buttonLayout.addWidget(self.startMeasurements)
      self.__layout.addRow(buttonLayout)

      # Fiducial Table
      self.table2 = qt.QTableWidget()
      self.table2.setRowCount(1)
      self.table2.setColumnCount(3)
      self.table2.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
      self.table2.setSizePolicy(qt.QSizePolicy.MinimumExpanding, qt.QSizePolicy.Preferred)
      self.table2.setMinimumWidth(400)
      self.table2.setMinimumHeight(215)
      self.table2.setMaximumHeight(215)
      self.table2.setStyleSheet(
          "QTableWidget {"
          "background-color: #2C2C2C; border: 1px solid #DDDDDD; font-size: 13px; padding: 5px;"
          "}"
          "QHeaderView::section {"
          "background-color: #1484CD; color: white; font-weight: bold; border: 1px solid #DDDDDD;"
          "}"
      )
      horizontalHeaders = ["Fiducial", "Level", "Side"]
      self.table2.setHorizontalHeaderLabels(horizontalHeaders)
      self.table2.itemSelectionChanged.connect(self.onTableCellClicked)
      self.__layout.addWidget(self.table2)

      self.deleteFid = qt.QPushButton("Remove Selected Fiducial")
      self.deleteFid.connect('clicked(bool)', self.deleteFiducial)
      self.__layout.addWidget(self.deleteFid)

      # Camera Transform Sliders

      transCam = ctk.ctkCollapsibleButton()
      transCam.text = "Shift Camera Position"
      transCam.collapsed = True
      self.__layout.addWidget(transCam)
      #transCam.collapsed = True
      camLayout = qt.QFormLayout(transCam)

      a = PythonQt.qMRMLWidgets.qMRMLTransformSliders()
      a.setMRMLTransformNode(slicer.mrmlScene.GetNodeByID('vtkMRMLLinearTransformNode4'))
      #transWidget = slicer.modules.transforms.createNewWidgetRepresentation()
      #transSelector = transWidget.findChild('qMRMLNodeComboBox')
      #transWidgetPart = transWidget.findChild('ctkCollapsibleButton')
      #transformSliders = transWidgetPart.findChildren('qMRMLTransformSliders')
      camLayout.addRow(a)


      qt.QTimer.singleShot(0, self.killButton)


    def onEntry(self, comingFrom, transitionType):

      super(LandmarksStep, self).onEntry(comingFrom, transitionType)

      qt.QTimer.singleShot(0, self.killButton)

      lm = slicer.app.layoutManager()
      if lm == None:
        return
      lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)

      pNode = self.parameterNode()
      logging.debug(pNode)
      self.levelselection = ["L1", "L2", "L3", "L4", "L5"]
      # self.vertebra = str(pNode.GetParameter('vertebra'))
      # self.inst_length = str(pNode.GetParameter('inst_length'))
      # self.approach = str(pNode.GetParameter('approach'))
      # for i in range(self.levels.index(self.vertebra),self.levels.index(self.vertebra)+int(self.inst_length)):
      #     #logging.debug(self.levels[i])
      #     self.levelselection.append(self.levels[i])
      # logging.debug(self.levelselection)

      camera = slicer.mrmlScene.GetNodeByID('vtkMRMLCameraNode1')
      camera.SetPosition(0, -600, 0)
      camera.SetViewUp([0, 0, 1])
      # if self.approach == 'Posterior':
      #     logging.debug("posterior")
      #     camera.SetPosition(0,-600,0)
      #     camera.SetViewUp([0,0,1])
      # elif self.approach == 'Anterior':
      #     logging.debug("Anterior")
      #     camera.SetPosition(0,600,0)
      #     camera.SetViewUp([0,0,1])
      # elif self.approach == 'Left':
      #     logging.debug("Left")
      #     camera.SetPosition(-600,0,0)
      #     camera.SetViewUp([0,0,1])
      # elif self.approach == 'Right':
      #     logging.debug("Right")
      #     camera.SetPosition(600,0,0)
      #     camera.SetViewUp([0,0,1])
      camera.ResetClippingRange()
      #pNode = self.parameterNode()
      #pNode.SetParameter('currentStep', self.stepid)

      fiducialNode = self.fiducialNode()
      self.startMeasurements.setCurrentNode(fiducialNode)
      self.fiducialNodeObservations.append(fiducialNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.addFiducialToTable))
      self.fiducialNodeObservations.append(fiducialNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionUndefinedEvent, self.addFiducialToTable))

      if comingFrom.id() == 'Segmentation':
          self.updateTable()

    def getLandmarksNode(self):
      return self.startMeasurements.currentNode()

    def onExit(self, goingTo, transitionType):

      if goingTo.id() == 'Measurements' or goingTo.id() == 'Segmentation':
          self.stop()
          fiducialNode = self.fiducialNode()
          for observation in self.fiducialNodeObservations:
            fiducialNode.RemoveObserver(observation)
          self.fiducialNodeObservations = []
          self.doStepProcessing()
          #logging.debug(self.table2.cellWidget(0,1).currentText)

      #if goingTo.id() == 'Threshold':
          #slicer.mrmlScene.RemoveNode(self.__outModel)

      if goingTo.id() != 'Segmentation' and goingTo.id() != 'Measurements':
          return

      super(LandmarksStep, self).onExit(goingTo, transitionType)

    def validate( self, desiredBranchId ):

      self.__parent.validate( desiredBranchId )
      self.__parent.validationSucceeded(desiredBranchId)

      #self.inputFiducialsNodeSelector.update()
      #fid = self.inputFiducialsNodeSelector.currentNode()
      fidNumber = self.fiducial.GetNumberOfDefinedControlPoints()

      #pNode = self.parameterNode()
      if fidNumber != 0:
      #  fidID = fid.GetID()
      #  if fidID != '':
      #    pNode = self.parameterNode()
      #    pNode.SetNodeReferenceID('fiducial', fidID)
          self.__parent.validationSucceeded(desiredBranchId)
      else:
          self.__parent.validationFailed(desiredBranchId, 'Error','Please place at least one fiducial on the model before proceeding')

    def doStepProcessing(self):
      #list = ['a','b','c']
      #listNode = self.parameterNode()
      #listNode.SetParameter = ('list', list)
      logging.debug('Done')
      self.lockFiducials()
