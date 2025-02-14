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

    def stop(self):
      self.startMeasurements.placeModeEnabled = False

    def onTableCellClicked(self):
      if self.table2.currentColumn() == 0:
          logging.debug(self.table2.currentRow())
          currentFid = self.table2.currentRow()
          position = [0,0,0]
          self.fiducial = self.fiducialNode()
          self.fiducial.GetNthControlPointPosition(currentFid,position)
          self.zoomIn()
          self.sliceChange()

    def sliceChange(self):
        logging.debug("changing")
        currentFid = self.table2.currentRow()
        coords = [0,0,0]
        if self.fiducial != None:
          self.fiducial.GetNthControlPointPosition(currentFid,coords)
          slicer.modules.markups.logic().JumpSlicesToLocation(coords[0], coords[1], coords[2], True)
        else:
            return

    def zoomIn(self):
      logging.debug("zoom")
      slicer.app.applicationLogic().PropagateVolumeSelection(1)
      # Get the slice logic for Red
      sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
      sliceLogic = sliceWidget.sliceLogic()
      sliceNode = sliceLogic.GetSliceNode()

      # Get current FOV
      fov = sliceNode.GetFieldOfView()

      # Reduce FOV by 20% to zoom in
      newFov = [fov[0] * 0.3, fov[1] * 0.3, fov[2]]
      sliceNode.SetFieldOfView(newFov[0], newFov[1], newFov[2])

      # Update slice matrices to apply changes
      sliceNode.UpdateMatrices()

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

    def getSegmentationMidlineX(self, segNode):
        rasBounds = [0] * 6
        segNode.GetRASBounds(rasBounds)  # [xmin, xmax, ymin, ymax, zmin, zmax]
        centerX = 0.5 * (rasBounds[0] + rasBounds[1])  # average of xmin and xmax
        return centerX

    def addFiducialToTable(self, observer, event):
      self.updateTable()

      # Assume that the new fiducial is the last one added.
      fidNumber = self.fiducial.GetNumberOfDefinedControlPoints()
      if fidNumber == 0:
          return

      fidIndex = fidNumber - 1
      position = [0, 0, 0]
      self.fiducial.GetNthControlPointPosition(fidIndex, position)

      # Get the segmentation node
      try:
          segNode = slicer.util.getNode("Segmentation")
      except Exception as e:
          # logging.error("Segmentation node not found: " + str(e))
          return

      # detect side from the segmentation
      centerX = self.getSegmentationMidlineX(segNode)
      if position[0] < centerX:
          detectedSide = "Left"
      else:
          detectedSide = "Right"

      # Update the side combo box in the table
      sideComboBox = self.table2.cellWidget(fidIndex, 2)
      if sideComboBox:
          sideComboBox.setCurrentText(detectedSide)

      # Store the side as an attribute on the fiducial node
      fidLabel = self.fiducial.GetNthControlPointLabel(fidIndex)
      self.fiducial.SetAttribute(f"{fidLabel}_Side", detectedSide)

      detectedLevel = self.getLumbarLevelAtPosition(position, segNode)
      if detectedLevel:
          # Update the combo box for the level column
          comboLevel = self.table2.cellWidget(fidIndex, 1)
          if comboLevel:
              comboLevel.setCurrentText(detectedLevel)
          # Also store the detected level as an attribute in the fiducial node
          fidLabel = self.fiducial.GetNthControlPointLabel(fidIndex)
          self.fiducial.SetAttribute(f"{fidLabel}_Level", detectedLevel)

    def getLumbarLevelAtPosition(self, position, segNode):
        """
        Given a world (RAS) position and a segmentation node,
        converts the point into voxel coordinates,
        and checks if the point lies inside the segment.
        """
        segmentation = segNode.GetSegmentation()
        segmentIDs = segmentation.GetSegmentIDs()

        for segId in segmentIDs:
            # Create a labelmap node to hold the exported segment
            labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

            success = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segNode,
                [segId],
                labelmapNode
            )
            if not success:
                slicer.mrmlScene.RemoveNode(labelmapNode)
                continue

            # Get the RAS->IJK transform
            rasToIjkMatrix = vtk.vtkMatrix4x4()
            labelmapNode.GetRASToIJKMatrix(rasToIjkMatrix)

            # Convert the RAS point to IJK coordinates
            rasPoint = [position[0], position[1], position[2], 1]
            ijkPoint = [0, 0, 0, 0]
            rasToIjkMatrix.MultiplyPoint(rasPoint, ijkPoint)

            # Round to the nearest voxel index
            ijkIndices = [int(round(ijkPoint[i])) for i in range(3)]

            # Get the actual voxel data from the labelmap node
            labelmapData = labelmapNode.GetImageData()
            if not labelmapData:
                slicer.mrmlScene.RemoveNode(labelmapNode)
                continue

            dims = labelmapData.GetDimensions()
            # Check index bounds
            if (ijkIndices[0] < 0 or ijkIndices[0] >= dims[0] or
                    ijkIndices[1] < 0 or ijkIndices[1] >= dims[1] or
                    ijkIndices[2] < 0 or ijkIndices[2] >= dims[2]):
                slicer.mrmlScene.RemoveNode(labelmapNode)
                continue

            # Check the voxel value at (i, j, k). If > 0, point is inside the segment
            voxelValue = labelmapData.GetScalarComponentAsDouble(
                ijkIndices[0],
                ijkIndices[1],
                ijkIndices[2],
                0
            )
            if voxelValue > 0:
                # Found a segment containing the point
                segment = segmentation.GetSegment(segId)
                if segment:
                    segName = segment.GetName()
                    detectedLevel = segName.split()[0]
                    if detectedLevel in self.levelselection:
                        slicer.mrmlScene.RemoveNode(labelmapNode)
                        return detectedLevel

            # Remove the temporary labelmap node to keep the scene clean
            slicer.mrmlScene.RemoveNode(labelmapNode)

        return None

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

      # Initialize the button style to inactive
      updateButtonStyle(False)

      # Connect to the same signal that indicates a change in the active placing mode
      self.startMeasurements.connect('activeMarkupsFiducialPlaceModeChanged(bool)', updateButtonStyle)

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
      camLayout = qt.QFormLayout(transCam)

      a = PythonQt.qMRMLWidgets.qMRMLTransformSliders()
      a.setMRMLTransformNode(slicer.mrmlScene.GetNodeByID('vtkMRMLLinearTransformNode4'))
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

      camera = slicer.mrmlScene.GetNodeByID('vtkMRMLCameraNode1')
      camera.SetPosition(0, -600, 0)
      camera.SetViewUp([0, 0, 1])
      camera.ResetClippingRange()

      # Get the segmentation node
      try:
          segNode = slicer.util.getNode("Segmentation")
      except Exception as e:
          logging.error("Segmentation node not found: " + str(e))
          segNode = None

      if segNode:
          segmentation = segNode.GetSegmentation()
          segmentIDs = segmentation.GetSegmentIDs()
          # For demonstration, we use the first segment.
          if segmentIDs:
              segId = segmentIDs[0]
              labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
              success = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                  segNode,
                  [segId],
                  labelmapNode
              )
              if success:
                  # Now, get the cropped bounds from the labelmap data.
                  labelmapData = labelmapNode.GetImageData()
                  if labelmapData:
                      rasBounds = [0] * 6
                      labelmapNode.GetRASBounds(rasBounds)
                      # Use these bounds to compute the first slice location
                      firstSliceLocation = [
                          0.5 * (rasBounds[0] + rasBounds[1]),  # x-center
                          0.5 * (rasBounds[2] + rasBounds[3]),  # y-center
                          0.9 * rasBounds[5]
                      ]
                      slicer.modules.markups.logic().JumpSlicesToLocation(
                          firstSliceLocation[0],
                          firstSliceLocation[1],
                          firstSliceLocation[2],
                          True
                      )
                  else:
                      logging.error("Labelmap image data not found.")
              else:
                  logging.error("Failed to export segment to labelmap.")
              # Clean up the temporary labelmap node when done
              slicer.mrmlScene.RemoveNode(labelmapNode)

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

      if goingTo.id() != 'Segmentation' and goingTo.id() != 'Measurements':
          return

      super(LandmarksStep, self).onExit(goingTo, transitionType)

    def validate( self, desiredBranchId ):

      self.__parent.validate( desiredBranchId )
      self.__parent.validationSucceeded(desiredBranchId)

      fidNumber = self.fiducial.GetNumberOfDefinedControlPoints()

      if fidNumber != 0:
          self.__parent.validationSucceeded(desiredBranchId)
      else:
          self.__parent.validationFailed(desiredBranchId, 'Error','Please place at least one fiducial on the model before proceeding')

    def doStepProcessing(self):
      logging.debug('Done')
      self.lockFiducials()