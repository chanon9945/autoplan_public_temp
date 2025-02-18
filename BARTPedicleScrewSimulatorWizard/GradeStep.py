import qt, ctk, vtk, slicer
import os
from .PedicleScrewSimulatorStep import *
from .Helper import *
import math

# TODO Add GR Score
class GradeStep(PedicleScrewSimulatorStep):

    def __init__( self, stepid ):
      self.initialize( stepid )
      self.setName( '6. Grade' )
      self.setDescription("Evaluate screw placement quality by calculating the percentage of the screw engaged with soft tissue, cancellous bone, and cortical bone (CT Hounsfield Units). Use the 'Save 3D Model' button to export the merged model of screws and segmentations.")
      self.fiduciallist = []
      self.__corticalMin = 375
      self.__corticalMax = 1200
      self.__cancellousMin = 135
      self.__cancellousMax = 375
      self.fiduciallist = []
      self.itemsqtcoP = []
      self.itemsqtcaP = []
      self.itemsqtotP = []
      self.pointsArray = []
      self.screwContact = []
      self.screwCount = 0
      self.cvn = None

      self.__parent = super( GradeStep, self )

    def killButton(self):
      # hide useless button
      bl = slicer.util.findChildren(text='Final')
      if len(bl):
        bl[0].hide()

    def createUserInterface( self ):

      self.__layout = self.__parent.createUserInterface()

      ln = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLLayoutNode')
      ln.SetViewArrangement(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalPlotView)

      # Paint Screw Button
      self.__selectScrewButton = qt.QPushButton("Grade Screws")
      self.__selectScrewButton.setStyleSheet("background-color: green;")
      self.__layout.addWidget(self.__selectScrewButton)
      self.__selectScrewButton.connect('clicked(bool)', self.gradeScrews)
      self.fiducial = self.fiducialNode()
      self.fidNumber = self.fiducial.GetNumberOfControlPoints()

      # Screw Table
      horizontalHeaders = ["Screw At","Screw\n Size","% Screw in\n Soft Tissue\n (<130HU)","% Screw in\n LD Bone\n (130-250HU)","% Screw in\n HD Bone\n (>250HU)" ]
      self.screwTable = qt.QTableWidget(self.fidNumber, 5)
      self.screwTable.sortingEnabled = False
      self.screwTable.setEditTriggers(1)
      self.screwTable.setMinimumHeight(self.screwTable.verticalHeader().length())
      self.screwTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
      self.screwTable.setSizePolicy (qt.QSizePolicy.MinimumExpanding, qt.QSizePolicy.Preferred)
      self.screwTable.itemSelectionChanged.connect(self.onTableCellClicked)
      self.__layout.addWidget(self.screwTable)

      self.screwTable.setHorizontalHeaderLabels(horizontalHeaders)

      # Add a button to save the 3D model as a single object
      self.saveModelButton = qt.QPushButton("Save 3D Model")
      self.saveModelButton.setStyleSheet("background-color: blue;")
      self.saveModelButton.toolTip = "Save the 3D view model into a single object."
      self.__layout.addWidget(self.saveModelButton)

      # Connect the button to the save function
      self.saveModelButton.clicked.connect(self.save3DModel)

      qt.QTimer.singleShot(0, self.killButton)

      self.updateTable()

    def save3DModel(self):
      # Prompt user to select output folder
      folder = qt.QFileDialog.getExistingDirectory(None, "Select Folder to Save Screws and Segmentations")
      if not folder:
          qt.QMessageBox.warning(None, "No Folder Selected", "No folder selected. Cannot save.")
          return

      # Create a new subfolder in the selected folder
      newFolderName = "Saved3DModels"
      newFolderPath = os.path.join(folder, newFolderName)
      if not os.path.exists(newFolderPath):
          os.makedirs(newFolderPath)
          print(f"Created new folder: {newFolderPath}")
      else:
          print(f"Using existing folder: {newFolderPath}")

      # Save visible screws
      screwNodes = [
          node for node in slicer.util.getNodesByClass("vtkMRMLModelNode")
          if node.GetDisplayVisibility() and node.GetName().startswith("Screw")
      ]
      if not screwNodes:
          qt.QMessageBox.warning(None, "No Screws Found", "No visible screws found.")
      else:
          for node in screwNodes:
              if node.GetParentTransformNode():
                  node.HardenTransform()
              savePath = os.path.join(newFolderPath, f"{node.GetName()}.obj")
              slicer.util.saveNode(node, savePath)
              print(f"Saved screw '{node.GetName()}' to {savePath}")

      # Merge and save visible segmentations
      segNodes = [
          node for node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
          if node.GetDisplayVisibility()
      ]
      if not segNodes:
          qt.QMessageBox.warning(None, "No Segmentations Found", "No visible segmentations found.")
          return

      # Append polydata from all visible segmentations
      appendFilter = vtk.vtkAppendPolyData()
      for segNode in segNodes:
          segmentation = segNode.GetSegmentation()
          for i in range(segmentation.GetNumberOfSegments()):
              segId = segmentation.GetNthSegmentID(i)
              segmentPolyData = vtk.vtkPolyData()
              segNode.GetClosedSurfaceRepresentation(segId, segmentPolyData)
              appendFilter.AddInputData(segmentPolyData)
      appendFilter.Update()

      # Create a single merged model node
      mergedNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "MergedSegmentations")
      mergedNode.SetAndObservePolyData(appendFilter.GetOutput())

      # Save merged model
      mergedFilePath = os.path.join(newFolderPath, "MergedSegmentation.obj")
      slicer.util.saveNode(mergedNode, mergedFilePath)

      qt.QMessageBox.information(
          None,
          "Saved Successfully",
          f"Screws and merged segmentations saved to:\n{newFolderPath}"
      )

    def onTableCellClicked(self):
      if self.screwTable.currentColumn() == 0:
          logging.debug(self.screwTable.currentRow())
          self.currentFid = self.screwTable.currentRow()
          position = [0,0,0]
          self.fiducial = self.fiducialNode()
          self.fiducial.GetNthControlPointPosition(self.currentFid,position)
          self.cameraFocus(position)
          self.sliceChange()
          #self.updateChart(self.screwList[self.currentFid])

    def cameraFocus(self, position):
      camera = slicer.mrmlScene.GetNodeByID('vtkMRMLCameraNode1')
      camera.SetFocalPoint(*position)
      camera.SetPosition(position[0],position[1],75)
      camera.SetViewUp([0,1,0])
      camera.ResetClippingRange()

    def updateTable(self):
      self.itemsLoc = []
      self.itemsLen = []
      self.itemsDia = []

      self.screwList = slicer.modules.BART_PlanningWidget.screwStep.screwList
      self.screwNumber = len(self.screwList)
      self.screwTable.setRowCount(self.screwNumber)

      for i in range(self.screwNumber):
          currentScrew = self.screwList[i]
          screwLoc = str(currentScrew[0])
          screwLen = str(currentScrew[1]) + " x " + str(currentScrew[2])

          qtscrewLoc = qt.QTableWidgetItem(screwLoc)
          qtscrewLen = qt.QTableWidgetItem(screwLen)
          #qtscrewDia = qt.QTableWidgetItem(screwDia)

          self.itemsLoc.append(qtscrewLoc)
          self.itemsLen.append(qtscrewLen)
          #self.itemsDia.append(qtscrewDia)

          self.screwTable.setItem(i, 0, qtscrewLoc)
          self.screwTable.setItem(i, 1, qtscrewLen)
          #self.screwTable.setItem(i, 2, qtscrewDia)

    def validate( self, desiredBranchId ):
      '''
      '''
      self.__parent.validate( desiredBranchId )
      self.__parent.validationSucceeded(desiredBranchId)

    def onEntry(self, comingFrom, transitionType):

      super(GradeStep, self).onEntry(comingFrom, transitionType)

      ln = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLLayoutNode')
      ln.SetViewArrangement(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalPlotView)

      pNode = self.parameterNode()

      self.fiduciallist = []
      self.fidNode = self.fiducialNode()
      for x in range (0,self.fidNode.GetNumberOfControlPoints()):
        label = self.fidNode.GetNthControlPointLabel(x)
        level = slicer.modules.BART_PlanningWidget.landmarksStep.table2.cellWidget(x,1).currentText
        side = slicer.modules.BART_PlanningWidget.landmarksStep.table2.cellWidget(x,2).currentText
        self.fiduciallist.append(label + " - " + level + " - " + side)

      logging.debug(self.fiduciallist)
      self.updateTable()

      qt.QTimer.singleShot(0, self.killButton)

    def onExit(self, goingTo, transitionType):

      if goingTo.id() == 'Screw':
          self.clearGrade()

      super(GradeStep, self).onExit(goingTo, transitionType)

    def updateWidgetFromParameters(self, pNode):
        pass

    def doStepProcessing(self):
        logging.debug('Done')

    def sliceChange(self):
        coords = [0,0,0]
        self.fiducial.GetNthControlPointPosition(self.currentFid,coords)

        lm = slicer.app.layoutManager()
        redWidget = lm.sliceWidget('Red')
        redController = redWidget.sliceController()

        yellowWidget = lm.sliceWidget('Yellow')
        yellowController = yellowWidget.sliceController()

        greenWidget = lm.sliceWidget('Green')
        greenController = greenWidget.sliceController()

        yellowController.setSliceOffsetValue(coords[0])
        greenController.setSliceOffsetValue(coords[1])
        redController.setSliceOffsetValue(coords[2])

    def gradeScrews(self):
        pNode = self.parameterNode()

        self.__inputScalarVol = pNode.GetNodeReference('baselineVolume')
        for screwIndex in range(0, len(self.fiduciallist)):
            fidName = self.fiduciallist[screwIndex]
            logging.debug(fidName)
            transformFid = slicer.mrmlScene.GetFirstNodeByName('Transform %s' % fidName)
            screwModel = slicer.mrmlScene.GetFirstNodeByName('Screw %s' % fidName)
            if screwModel != None:
                self.gradeScrew(screwModel, transformFid, fidName, screwIndex)
                self.screwCount += 1
                logging.debug("yes")
            else:
                #self.clearGrade()
                logging.debug("no")
                return
        self.chartContact(self.screwCount)

    # Crops out head of the screw
    def cropScrew(self,input, area):
        # Get bounds of screw
        bounds = input.GetPolyData().GetBounds()
        logging.debug(bounds)

        # Define bounds for cropping out the head or shaft of the screw respectively
        if area == 'head':
            i = bounds[2]
            j = bounds[3]-15
        elif area == 'shaft':
            i = bounds[3]-15
            j = bounds[3]

        # Create a box with bounds equal to that of the screw minus the head (-15)
        cropBox = vtk.vtkBox()
        cropBox.SetBounds(bounds[0],bounds[1],i,j,bounds[4],bounds[5])

        # Crop out head of screw
        extract = vtk.vtkExtractPolyDataGeometry()
        extract.SetImplicitFunction(cropBox)
        extract.SetInputData(input.GetPolyData())
        extract.Update()

        # PolyData of cropped screw
        output = extract.GetOutput()
        return output

    # Select all points in the shaft of the screw for grading
    def cropPoints(self, input):
        # Get bounds of screw
        bounds = input.GetPolyData().GetBounds()

        # Create a cube with bounds equal to that of the screw minus the head (-15)
        cropCube = vtk.vtkCubeSource()
        cropCube.SetBounds(bounds[0],bounds[1],bounds[2],bounds[3]-15,bounds[4],bounds[5])
        cropCube.Update()

        # Select points on screw within cube
        select = vtk.vtkSelectEnclosedPoints()
        select.SetInputData(input.GetPolyData())
        select.SetSurfaceData(cropCube.GetOutput())
        select.Update()

        return select

    def gradeScrew(self, screwModel, transformFid, fidName, screwIndex):
        # Reset screws
        # self.clearGrade()

        # Crop out head of screw
        croppedScrew = self.cropScrew(screwModel, 'head')

        # Clone screw model poly data
        inputModel = slicer.vtkMRMLModelNode()
        inputModel.SetAndObservePolyData(croppedScrew)
        inputModel.SetAndObserveTransformNodeID(transformFid.GetID())
        slicer.mrmlScene.AddNode(inputModel)
        slicer.vtkSlicerTransformLogic.hardenTransform(inputModel)

        # Create new model for output
        output = slicer.vtkMRMLModelNode()
        output.SetName('Grade model %s' % fidName)
        slicer.mrmlScene.AddNode(output)

        # Parameters for ProbeVolumeWithModel
        parameters = {}
        parameters["InputVolume"] = self.__inputScalarVol.GetID()
        parameters["InputModel"] = inputModel.GetID()
        parameters["OutputModel"] = output.GetID()

        probe = slicer.modules.probevolumewithmodel
        slicer.cli.run(probe, None, parameters, wait_for_completion=True)

        # Hide original screw
        modelDisplay = screwModel.GetDisplayNode()
        modelDisplay.SetColor(0,1,0)
        modelDisplay.VisibilityOff()

        # Highlight screw head
        headModel = slicer.vtkMRMLModelNode()
        headModel.SetName('Head %s' % fidName)
        headModel.SetAndObservePolyData(self.cropScrew(screwModel, 'shaft'))
        headModel.SetAndObserveTransformNodeID(transformFid.GetID())
        slicer.mrmlScene.AddNode(headModel)

        headDisplay = slicer.vtkMRMLModelDisplayNode()
        headDisplay.SetColor(0,1,0)
        slicer.mrmlScene.AddNode(headDisplay)
        headModel.SetAndObserveDisplayNodeID(headDisplay.GetID())

        # Remove clone
        slicer.mrmlScene.RemoveNode(inputModel)

        # Grade and chart screw
        self.contact(output, screwModel, fidName, screwIndex)

    def contact(self, input, screwModel, fidName, screwIndex):
        # Get points in shaft of screw
        insidePoints = self.cropPoints(screwModel)

        # Get scalars to array
        scalarsArray = input.GetPolyData().GetPointData().GetScalars('NRRDImage')
        self.pointsArray = screwModel.GetPolyData()

        # Get total number of tuples/points
        numTuples = scalarsArray.GetNumberOfTuples()

        # Reset variables
        bounds = self.pointsArray.GetPoints().GetBounds()
        lowerBound = bounds[2]  # starting point along Y
        shaftBounds = 30  # FOR NOW - adjust as needed based on the screw's geometry
        logging.debug(bounds)
        xCenter = (bounds[0] + bounds[1]) / 2
        zCenter = (bounds[4] + bounds[5]) / 2

        # Spatial Resolution
        numSegments = 50

        # Initialize arrays to accumulate HU values along the screw
        count = [0.0] * numSegments
        points = [0] * numSegments
        # Use list comprehensions to avoid issues with shallow copies
        countQ = [[0.0 for _ in range(numSegments)] for _ in range(4)]
        pointsQ = [[0 for _ in range(numSegments)] for _ in range(4)]

        corticalCount = 0
        cancellousCount = 0
        totalCount = 0

        # Process each point in the screw model
        pointHU = [0]
        pointCoord = [0, 0, 0]
        for i in range(numTuples):
            # Only consider points within the screw shaft
            if insidePoints.IsInside(i) != 1:
                continue

            totalCount += 1
            scalarsArray.GetTypedTuple(i, pointHU)
            self.pointsArray.GetPoints().GetPoint(i, pointCoord)

            # Calculate the segment index
            longitudinalIndex = int(math.floor((pointCoord[1] - lowerBound) / shaftBounds * numSegments))
            if longitudinalIndex < 0 or longitudinalIndex >= numSegments:
                continue

            # Determine the quadrant based on x and z coordinates relative to the center
            if pointCoord[0] < xCenter and pointCoord[2] >= zCenter:
                quadrantIndex = 0
            elif pointCoord[0] >= xCenter and pointCoord[2] >= zCenter:
                quadrantIndex = 1
            elif pointCoord[0] < xCenter and pointCoord[2] < zCenter:
                quadrantIndex = 2
            else:
                quadrantIndex = 3

            # Accumulate HU values and count points
            count[longitudinalIndex] += pointHU[0]
            points[longitudinalIndex] += 1
            countQ[quadrantIndex][longitudinalIndex] += pointHU[0]
            pointsQ[quadrantIndex][longitudinalIndex] += 1

            # Count points that meet cortical or cancellous criteria
            if pointHU[0] >= self.__corticalMin:
                corticalCount += 1
            elif self.__cancellousMin <= pointHU[0] < self.__corticalMin:
                cancellousCount += 1

        # Calculate the average HU value for each longitudinal segment along the screw
        avg = [0.0] * numSegments
        for j in range(numSegments):
            if points[j] > 0:
                avg[j] = count[j] / points[j]

        self.screwContact.insert(screwIndex, avg)

        # Calculate percentages of contact types
        corticalPercent = float(corticalCount) / float(totalCount) * 100 if totalCount else 0
        cancellousPercent = float(cancellousCount) / float(totalCount) * 100 if totalCount else 0
        otherPercent = 100 - corticalPercent - cancellousPercent

        # Update the table items
        qtcoP = qt.QTableWidgetItem(str("%.0f" % corticalPercent))
        qtcap = qt.QTableWidgetItem(str("%.0f" % cancellousPercent))
        qtotP = qt.QTableWidgetItem(str("%.0f" % otherPercent))

        self.itemsqtcoP.append(qtcoP)
        self.itemsqtcaP.append(qtcap)
        self.itemsqtotP.append(qtotP)

        self.screwTable.setItem(screwIndex, 4, qtcoP)
        self.screwTable.setItem(screwIndex, 3, qtcap)
        self.screwTable.setItem(screwIndex, 2, qtotP)

    def chartContact(self, screwCount):
        # Show this chart in the plot view
        plotWidget = slicer.app.layoutManager().plotWidget(0)
        plotViewNode = plotWidget.mrmlPlotViewNode()

        # Retrieve/Create plot chart node
        plotChartNode = plotViewNode.GetPlotChartNode()
        if not plotChartNode:
            plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", "Screw - Bone Contact chart")
            plotViewNode.SetPlotChartNodeID(plotChartNode.GetID())
        plotChartNode.SetTitle("Screw - Bone Contact")
        plotChartNode.SetXAxisTitle('Screw Percentile (Head - Tip)')
        plotChartNode.SetYAxisTitle('Average HU Contact')

        # Retrieve/Create plot table node
        firstPlotSeries = plotChartNode.GetNthPlotSeriesNode(0)
        plotTableNode = firstPlotSeries.GetTableNode() if firstPlotSeries else None
        if not plotTableNode:
            plotTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "Screw - Bone Contact table")

        # Remove all existing columns and add new ones
        plotTableNode.RemoveAllColumns()
        arrX = vtk.vtkFloatArray()
        arrX.SetName("Screw Percentile")
        plotTableNode.AddColumn(arrX)
        arrCortical = vtk.vtkFloatArray()
        arrCortical.SetName("Cortical Bone")
        plotTableNode.AddColumn(arrCortical)
        arrCancellous = vtk.vtkFloatArray()
        arrCancellous.SetName("Cancellous Bone")
        plotTableNode.AddColumn(arrCancellous)

        # Set the number of segments to match the increased spatial resolution
        numSegments = 50
        plotTable = plotTableNode.GetTable()
        plotTable.SetNumberOfRows(numSegments)
        for i in range(numSegments):
            plotTable.SetValue(i, 0, i * (100.0 / numSegments))
            plotTable.SetValue(i, 1, 375)
            plotTable.SetValue(i, 2, 135)

        arrays = [arrCortical, arrCancellous]

        # Add the screw contact data arrays for each screw
        for i in range(screwCount):
            arrScrew = vtk.vtkFloatArray()
            arrScrew.SetName('Screw %s' % i)
            arrScrew.SetNumberOfValues(numSegments)
            screwValues = self.screwContact[i]
            for j in range(numSegments):
                arrScrew.SetValue(j, screwValues[j])
            plotTableNode.AddColumn(arrScrew)
            arrays.append(arrScrew)

        # Update or create plot series nodes
        for arrIndex, arr in enumerate(arrays):
            plotSeriesNode = plotChartNode.GetNthPlotSeriesNode(arrIndex)
            if not plotSeriesNode:
                plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
                plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())
            plotSeriesNode.SetName("{0}".format(arr.GetName()))
            plotSeriesNode.SetAndObserveTableNodeID(plotTableNode.GetID())
            plotSeriesNode.SetXColumnName(arrX.GetName())
            plotSeriesNode.SetYColumnName(arr.GetName())
            plotSeriesNode.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
            if arrIndex < 2:
                plotSeriesNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleDash)
            else:
                plotSeriesNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleSolid)
            plotSeriesNode.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)
            plotSeriesNode.SetUniqueColor()

        # Remove any extra plot series nodes
        while plotChartNode.GetNumberOfPlotSeriesNodes() > len(arrays):
            plotChartNode.RemoveNthPlotSeriesNodeID(plotChartNode.GetNumberOfPlotSeriesNodes() - 1)

    def clearGrade(self):
        # Clear chart
        if self.cvn:
            self.cvn.SetChartNodeID(None)

        # For each fiducial, restore original screw model and remove graded screw model
        fiducial = self.fiducialNode()
        fidCount = fiducial.GetNumberOfControlPoints()
        for i in range(fidCount):
          # fiducial.SetNthControlPointVisibility(i, False)
          fidName = fiducial.GetNthControlPointLabel(i)
          screwModel = slicer.mrmlScene.GetFirstNodeByName('Screw %s' % fidName)
          if screwModel != None:
              modelDisplay = screwModel.GetDisplayNode()
              modelDisplay.SetColor(0.12,0.73,0.91)
              modelDisplay.VisibilityOn()

          gradeModel = slicer.mrmlScene.GetFirstNodeByName('Grade model %s' % fidName)
          if gradeModel != None:
              slicer.mrmlScene.RemoveNode(gradeModel)

          headModel = slicer.mrmlScene.GetFirstNodeByName('Head %s' % fidName)
          if headModel != None:
              slicer.mrmlScene.RemoveNode(headModel)
