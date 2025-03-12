import qt, ctk, vtk, slicer
import os
from .PedicleScrewSimulatorStep import *
from .Helper import *
import math

class GradeStep(PedicleScrewSimulatorStep):

    def __init__(self, stepid):
        self.initialize(stepid)
        self.setName('6. Grade')
        self.setDescription(
            "Evaluate screw placement quality by calculating the percentage "
            "of the screw engaged with soft tissue, cancellous bone, and cortical "
            "bone (CT Hounsfield Units). Use the 'Save 3D Model' button to export "
            "the model of screws and segmentations."
        )

        self.__parent = super(GradeStep, self)

        # Parameter defaults
        self.__corticalMin = 375
        self.__corticalMax = 1200
        self.__cancellousMin = 135
        self.__cancellousMax = 375

        # Internal bookkeeping
        self.fiduciallist = []
        self.itemsqtcoP = []
        self.itemsqtcaP = []
        self.itemsqtotP = []
        self.pointsArray = []
        self.screwContact = []
        self.screwCount = 0
        self.cvn = None

    def killButton(self):
        # Hide the "Final" button in the workflow
        bl = slicer.util.findChildren(text='Final')
        if len(bl):
            bl[0].hide()

    def createUserInterface(self):
        self.__layout = self.__parent.createUserInterface()
        ln = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLLayoutNode')
        ln.SetViewArrangement(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalPlotView)

        # "Grade Screws" Button
        self.__selectScrewButton = qt.QPushButton("Grade Screws")
        self.__selectScrewButton.setStyleSheet("background-color: green;")
        self.__layout.addWidget(self.__selectScrewButton)
        self.__selectScrewButton.connect('clicked(bool)', self.gradeScrews)

        # Table for screw information
        horizontalHeaders = [
            "Screw At",
            "Screw\n Size",
            "% Screw in\n LD Bone\n (135-375HU)",
            "% Screw in\n HD Bone\n (>375HU)",
            "GR Score"
        ]
        # 5 columns
        self.screwTable = qt.QTableWidget(0, 5)
        self.screwTable.sortingEnabled = False
        self.screwTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.screwTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
        self.screwTable.setSizePolicy(qt.QSizePolicy.MinimumExpanding, qt.QSizePolicy.Preferred)
        self.screwTable.itemSelectionChanged.connect(self.onTableCellClicked)
        self.__layout.addWidget(self.screwTable)
        self.screwTable.setHorizontalHeaderLabels(horizontalHeaders)

        # Button to save 3D models
        self.saveModelButton = qt.QPushButton("Save 3D Model")
        self.saveModelButton.setStyleSheet("background-color: blue;")
        self.saveModelButton.toolTip = "Save the 3D view model into a single object."
        self.__layout.addWidget(self.saveModelButton)
        self.saveModelButton.clicked.connect(self.save3DModel)

        # Prepare table rows
        self.fiducial = self.fiducialNode()
        self.fidNumber = self.fiducial.GetNumberOfControlPoints()
        self.screwTable.setRowCount(self.fidNumber)

        qt.QTimer.singleShot(0, self.killButton)
        self.updateTable()

    def save3DModel(self):
        folder = qt.QFileDialog.getExistingDirectory(None, "Select Folder to Save Screws and Segmentations")
        if not folder:
            qt.QMessageBox.warning(None, "No Folder Selected", "No folder selected. Cannot save.")
            return

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

        appendFilter = vtk.vtkAppendPolyData()
        for segNode in segNodes:
            segmentation = segNode.GetSegmentation()
            for i in range(segmentation.GetNumberOfSegments()):
                segId = segmentation.GetNthSegmentID(i)
                segmentPolyData = vtk.vtkPolyData()
                segNode.GetClosedSurfaceRepresentation(segId, segmentPolyData)
                appendFilter.AddInputData(segmentPolyData)
        appendFilter.Update()

        # Create a merged model node
        mergedNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "MergedSegmentations")
        mergedNode.SetAndObservePolyData(appendFilter.GetOutput())

        mergedFilePath = os.path.join(newFolderPath, "MergedSegmentation.obj")
        slicer.util.saveNode(mergedNode, mergedFilePath)

        qt.QMessageBox.information(
            None,
            "Saved Successfully",
            f"Screws and merged segmentations saved to:\n{newFolderPath}"
        )

    def onTableCellClicked(self):
        # If user clicks column 0, we "focus" camera on that screw
        if self.screwTable.currentColumn() == 0:
            self.currentFid = self.screwTable.currentRow()
            position = [0, 0, 0]
            self.fiducial = self.fiducialNode()
            self.fiducial.GetNthControlPointPosition(self.currentFid, position)
            self.cameraFocus(position)
            self.sliceChange()

    def cameraFocus(self, position):
        camera = slicer.mrmlScene.GetNodeByID('vtkMRMLCameraNode1')
        camera.SetFocalPoint(*position)
        camera.SetPosition(position[0], position[1], 75)
        camera.SetViewUp([0,1,0])
        camera.ResetClippingRange()

    def updateTable(self):
        self.itemsLoc = []
        self.itemsLen = []

        # Populate rows based on your screwList
        self.screwList = slicer.modules.BART_PlanningWidget.screwStep.screwList
        self.screwNumber = len(self.screwList)
        self.screwTable.setRowCount(self.screwNumber)

        for i in range(self.screwNumber):
            currentScrew = self.screwList[i]
            screwLoc = str(currentScrew[0])
            screwLen = str(currentScrew[1]) + " x " + str(currentScrew[2])

            qtscrewLoc = qt.QTableWidgetItem(screwLoc)
            qtscrewLen = qt.QTableWidgetItem(screwLen)

            self.itemsLoc.append(qtscrewLoc)
            self.itemsLen.append(qtscrewLen)

            self.screwTable.setItem(i, 0, qtscrewLoc)
            self.screwTable.setItem(i, 1, qtscrewLen)

    def validate(self, desiredBranchId):
        self.__parent.validate(desiredBranchId)
        self.__parent.validationSucceeded(desiredBranchId)

    def onEntry(self, comingFrom, transitionType):
        super(GradeStep, self).onEntry(comingFrom, transitionType)
        ln = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLLayoutNode')
        ln.SetViewArrangement(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalPlotView)

        self.fiduciallist = []
        self.fidNode = self.fiducialNode()
        for x in range(self.fidNode.GetNumberOfControlPoints()):
            label = self.fidNode.GetNthControlPointLabel(x)
            level = slicer.modules.BART_PlanningWidget.landmarksStep.table2.cellWidget(x, 1).currentText
            side = slicer.modules.BART_PlanningWidget.landmarksStep.table2.cellWidget(x, 2).currentText
            self.fiduciallist.append(label + " - " + level + " - " + side)

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
        coords = [0, 0, 0]
        self.fiducial.GetNthControlPointPosition(self.currentFid, coords)

        lm = slicer.app.layoutManager()
        redController = lm.sliceWidget('Red').sliceController()
        yellowController = lm.sliceWidget('Yellow').sliceController()
        greenController = lm.sliceWidget('Green').sliceController()

        yellowController.setSliceOffsetValue(coords[0])
        greenController.setSliceOffsetValue(coords[1])
        redController.setSliceOffsetValue(coords[2])

    def createOutsidePortionModel(self, inputModelNode, labelArrayName="NRRDImage", outsideModelName="OutsideSeg"):
        """
        Given a probed model node (points labeled > 0 = inside seg, 0 = outside seg),
        extract all 'outside' points (i.e. label = 0) and make a new model node to visualize them.
        """
        polyData = inputModelNode.GetPolyData()
        if not polyData or not polyData.GetPointData().GetScalars(labelArrayName):
            logging.warning(f"No '{labelArrayName}' array found on {inputModelNode.GetName()}")
            return None

        # threshold 0.0 to grab the 'outside' portion
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(polyData)
        threshold.SetInputArrayToProcess(0, 0, 0, 0, labelArrayName)
        threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
        threshold.SetLowerThreshold(0.0)
        threshold.SetUpperThreshold(0.0)
        threshold.Update()

        outsideDataSet = threshold.GetOutput()
        if not outsideDataSet or outsideDataSet.GetNumberOfPoints() == 0:
            logging.info(f"No outside points for {inputModelNode.GetName()}")
            return None

        # Convert unstructured grid -> polydata
        surfaceFilter = vtk.vtkGeometryFilter()
        surfaceFilter.SetInputData(outsideDataSet)
        surfaceFilter.Update()
        outsidePolyData = surfaceFilter.GetOutput()
        if not outsidePolyData or outsidePolyData.GetNumberOfPoints() == 0:
            logging.info(f"No outside geometry for {inputModelNode.GetName()}")
            return None

        # Make a new Model Node
        outsideModelNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", f"{outsideModelName}_{inputModelNode.GetName()}"
        )
        outsideModelNode.SetAndObservePolyData(outsidePolyData)

        # Give it a display node
        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        displayNode.SetColor(1.0, 0.0, 0.0)
        displayNode.SetOpacity(0.6)
        outsideModelNode.SetAndObserveDisplayNodeID(displayNode.GetID())

        # Copy any parent transform
        parentTx = inputModelNode.GetParentTransformNode()
        if parentTx:
            outsideModelNode.SetAndObserveTransformNodeID(parentTx.GetID())

        return outsideModelNode

    def getMergedSegmentationPolyData(self, segmentationNode):
        """
        Merge all segments in 'segmentationNode' into a single polydata surface.
        """
        appendFilter = vtk.vtkAppendPolyData()
        segmentation = segmentationNode.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):
            segId = segmentation.GetNthSegmentID(i)
            segPolyData = vtk.vtkPolyData()
            segmentationNode.GetClosedSurfaceRepresentation(segId, segPolyData)
            appendFilter.AddInputData(segPolyData)
        appendFilter.Update()

        # Clean
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(appendFilter.GetOutput())
        cleaner.Update()
        return cleaner.GetOutput()

    def gradeScrews(self):
        pNode = self.parameterNode()
        self.__inputScalarVol = pNode.GetNodeReference('baselineVolume')

        self.screwCount = 0
        for screwIndex in range(len(self.fiduciallist)):
            fidName = self.fiduciallist[screwIndex]
            transformFid = slicer.mrmlScene.GetFirstNodeByName('Transform %s' % fidName)
            screwModel = slicer.mrmlScene.GetFirstNodeByName('Screw %s' % fidName)

            if screwModel is not None:
                self.gradeScrew(screwModel, transformFid, fidName, screwIndex)
                self.screwCount += 1
            else:
                logging.debug(f"No screw found for {fidName}, skipping.")

        self.chartContact(self.screwCount)

    def cropScrew(self, inputModelNode, area):
        """
        Utility to isolate the 'head' or 'shaft' portion of the screw.
        """
        bounds = inputModelNode.GetPolyData().GetBounds()

        if area == 'head':
            yMin = bounds[2]
            yMax = bounds[3] - 20

            cylinder = vtk.vtkCylinder()
            cylinder.SetAxis(0, 1, 0)

            centerX = (bounds[0] + bounds[1]) / 2.0
            centerZ = (bounds[4] + bounds[5]) / 2.0
            cylinder.SetCenter(centerX, 0, centerZ)

            radiusX = (bounds[1] - bounds[0]) / 2.0
            radiusZ = (bounds[5] - bounds[4]) / 2.0
            cylinder.SetRadius(max(radiusX, radiusZ))

            lowerPlane = vtk.vtkPlane()
            lowerPlane.SetOrigin(0, yMin, 0)
            lowerPlane.SetNormal(0, -1, 0)  # y >= yMin

            upperPlane = vtk.vtkPlane()
            upperPlane.SetOrigin(0, yMax, 0)
            upperPlane.SetNormal(0, 1, 0)   # y <= yMax

            clipFunction = vtk.vtkImplicitBoolean()
            clipFunction.SetOperationTypeToIntersection()
            clipFunction.AddFunction(cylinder)
            clipFunction.AddFunction(lowerPlane)
            clipFunction.AddFunction(upperPlane)

            extract = vtk.vtkExtractPolyDataGeometry()
            extract.SetImplicitFunction(clipFunction)
            extract.SetInputData(inputModelNode.GetPolyData())
            extract.Update()
            return extract.GetOutput()

        elif area == 'shaft':
            yMin = bounds[3] - 20
            yMax = bounds[3]
            cropBox = vtk.vtkBox()
            cropBox.SetBounds(bounds[0], bounds[1], yMin, yMax, bounds[4], bounds[5])

            extract = vtk.vtkExtractPolyDataGeometry()
            extract.SetImplicitFunction(cropBox)
            extract.SetInputData(inputModelNode.GetPolyData())
            extract.Update()
            return extract.GetOutput()

        else:
            raise ValueError(f"Unknown area specified: {area}. Valid options are 'head' or 'shaft'.")

    def computeGRScore(self, screwModelNode, segmentationNode):
        """
        Classify as:
            Grade A: Entire shaft is inside segmentation (no cortical breach).
            Grade B: Breach but <= 2 mm.
            Grade C: Breach <= 4 mm.
            Grade D: Breach <= 6 mm.
            Grade E: No coverage or breach > 6 mm

        """
        if not segmentationNode:
            return "E"

        # Crop the head
        shaftPolyData = self.cropScrew(screwModelNode, 'head')
        if (not shaftPolyData) or (shaftPolyData.GetNumberOfPoints() == 0):
            # No coverage => Grade E
            return "E"

        # Make a temporary shaft model node
        shaftModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TempShaft")
        shaftModelNode.SetAndObservePolyData(shaftPolyData)

        parentTx = screwModelNode.GetParentTransformNode()
        if parentTx:
            shaftModelNode.SetAndObserveTransformNodeID(parentTx.GetID())
            slicer.vtkSlicerTransformLogic().hardenTransform(shaftModelNode)

        # Export segmentation to labelmap
        labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TempLabel")
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentationNode, labelmapNode)

        # Probe the shaft
        probedModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TempProbedShaft")
        parameters = {
            "InputVolume": labelmapNode.GetID(),
            "InputModel": shaftModelNode.GetID(),
            "OutputModel": probedModelNode.GetID(),
        }
        slicer.cli.runSync(slicer.modules.probevolumewithmodel, None, parameters)

        coverage = "No"
        polyData = probedModelNode.GetPolyData()
        if polyData and polyData.GetNumberOfPoints() > 0:
            labelArray = polyData.GetPointData().GetScalars("NRRDImage")
            if labelArray:
                numPoints = labelArray.GetNumberOfTuples()
                insideCount = 0
                for i in range(numPoints):
                    if labelArray.GetTuple1(i) > 0:
                        insideCount += 1

                if insideCount == 0:
                    coverage = "No"
                elif insideCount == numPoints:
                    coverage = "A"
                else:
                    coverage = "<A"

        finalGrade = "E"

        if coverage == "A":
            # Completely inside => A
            finalGrade = "A"

        elif coverage == "No":
            # No coverage => E
            finalGrade = "E"

        else:
            # create outside portion + measure max distance
            outsideModelNode = self.createOutsidePortionModel(
                probedModelNode, "NRRDImage", f"OutsideSeg_{screwModelNode.GetName()}"
            )
            if outsideModelNode:
                segPolyData = self.getMergedSegmentationPolyData(segmentationNode)
                if segPolyData and segPolyData.GetNumberOfPoints() > 0:
                    #Fast Spatial Queries
                    locator = vtk.vtkCellLocator()
                    locator.SetDataSet(segPolyData)
                    locator.BuildLocator()

                    outsidePolyData = outsideModelNode.GetPolyData()
                    nOutside = outsidePolyData.GetNumberOfPoints()

                    maxDistance = 0.0
                    maxOutsidePt = [0.0, 0.0, 0.0]
                    maxBoundaryPt = [0.0, 0.0, 0.0]
                    tempClosest = [0.0, 0.0, 0.0]

                    for i in range(nOutside):
                        outsidePt = [0.0, 0.0, 0.0]
                        outsidePolyData.GetPoint(i, outsidePt)

                        cellId = vtk.mutable(0)
                        subId = vtk.mutable(0)
                        dist2 = vtk.mutable(0.0)
                        locator.FindClosestPoint(outsidePt, tempClosest, cellId, subId, dist2)

                        dist = math.sqrt(dist2.get())
                        if dist > maxDistance:
                            maxDistance = dist
                            maxOutsidePt[:] = outsidePt
                            maxBoundaryPt[:] = tempClosest

                    # classify based on maxDistance
                    if maxDistance <= 2.0:
                        finalGrade = "B"
                    elif maxDistance <= 4.0:
                        finalGrade = "C"
                    elif maxDistance <= 6.0:
                        finalGrade = "D"
                    else:
                        finalGrade = "E"

                    # create the line model for vector
                    if maxDistance > 0.0:
                        lineSource = vtk.vtkLineSource()
                        lineSource.SetPoint1(maxOutsidePt)
                        lineSource.SetPoint2(maxBoundaryPt)
                        lineSource.Update()

                        lineModelNode = slicer.mrmlScene.AddNewNodeByClass(
                            "vtkMRMLModelNode", f"Vector_{screwModelNode.GetName()}"
                        )
                        lineModelNode.SetAndObservePolyData(lineSource.GetOutput())

                        lineDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
                        lineDisplay.SetColor(1.0, 1.0, 0.0)  # e.g. Yellow
                        lineDisplay.SetLineWidth(3.0)
                        lineModelNode.SetAndObserveDisplayNodeID(lineDisplay.GetID())

            else:
                finalGrade = "E"

        # Cleanup
        slicer.mrmlScene.RemoveNode(shaftModelNode)
        slicer.mrmlScene.RemoveNode(labelmapNode)
        slicer.mrmlScene.RemoveNode(probedModelNode)

        return finalGrade

    def cropPoints(self, screwModelNode, showCylinder=True, headLength=15.1):
        """
        Create a vtkSelectEnclosedPoints object that indicates which points
        lie within a "shaft" portion of the screw.
        """
        fullBounds = screwModelNode.GetPolyData().GetBounds()

        # Clip off the top "headLength" mm in Y.
        yClip = fullBounds[3] - headLength
        if yClip < fullBounds[2]:
            yClip = fullBounds[2]

        plane = vtk.vtkPlane()
        plane.SetOrigin(0, yClip, 0)
        plane.SetNormal(0, 1, 0)  # inside: y <= yClip

        clipFilter = vtk.vtkExtractPolyDataGeometry()
        clipFilter.SetImplicitFunction(plane)
        clipFilter.SetInputData(screwModelNode.GetPolyData())
        clipFilter.Update()

        shaftPolyData = clipFilter.GetOutput()
        if not shaftPolyData or shaftPolyData.GetNumberOfPoints() == 0:
            # If there's nothing after clipping, return a dummy
            select = vtk.vtkSelectEnclosedPoints()
            select.SetInputData(screwModelNode.GetPolyData())
            emptyPoly = vtk.vtkPolyData()  # empty surface
            select.SetSurfaceData(emptyPoly)
            select.Update()
            return select

        # bounding box for clipped region
        shaftBounds = shaftPolyData.GetBounds()
        shaftHeight = shaftBounds[3] - shaftBounds[2]
        centerX = (shaftBounds[0] + shaftBounds[1]) / 2.0
        centerY = (shaftBounds[2] + shaftBounds[3]) / 2.0
        centerZ = (shaftBounds[4] + shaftBounds[5]) / 2.0

        radiusX = (shaftBounds[1] - shaftBounds[0]) / 2.0
        radiusZ = (shaftBounds[5] - shaftBounds[4]) / 2.0
        shaftRadius = max(radiusX, radiusZ)

        cylinder = vtk.vtkCylinderSource()
        cylinder.SetResolution(50)
        cylinder.SetHeight(shaftHeight)
        cylinder.SetRadius(shaftRadius)
        cylinder.CappingOn()
        cylinder.Update()

        rotateTransform = vtk.vtkTransform()
        rotateFilter = vtk.vtkTransformPolyDataFilter()
        rotateFilter.SetTransform(rotateTransform)
        rotateFilter.SetInputConnection(cylinder.GetOutputPort())
        rotateFilter.Update()

        translateTransform = vtk.vtkTransform()
        translateTransform.Translate(centerX, centerY, centerZ)
        translateFilter = vtk.vtkTransformPolyDataFilter()
        translateFilter.SetTransform(translateTransform)
        translateFilter.SetInputConnection(rotateFilter.GetOutputPort())
        translateFilter.Update()

        finalCylinderPolyData = translateFilter.GetOutput()

        # Optionally show the cylinder
        if showCylinder:
            cylinderModelName = f"Cylinder_{screwModelNode.GetName()}"
            cylinderModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", cylinderModelName)
            cylinderModelNode.SetAndObservePolyData(finalCylinderPolyData)

            cylinderDisplayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            cylinderDisplayNode.SetColor(1, 0, 0)  # red
            cylinderDisplayNode.SetOpacity(0.3)
            cylinderModelNode.SetAndObserveDisplayNodeID(cylinderDisplayNode.GetID())

            parentTransform = screwModelNode.GetParentTransformNode()
            if parentTransform:
                cylinderModelNode.SetAndObserveTransformNodeID(parentTransform.GetID())

        select = vtk.vtkSelectEnclosedPoints()
        select.SetInputData(screwModelNode.GetPolyData())
        select.SetSurfaceData(finalCylinderPolyData)
        select.Update()

        return select

    def gradeScrew(self, screwModel, transformFid, fidName, screwIndex):
        # Crop out head portion
        croppedScrew = self.cropScrew(screwModel, 'head')

        # Make a temp node for ProbeVolumeWithModel
        inputModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TempInputModel")
        inputModel.SetAndObservePolyData(croppedScrew)
        inputModel.SetAndObserveTransformNodeID(transformFid.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(inputModel)

        # Create the probed output
        output = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"Grade model {fidName}")
        parameters = {
            "InputVolume": self.__inputScalarVol.GetID(),
            "InputModel": inputModel.GetID(),
            "OutputModel": output.GetID(),
        }
        slicer.cli.runSync(slicer.modules.probevolumewithmodel, None, parameters)

        # Hide original screw
        if screwModel.GetDisplayNode():
            screwModel.GetDisplayNode().SetColor(0,1,0)
            screwModel.GetDisplayNode().VisibilityOff()

        # Show "head" portion as a new model
        headPolyData = self.cropScrew(screwModel, 'shaft')
        headModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"Head {fidName}")
        headModel.SetAndObservePolyData(headPolyData)
        headModel.SetAndObserveTransformNodeID(transformFid.GetID())

        headDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        headDisplay.SetColor(0,1,0)
        headModel.SetAndObserveDisplayNodeID(headDisplay.GetID())

        # GRScore
        segNode = slicer.mrmlScene.GetFirstNodeByName("Segmentation")
        GRScore = self.computeGRScore(screwModel, segNode)
        GRItem = qt.QTableWidgetItem(GRScore)
        self.screwTable.setItem(screwIndex, 4, GRItem)

        # Cleanup
        slicer.mrmlScene.RemoveNode(inputModel)

        # 8. HU-based contact
        self.contact(output, screwModel, fidName, screwIndex)

    def contact(self, outputModel, screwModel, fidName, screwIndex):
        """
        Classify points in the shaft region by HU ranges:
         - Cancellous: < corticalMin
         - Cortical: >= corticalMin
        Then store the percentages in columns 2 and 3.
        """
        insidePoints = self.cropPoints(screwModel, showCylinder=True)
        polyData = outputModel.GetPolyData()
        scalarsArray = polyData.GetPointData().GetScalars('NRRDImage') if polyData else None
        if not scalarsArray:
            self.screwTable.setItem(screwIndex, 2, qt.QTableWidgetItem("0"))
            self.screwTable.setItem(screwIndex, 3, qt.QTableWidgetItem("0"))
            return

        numTuples = scalarsArray.GetNumberOfTuples()
        pointsArray = screwModel.GetPolyData()

        bounds = pointsArray.GetPoints().GetBounds()
        lowerBound = bounds[2]

        screwSize = self.screwList[screwIndex]
        screwLength = float(screwSize[2])
        shaftBounds = screwLength

        numSegments = 50
        count = [0.0] * numSegments
        nPoints = [0] * numSegments

        corticalCount = 0
        totalCount = 0

        tmpHU = [0]
        coord = [0, 0, 0]
        for i in range(numTuples):
            # Only consider points that are inside the 'shaft'
            if insidePoints.IsInside(i) != 1:
                continue
            totalCount += 1
            scalarsArray.GetTypedTuple(i, tmpHU)
            pointsArray.GetPoints().GetPoint(i, coord)

            longIndex = int(math.floor((coord[1] - lowerBound) / shaftBounds * numSegments))
            if 0 <= longIndex < numSegments:
                count[longIndex] += tmpHU[0]
                nPoints[longIndex] += 1

            if tmpHU[0] >= self.__corticalMin:
                corticalCount += 1

        # Mean HU along the shaft
        avgHUalongShaft = [0.0]*numSegments
        for j in range(numSegments):
            if nPoints[j] > 0:
                avgHUalongShaft[j] = count[j] / nPoints[j]

        # Keep for chart display
        self.screwContact.insert(screwIndex, avgHUalongShaft)

        corticalPercent = 0.0
        if totalCount > 0:
            corticalPercent = 100.0 * corticalCount / totalCount
            cancellousPercent = 100.0 - corticalPercent
        else:
            cancellousPercent = 0.0

        ldItem = qt.QTableWidgetItem(str(int(round(cancellousPercent))))
        hdItem = qt.QTableWidgetItem(str(int(round(corticalPercent))))

        self.screwTable.setItem(screwIndex, 2, ldItem)
        self.screwTable.setItem(screwIndex, 3, hdItem)

    def chartContact(self, screwCount):
        plotWidget = slicer.app.layoutManager().plotWidget(0)
        plotViewNode = plotWidget.mrmlPlotViewNode()

        plotChartNode = plotViewNode.GetPlotChartNode()
        if not plotChartNode:
            plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", "Screw - Bone Contact chart")
            plotViewNode.SetPlotChartNodeID(plotChartNode.GetID())

        plotChartNode.SetTitle("Screw - Bone Contact")
        plotChartNode.SetXAxisTitle('Screw Percentile (Tip - Head)')
        plotChartNode.SetYAxisTitle('Average HU Contact')

        # Retrieve (or create) a table node for data
        firstPlotSeries = plotChartNode.GetNthPlotSeriesNode(0)
        plotTableNode = firstPlotSeries.GetTableNode() if firstPlotSeries else None
        if not plotTableNode:
            plotTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "Screw - Bone Contact table")

        # Clear any old data
        plotTableNode.RemoveAllColumns()

        # X column + reference line
        arrX = vtk.vtkFloatArray()
        arrX.SetName("Screw Percentile")
        plotTableNode.AddColumn(arrX)

        arrCortical = vtk.vtkFloatArray()
        arrCortical.SetName("Cortical Bone")
        plotTableNode.AddColumn(arrCortical)

        numSegments = 50
        plotTable = plotTableNode.GetTable()
        plotTable.SetNumberOfRows(numSegments)
        for i in range(numSegments):
            plotTable.SetValue(i, 0, i * (100.0 / numSegments))
            plotTable.SetValue(i, 1, self.__corticalMin)

        arrays = [arrCortical]

        # Add each screw's contact array
        for i in range(screwCount):
            arrScrew = vtk.vtkFloatArray()
            arrScrew.SetName(f"Screw {i}")
            arrScrew.SetNumberOfValues(numSegments)
            screwValues = self.screwContact[i]
            for j in range(numSegments):
                arrScrew.SetValue(j, screwValues[j])
            plotTableNode.AddColumn(arrScrew)
            arrays.append(arrScrew)

        # Some color palette
        colors = [
            (1, 0, 0),
            (0.74, 0.25, 0.11),
            (0.11, 0.32, 0.64),
            (0.89, 0.70, 0.02),
            (0.47, 0.67, 0.33),
            (0.56, 0.39, 0.64),
            (0.95, 0.61, 0.37),
            (0.53, 0.00, 0.00),
            (0.00, 0.80, 0.95),
            (0.35, 0.34, 0.30),
            (0.47, 0.31, 0.22)
        ]

        for arrIndex, arr in enumerate(arrays):
            plotSeriesNode = plotChartNode.GetNthPlotSeriesNode(arrIndex)
            if not plotSeriesNode:
                plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
                plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())

            plotSeriesNode.SetName(arr.GetName())
            plotSeriesNode.SetAndObserveTableNodeID(plotTableNode.GetID())
            plotSeriesNode.SetXColumnName(arrX.GetName())
            plotSeriesNode.SetYColumnName(arr.GetName())
            plotSeriesNode.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)

            # dashed line for reference, solid for screws
            if arrIndex == 0:
                plotSeriesNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleDash)
            else:
                plotSeriesNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleSolid)
            plotSeriesNode.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)

            color = colors[arrIndex % len(colors)]
            plotSeriesNode.SetColor(color[0], color[1], color[2])

            if hasattr(plotSeriesNode, 'SetLineWidth'):
                plotSeriesNode.SetLineWidth(2)

        # Remove extras
        while plotChartNode.GetNumberOfPlotSeriesNodes() > len(arrays):
            plotChartNode.RemoveNthPlotSeriesNodeID(
                plotChartNode.GetNumberOfPlotSeriesNodes() - 1
            )

    def clearGrade(self):
        # Remove chart if exists
        if self.cvn:
            self.cvn.SetChartNodeID(None)

        fiducial = self.fiducialNode()
        fidCount = fiducial.GetNumberOfControlPoints()
        for i in range(fidCount):
            fidName = fiducial.GetNthControlPointLabel(i)
            screwModel = slicer.mrmlScene.GetFirstNodeByName('Screw %s' % fidName)
            if screwModel and screwModel.GetDisplayNode():
                screwModel.GetDisplayNode().SetColor(0.12, 0.73, 0.91)
                screwModel.GetDisplayNode().VisibilityOn()

            gradeModel = slicer.mrmlScene.GetFirstNodeByName('Grade model %s' % fidName)
            if gradeModel:
                slicer.mrmlScene.RemoveNode(gradeModel)

            headModel = slicer.mrmlScene.GetFirstNodeByName('Head %s' % fidName)
            if headModel:
                slicer.mrmlScene.RemoveNode(headModel)
