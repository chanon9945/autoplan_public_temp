import qt, ctk, vtk, slicer

from .PedicleScrewSimulatorStep import *
from .Helper import *
import PythonQt
import string

class MeasurementsStep(PedicleScrewSimulatorStep):

    def __init__(self, stepid):
        self.initialize(stepid)
        self.setName('4. Measurements')
        self.setDescription('Make Anatomical Measurements')

        self.__parent = super(MeasurementsStep, self)
        qt.QTimer.singleShot(0, self.killButton)

        # Various state
        self.adjustCount = 0
        self.adjustCount2 = 0
        self.rulerList = []
        self.rulerLengths = []
        self.measureCount = 0
        self.entryCount = 0
        self.rulerStatus = 0
        self.selectedComboValues = {}

    def killButton(self):
        bl = slicer.util.findChildren(text='Final')
        if len(bl):
            bl[0].hide()

    def createUserInterface(self):
        slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, self.onNodeAddedRemoved)
        slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeRemovedEvent, self.onNodeAddedRemoved)

        rulers = slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode')
        for ruler in rulers:
            ruler.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.rulerLengthCheck)

        self.__layout = self.__parent.createUserInterface()

        self.isMeasuring = False

        self.startMeasurements = qt.QPushButton("Start Measuring")
        self.startMeasurements.setStyleSheet("background-color: green;")
        self.startMeasurements.connect('clicked(bool)', self.startMeasure)

        self.adjustFiducials = qt.QPushButton("Adjust Landmarks")
        self.adjustFiducials.connect('clicked(bool)', self.makeFidAdjustments)

        self.crosshair = qt.QPushButton("Hide Crosshair")
        self.crosshair.connect('clicked(bool)', self.crosshairVisible)

        buttonLayout = qt.QHBoxLayout()
        buttonLayout.addWidget(self.startMeasurements)
        self.__layout.addRow(buttonLayout)

        buttonLayout2 = qt.QHBoxLayout()
        buttonLayout2.addWidget(self.adjustFiducials)
        buttonLayout2.addWidget(self.crosshair)
        self.__layout.addRow(buttonLayout2)

        self.fiducial = self.fiducialNode()
        self.fidNumber = self.fiducial.GetNumberOfControlPoints()
        self.fidLabels = []
        self.fidLevels = []
        self.fidSides = []
        self.oldPosition = 0

        horizontalHeaders = ["Fiducial","Level","Side","Pedicle\n Length", "Pedicle\n Width"]
        self.angleTable = qt.QTableWidget(self.fidNumber, 5)
        self.angleTable.sortingEnabled = False
        self.angleTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)  # or 1 if you want edits
        self.angleTable.setMinimumHeight(self.angleTable.verticalHeader().length())
        self.angleTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
        self.angleTable.setSizePolicy(qt.QSizePolicy.MinimumExpanding, qt.QSizePolicy.Preferred)
        self.angleTable.itemSelectionChanged.connect(self.onTableCellClicked)
        self.__layout.addWidget(self.angleTable)
        self.angleTable.setHorizontalHeaderLabels(horizontalHeaders)

        reconCollapsibleButton = ctk.ctkCollapsibleButton()
        reconCollapsibleButton.text = "Change Slice Reconstruction"
        self.__layout.addWidget(reconCollapsibleButton)
        reconCollapsibleButton.collapsed = True

        reconLayout = qt.QFormLayout(reconCollapsibleButton)
        reconLabel = qt.QLabel('Recon Slice:')
        rotationLabel = qt.QLabel('Rotation Angle:')

        self.selector = slicer.qMRMLNodeComboBox()
        self.selector.nodeTypes = ['vtkMRMLSliceNode']
        self.selector.toolTip = "Change Slice Reconstruction"
        self.selector.setMRMLScene(slicer.mrmlScene)
        self.selector.addEnabled = True

        reconLayout.addRow(reconLabel, self.selector)

        self.slider = ctk.ctkSliderWidget()
        self.slider.connect('valueChanged(double)', self.sliderValueChanged)
        self.slider.minimum = -100
        self.slider.maximum = 100
        reconLayout.addRow(rotationLabel, self.slider)

        qt.QTimer.singleShot(0, self.killButton)
        self.updateTable()

    def getParameterNode(self):
        paramNode = slicer.mrmlScene.GetSingletonNode("MyPedicleParameters", "vtkMRMLScriptedModuleNode")
        if not paramNode:
            paramNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScriptedModuleNode")
            paramNode.SetName("MyPedicleParameters")
            paramNode.SetSingletonTag("MyPedicleParameters")
            slicer.mrmlScene.AddNode(paramNode)
        return paramNode

    def updateTable(self):
        self.storeCurrentComboSelectionsInDict()

        self.fiducial = self.fiducialNode()
        self.fidNumber = self.fiducial.GetNumberOfControlPoints()
        self.angleTable.setRowCount(self.fidNumber)

        self.fidLabels = []
        self.fidLevels = []
        self.fidSides = []
        self.lengthCombo = []
        self.widthCombo = []

        for i in range(self.fidNumber):
            label = slicer.modules.BART_PlanningWidget.landmarksStep.table2.item(i, 0).text()
            level = slicer.modules.BART_PlanningWidget.landmarksStep.table2.cellWidget(i, 1).currentText
            side  = slicer.modules.BART_PlanningWidget.landmarksStep.table2.cellWidget(i, 2).currentText

            self.fidLabels.append(label)
            self.fidLevels.append(level)
            self.fidSides.append(side)

            qtLabel = qt.QTableWidgetItem(label)
            qtLevel = qt.QTableWidgetItem(level)
            qtSide  = qt.QTableWidgetItem(side)
            self.angleTable.setItem(i, 0, qtLabel)
            self.angleTable.setItem(i, 1, qtLevel)
            self.angleTable.setItem(i, 2, qtSide)

            lengthCB = qt.QComboBox()
            widthCB  = qt.QComboBox()
            lengthCB.addItem(" ")
            widthCB.addItem(" ")

            self.angleTable.setCellWidget(i, 3, lengthCB)
            self.angleTable.setCellWidget(i, 4, widthCB)

            lengthCB.connect('currentIndexChanged(int)', lambda idx, rowIndex=i: self.onLengthComboChanged(rowIndex))
            widthCB.connect('currentIndexChanged(int)',  lambda idx, rowIndex=i: self.onWidthComboChanged(rowIndex))

            self.lengthCombo.append(lengthCB)
            self.widthCombo.append(widthCB)

        self.rulerLengthCheck()

        for i, label in enumerate(self.fidLabels):
            if label in self.selectedComboValues:
                savedLength, savedWidth = self.selectedComboValues[label]
                idxLength = self.lengthCombo[i].findText(savedLength)
                if idxLength >= 0:
                    self.lengthCombo[i].setCurrentIndex(idxLength)
                idxWidth = self.widthCombo[i].findText(savedWidth)
                if idxWidth >= 0:
                    self.widthCombo[i].setCurrentIndex(idxWidth)

        paramNode = self.getParameterNode()
        for i, label in enumerate(self.fidLabels):
            nodeLength = paramNode.GetParameter(f"{label}_Length") or ""
            nodeWidth  = paramNode.GetParameter(f"{label}_Width")  or ""
            if nodeLength != "":
                idxLen = self.lengthCombo[i].findText(nodeLength)
                if idxLen >= 0:
                    self.lengthCombo[i].setCurrentIndex(idxLen)
            if nodeWidth != "":
                idxWid = self.widthCombo[i].findText(nodeWidth)
                if idxWid >= 0:
                    self.widthCombo[i].setCurrentIndex(idxWid)

    def storeCurrentComboSelectionsInDict(self):
        if not hasattr(self, 'fidLabels') or not hasattr(self, 'lengthCombo'):
            return

        for i, label in enumerate(self.fidLabels):
            selectedLength = self.lengthCombo[i].currentText
            selectedWidth  = self.widthCombo[i].currentText
            self.selectedComboValues[label] = (selectedLength, selectedWidth)

    def onLengthComboChanged(self, fidIndex):
        label = self.fidLabels[fidIndex]
        newValue = self.lengthCombo[fidIndex].currentText
        (_, oldWidth) = self.selectedComboValues.get(label, (" ", " "))
        self.selectedComboValues[label] = (newValue, oldWidth)

    def onWidthComboChanged(self, fidIndex):
        label = self.fidLabels[fidIndex]
        newValue = self.widthCombo[fidIndex].currentText
        (oldLength, _) = self.selectedComboValues.get(label, (" ", " "))
        self.selectedComboValues[label] = (oldLength, newValue)

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeAddedRemoved(self, caller, event, calldata):
        node = calldata
        if isinstance(node, slicer.vtkMRMLMarkupsLineNode):
            self.rulerAdded()

    def rulerAdded(self):
        logging.debug("ruler added: {0}".format(self.entryCount))
        rulers = slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode')

        rulerX = rulers[-1]
        rounded_length = self.round_to_05(rulerX.GetMeasurement("length").GetValue())
        rounded_length_str = "%.1f" % rounded_length
        self.rulerList.append(rounded_length_str)

        rulerX.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.rulerLengthCheck)

    def rulerLengthCheck(self, observer=None, event=None):
      self.storeCurrentComboSelectionsInDict()

      # Gather all MarkupsLineNodes that have at least 2 valid points
      rulers = [
        r for r in slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode')
        if r.GetNumberOfDefinedControlPoints() >= 2
      ]

      widthValues = set()
      lengthValues = set()

      # Figure out which measured lengths are "width" and which are "length".
      for ruler in rulers:
        val = self.round_to_05(ruler.GetMeasurement("length").GetValue())
        if val == 0.0:
          continue
        if val < 15.0:
          widthValues.add(val)
        else:
          lengthValues.add(val)

      # Re-populate all combos, but do NOT lose the user’s existing choices.
      for i, widthCB in enumerate(self.widthCombo):
        currentFidLabel = self.fidLabels[i]
        (savedLength, savedWidth) = self.selectedComboValues.get(currentFidLabel, (" ", " "))

        # Clear and re-populate
        widthCB.blockSignals(True)  # Temporarily block signals to avoid triggering onWidthComboChanged repeatedly
        widthCB.clear()
        widthCB.addItem(" ")
        for val in sorted(widthValues):
          widthCB.addItem(f"{val:.1f}")
        widthCB.blockSignals(False)

        # Restore user selection if it still exists in the new list
        if savedWidth.strip():
          idx = widthCB.findText(savedWidth)
          if idx >= 0:
            widthCB.setCurrentIndex(idx)

      for i, lengthCB in enumerate(self.lengthCombo):
        currentFidLabel = self.fidLabels[i]
        (savedLength, savedWidth) = self.selectedComboValues.get(currentFidLabel, (" ", " "))

        lengthCB.blockSignals(True)
        lengthCB.clear()
        lengthCB.addItem(" ")
        for val in sorted(lengthValues):
          lengthCB.addItem(f"{val:.1f}")
        lengthCB.blockSignals(False)

        if savedLength.strip():
          idx = lengthCB.findText(savedLength)
          if idx >= 0:
            lengthCB.setCurrentIndex(idx)

    def round_to_05(self, value):
        return round(value * 2) / 2

    def onTableCellClicked(self):
        if self.angleTable.currentColumn() <= 2:
            self.currentFid = self.angleTable.currentRow()
            self.zoomIn()
            self.sliceChange()
            self.fiducial.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.fidMove)

    def fidMove(self, observer, event):
        pass

    def sliceChange(self):
        coords = [0,0,0]
        if self.fiducial is not None:
            self.fiducial.GetNthControlPointPosition(self.currentFid, coords)
            slicer.modules.markups.logic().JumpSlicesToLocation(coords[0], coords[1], coords[2], True)

    def zoomIn(self):
        slicer.app.applicationLogic().PropagateVolumeSelection(1)
        sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
        sliceLogic  = sliceWidget.sliceLogic()
        sliceNode   = sliceLogic.GetSliceNode()
        fov = sliceNode.GetFieldOfView()
        newFov = [fov[0]*0.2, fov[1]*0.2, fov[2]]
        sliceNode.SetFieldOfView(newFov[0], newFov[1], newFov[2])
        sliceNode.UpdateMatrices()

    def makeFidAdjustments(self):
        fidNode = self.fiducialNode()
        if self.adjustCount == 0:
            slicer.modules.markups.logic().SetAllControlPointsLocked(fidNode, False)
            self.adjustCount = 1
            self.adjustFiducials.setText("Fix Landmarks")
            if self.measureCount == 1:
                self.startMeasure()
        else:
            slicer.modules.markups.logic().SetAllControlPointsLocked(fidNode, True)
            self.adjustCount = 0
            self.adjustFiducials.setText("Adjust Landmarks")

    def crosshairVisible(self):
        viewNodes = slicer.util.getNodesByClass('vtkMRMLSliceDisplayNode')
        if self.adjustCount2 == 0:
            for viewNode in viewNodes:
                viewNode.SetIntersectingSlicesVisibility(0)
            self.adjustCount2 = 1
            self.crosshair.setText("Show Crosshair")
        else:
            for viewNode in viewNodes:
                viewNode.SetIntersectingSlicesVisibility(1)
            self.adjustCount2 = 0
            self.crosshair.setText("Hide Crosshair")

    def begin(self):
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsLineNode")
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        placeModePersistence = 1
        interactionNode.SetPlaceModePersistence(placeModePersistence)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    def stop(self):
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsLineNode")
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        placeModePersistence = 1
        interactionNode.SetPlaceModePersistence(placeModePersistence)
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    def startMeasure(self):
        if self.measureCount == 0:
            self.begin()
            self.measureCount = 1
            self.startMeasurements.setText("Stop Measuring")
            self.startMeasurements.setStyleSheet("background-color: red;")
        else:
            self.stop()
            self.measureCount = 0
            self.startMeasurements.setText("Start Measuring")
            self.startMeasurements.setStyleSheet("background-color: green;")

    def sliderValueChanged(self, value):
        transform = vtk.vtkTransform()
        if self.selector.currentNodeID == 'vtkMRMLSliceNodeRed':
            redSlice = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeRed')
            transform.SetMatrix(redSlice.GetSliceToRAS())
            transform.RotateX(value - self.oldPosition)
            redSlice.GetSliceToRAS().DeepCopy(transform.GetMatrix())
            redSlice.UpdateMatrices()
        elif self.selector.currentNodeID == 'vtkMRMLSliceNodeYellow':
            yellowSlice = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeYellow')
            transform.SetMatrix(yellowSlice.GetSliceToRAS())
            transform.RotateY(value - self.oldPosition)
            yellowSlice.GetSliceToRAS().DeepCopy(transform.GetMatrix())
            yellowSlice.UpdateMatrices()
        elif self.selector.currentNodeID == 'vtkMRMLSliceNodeGreen':
            greenSlice = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeGreen')
            transform.SetMatrix(greenSlice.GetSliceToRAS())
            transform.RotateZ(value - self.oldPosition)
            greenSlice.GetSliceToRAS().DeepCopy(transform.GetMatrix())
            greenSlice.UpdateMatrices()
        self.oldPosition = value

    def onEntry(self, comingFrom, transitionType):
        super(MeasurementsStep, self).onEntry(comingFrom, transitionType)
        qt.QTimer.singleShot(0, self.killButton)

        lm = slicer.app.layoutManager()
        if lm:
            lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)

        viewNodes = slicer.util.getNodesByClass('vtkMRMLSliceDisplayNode')
        for viewNode in viewNodes:
            viewNode.SetIntersectingSlicesVisibility(1)

        rulers = slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode')
        for rulerX in rulers:
            rulerX.GetDisplayNode().SetVisibility(True)

        paramNode = self.getParameterNode()
        self.updateTable()

    def onExit(self, goingTo, transitionType):
        # Save combos to param node
        paramNode = self.getParameterNode()
        self.storeCurrentComboSelectionsInDict()

        for label, (lengthStr, widthStr) in self.selectedComboValues.items():
            paramNode.SetParameter(f"{label}_Length", lengthStr)
            paramNode.SetParameter(f"{label}_Width",  widthStr)

        super(MeasurementsStep, self).onExit(goingTo, transitionType)
        logging.debug("exiting measurements step")

        viewNodes = slicer.util.getNodesByClass('vtkMRMLSliceDisplayNode')
        for viewNode in viewNodes:
            viewNode.SetIntersectingSlicesVisibility(0)

        rulers = slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode')
        for rulerX in rulers:
            rulerX.GetDisplayNode().SetVisibility(False)

        self.stop()
        self.measureCount = 0
        self.startMeasurements.setText("Start Measuring")
        self.startMeasurements.setStyleSheet("background-color: green;")

        if goingTo.id() == 'Screw':
            logging.debug("screw step next")
            self.doStepProcessing()

        self.stop()
        self.measureCount = 0
        self.startMeasurements.setText("Start Measuring")

        if goingTo.id() not in ('Landmarks','Screw'):
            return

    def validate(self, desiredBranchId):
        self.__parent.validate(desiredBranchId)
        self.__parent.validationSucceeded(desiredBranchId)

    def doStepProcessing(self):
        logging.debug('Done')
