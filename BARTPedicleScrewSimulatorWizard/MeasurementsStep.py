import qt, ctk, vtk, slicer

from .PedicleScrewSimulatorStep import *
from .Helper import *
import PythonQt
import string

class MeasurementsStep( PedicleScrewSimulatorStep ):

    def __init__( self, stepid ):
      self.initialize( stepid )
      self.setName( '4. Measurements' )
      self.setDescription( 'Make Anatomical Measurements' )

      self.__parent = super( MeasurementsStep, self )
      qt.QTimer.singleShot(0, self.killButton)

      #self.__xnode = None
      self.adjustCount = 0
      self.adjustCount2 = 0
      self.rulerList = []
      self.rulerLengths = []
      self.measureCount = 0
      self.entryCount = 0
      self.rulerStatus = 0

    def killButton(self):
      # hide useless button
      bl = slicer.util.findChildren(text='Final')
      if len(bl):
        bl[0].hide()

    def updateTable(self):
      self.fiducial = self.fiducialNode()
      self.fidNumber = self.fiducial.GetNumberOfControlPoints()
      self.fidLabels = []
      self.fidLevels = []
      self.fidSides = []
      self.itemsLabels = []
      self.itemsLevels = []
      self.itemsSides = []
      self.rulerList = []
      self.lengthCombo = []
      self.widthCombo = []

      self.angleTable.setRowCount(self.fidNumber)

      for i in range(0,self.fidNumber):
          self.fidLabels.append(slicer.modules.BART_PlanningWidget.landmarksStep.table2.item(i,0).text())
          self.fidLevels.append(slicer.modules.BART_PlanningWidget.landmarksStep.table2.cellWidget(i,1).currentText)
          self.fidSides.append(slicer.modules.BART_PlanningWidget.landmarksStep.table2.cellWidget(i,2).currentText)

      for i in range(0,self.fidNumber):
          Label = str(self.fidLabels[i])
          Level = str(self.fidLevels[i])
          Side = str(self.fidSides[i])
          qtLabel = qt.QTableWidgetItem(Label)
          qtLevel = qt.QTableWidgetItem(Level)
          qtSide = qt.QTableWidgetItem(Side)
          self.itemsLabels.append(qtLabel)
          self.itemsLevels.append(qtLevel)
          self.itemsSides.append(qtSide)
          self.angleTable.setItem(i, 0, qtLabel)
          self.angleTable.setItem(i, 1, qtLevel)
          self.angleTable.setItem(i, 2, qtSide)

          self.lengthCombo.insert(i,qt.QComboBox())
          self.widthCombo.insert(i,qt.QComboBox())
          self.lengthCombo[i].addItem(" ")
          self.widthCombo[i].addItem(" ")
          if self.entryCount == 0:
            self.angleTable.setCellWidget(i,3, self.lengthCombo[i])
            self.angleTable.setCellWidget(i,4, self.widthCombo[i])

    def onTableCellClicked(self):
      if self.angleTable.currentColumn() <= 2:
          logging.debug(self.angleTable.currentRow())
          self.currentFid = self.angleTable.currentRow()
          self.zoomIn()
          self.sliceChange()
          self.fiducial.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.fidMove)

    def fidMove(self, observer, event):

      #coords = [0,0,0]
      #observer.GetFiducialCoordinates(coords)
      # self.sliceChange()
      pass

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
      rulers = [r for r in slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode')
                if r.GetNumberOfDefinedControlPoints() >= 2]

      # Build sets of unique "width-like" lengths and "length-like" lengths
      widthValues = set()
      lengthValues = set()
      for ruler in rulers:
        val = self.round_to_05(ruler.GetMeasurement("length").GetValue())
        if val == 0.0:
          continue
        if val < 15.0:
          widthValues.add(val)
        else:
          lengthValues.add(val)

      # For each row’s combobox, clear and then repopulate
      for widthCombo in self.widthCombo:
        # Start by clearing, re-add blank item
        widthCombo.clear()
        widthCombo.addItem(" ")
        for val in sorted(widthValues):
          widthCombo.addItem(f"{val:.1f}")

      for lengthCombo in self.lengthCombo:
        lengthCombo.clear()
        lengthCombo.addItem(" ")
        for val in sorted(lengthValues):
          lengthCombo.addItem(f"{val:.1f}")

    def round_to_05(self, value):
      return round(value * 2) / 2

    def sliceChange(self):
        logging.debug("changing")
        coords = [0,0,0]
        if self.fiducial != None:
          self.fiducial.GetNthControlPointPosition(self.currentFid,coords)
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
      newFov = [fov[0] * 0.2, fov[1] * 0.2, fov[2]]
      sliceNode.SetFieldOfView(newFov[0], newFov[1], newFov[2])

      # Update slice matrices to apply changes
      sliceNode.UpdateMatrices()

    def makeFidAdjustments(self):
      if self.adjustCount == 0:
        fidNode = self.fiducialNode()
        slicer.modules.markups.logic().SetAllControlPointsLocked(fidNode,False)
        self.adjustCount = 1
        self.adjustFiducials.setText("Fix Landmarks")
        if self.measureCount == 1:
          self.startMeasure()
      elif self.adjustCount == 1:
        fidNode = self.fiducialNode()
        slicer.modules.markups.logic().SetAllControlPointsLocked(fidNode,True)
        self.adjustCount = 0
        self.adjustFiducials.setText("Adjust Landmarks")

    def crosshairVisible(self):
      if self.adjustCount2 == 0:
        # Disable Slice Intersections
        viewNodes = slicer.util.getNodesByClass('vtkMRMLSliceDisplayNode')
        for viewNode in viewNodes:
          viewNode.SetIntersectingSlicesVisibility(0)
        self.adjustCount2 = 1
        self.crosshair.setText("Show Crosshair")
      else:
        # Enable Slice Intersections
        viewNodes = slicer.util.getNodesByClass('vtkMRMLSliceDisplayNode')
        for viewNode in viewNodes:
          viewNode.SetIntersectingSlicesVisibility(1)
        self.adjustCount2 = 0
        self.crosshair.setText("Hide Crosshair")

    def begin(self):
      selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
      # place rulers
      selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsLineNode")
      interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
      # mode 1 => place mode
      placeModePersistence = 1
      interactionNode.SetPlaceModePersistence(placeModePersistence)
      interactionNode.SetCurrentInteractionMode(1)

    def stop(self):
      selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
      # place rulers
      selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsLineNode")
      interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
      # mode 2 => view transform mode
      placeModePersistence = 1
      interactionNode.SetPlaceModePersistence(placeModePersistence)
      interactionNode.SetCurrentInteractionMode(2)

    def startMeasure(self):
      if self.measureCount == 0:
        # Switch to measuring
        self.begin()  # your function that sets place mode
        self.measureCount = 1
        self.startMeasurements.setText("Stop Measuring")
        self.startMeasurements.setStyleSheet("background-color: red;")
      else:
        # Switch to not measuring
        self.stop()  # your function that ends place mode
        self.measureCount = 0
        self.startMeasurements.setText("Start Measuring")
        self.startMeasurements.setStyleSheet("background-color: green;")

    def createUserInterface( self ):
      slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, self.onNodeAddedRemoved)
      slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeRemovedEvent, self.onNodeAddedRemoved)

      rulers = slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode')
      for ruler in rulers:
        ruler.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.rulerLengthCheck)

      self.__layout = self.__parent.createUserInterface()

      # Track whether we are currently measuring
      self.isMeasuring = False

      self.startMeasurements = qt.QPushButton("Start Measuring")
      self.startMeasurements.setStyleSheet("background-color: green;")
      # Connect the button's click to our toggling function
      self.startMeasurements.connect('clicked(bool)', self.startMeasure)

      # 2) Create any other buttons you want
      self.adjustFiducials = qt.QPushButton("Adjust Landmarks")
      self.adjustFiducials.connect('clicked(bool)', self.makeFidAdjustments)

      self.crosshair = qt.QPushButton("Hide Crosshair")
      self.crosshair.connect('clicked(bool)', self.crosshairVisible)

      # 3) Add buttons to layout
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

      logging.debug(self.fidLabels)
      logging.debug(self.fidLevels)
      logging.debug(self.fidSides)
      # Angle Table
      horizontalHeaders = ["Fiducial","Level","Side","Pedicle\n Length", "Pedicle\n Width"]

      self.angleTable = qt.QTableWidget(self.fidNumber, 5)
      self.angleTable.sortingEnabled = False
      self.angleTable.setEditTriggers(1)
      self.angleTable.setMinimumHeight(self.angleTable.verticalHeader().length())
      self.angleTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
      self.angleTable.setSizePolicy (qt.QSizePolicy.MinimumExpanding, qt.QSizePolicy.Preferred)
      self.angleTable.itemSelectionChanged.connect(self.onTableCellClicked)
      self.__layout.addWidget(self.angleTable)

      self.angleTable.setHorizontalHeaderLabels(horizontalHeaders)
      self.items = []

      reconCollapsibleButton = ctk.ctkCollapsibleButton()
      reconCollapsibleButton.text = "Change Slice Reconstruction"
      self.__layout.addWidget(reconCollapsibleButton)
      reconCollapsibleButton.collapsed = True
      # Layout
      reconLayout = qt.QFormLayout(reconCollapsibleButton)

      #label for slice selector
      reconLabel = qt.QLabel( 'Recon Slice:' )
      rotationLabel = qt.QLabel( 'Rotation Angle:' )

      #creates combobox and populates it with all slice nodes in the scene
      self.selector = slicer.qMRMLNodeComboBox()
      self.selector.nodeTypes = ['vtkMRMLSliceNode']
      self.selector.toolTip = "Change Slice Reconstruction"
      self.selector.setMRMLScene(slicer.mrmlScene)
      self.selector.addEnabled = 1

      #add label + combobox
      reconLayout.addRow( reconLabel, self.selector )

      self.slider = ctk.ctkSliderWidget()
      self.slider.connect('valueChanged(double)', self.sliderValueChanged)
      self.slider.minimum = -100
      self.slider.maximum = 100
      reconLayout.addRow( rotationLabel, self.slider)

      qt.QTimer.singleShot(0, self.killButton)
      self.updateTable()

    def sliderValueChanged(self, value):
      logging.debug(value)
      logging.debug(self.oldPosition)

      transform = vtk.vtkTransform()

      if self.selector.currentNodeID == 'vtkMRMLSliceNodeRed':
        logging.debug("red")
        redSlice = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeRed')
        transform.SetMatrix(redSlice.GetSliceToRAS())
        transform.RotateX(value - self.oldPosition)
        redSlice.GetSliceToRAS().DeepCopy(transform.GetMatrix())
        redSlice.UpdateMatrices()

      elif self.selector.currentNodeID == 'vtkMRMLSliceNodeYellow':
        logging.debug("yellow")
        redSlice = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeYellow')
        transform.SetMatrix(redSlice.GetSliceToRAS())
        transform.RotateY(value - self.oldPosition)
        redSlice.GetSliceToRAS().DeepCopy(transform.GetMatrix())
        redSlice.UpdateMatrices()

      elif self.selector.currentNodeID == 'vtkMRMLSliceNodeGreen':
        logging.debug("green")
        redSlice = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeGreen')
        transform.SetMatrix(redSlice.GetSliceToRAS())
        transform.RotateZ(value - self.oldPosition)
        redSlice.GetSliceToRAS().DeepCopy(transform.GetMatrix())
        redSlice.UpdateMatrices()
      self.oldPosition = value

    def validate( self, desiredBranchId ):
      self.__parent.validate( desiredBranchId )
      #volCheck = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')[0]
      #if volCheck != None:
      #  self.__parent.validationSucceeded('pass')
      #else:
      #slicer.mrmlScene.Clear(0)
      #  self.__parent.validationSucceeded('fail')
      self.__parent.validationSucceeded(desiredBranchId)

    def onEntry(self, comingFrom, transitionType):

      super(MeasurementsStep, self).onEntry(comingFrom, transitionType)

      qt.QTimer.singleShot(0, self.killButton)

      lm = slicer.app.layoutManager()
      if lm == None:
        return
      lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)

      logging.debug("entering measurements")
      # self.zoomIn()

      # Enable Slice Intersections
      viewNodes = slicer.util.getNodesByClass('vtkMRMLSliceDisplayNode')
      for viewNode in viewNodes:
        viewNode.SetIntersectingSlicesVisibility(1)

      rulers = slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode')
      for rulerX in rulers:
        rulerX.GetDisplayNode().SetVisibility(True)

      self.updateTable()

    def onExit(self, goingTo, transitionType):
      super(MeasurementsStep, self).onExit(goingTo, transitionType)
      logging.debug("exiting measurements step")

      # Turn off slice intersections
      viewNodes = slicer.util.getNodesByClass('vtkMRMLSliceDisplayNode')
      for viewNode in viewNodes:
        viewNode.SetIntersectingSlicesVisibility(0)

      # Hide all markups lines (rulers)
      rulers = slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode')
      for rulerX in rulers:
        rulerX.GetDisplayNode().SetVisibility(False)

      # Reset measuring state so next time we enter, it's "Start Measuring"
      self.stop()  # ensure place mode is off
      self.measureCount = 0
      self.startMeasurements.setText("Start Measuring")
      self.startMeasurements.setStyleSheet("background-color: green;")

      # If next step is 'Screw', do extra processing
      if goingTo.id() == 'Screw':
        logging.debug("screw step next")
        self.doStepProcessing()

      self.stop()
      self.measureCount = 0
      self.startMeasurements.setText("Start Measuring")

      # extra error checking, in case the user manages to click ReportROI button
      if goingTo.id() != 'Landmarks' and goingTo.id() != 'Screw':
        return

    def doStepProcessing(self):
      logging.debug('Done')
