import qt, ctk, vtk, slicer

from .PedicleScrewSimulatorStep import *
from .Helper import *
import DICOM
import logging

class LoadDataStep(PedicleScrewSimulatorStep):

    def __init__( self, stepid ):
      self.initialize( stepid )
      self.setName( '1. Load Image Volume' )
      self.setDescription( "Load a volume into the scene. Click 'Load spine CT from DICOM' to open the DICOM browser window. Click 'Load spine CT from other file' to import other file types, including .nrrd" )

      self.__parent = super( LoadDataStep, self )

    def killButton(self):
      # hide useless button
      bl = slicer.util.findChildren(text='Final')
      if len(bl):
        bl[0].hide()

    def createUserInterface( self ):

      self.__layout = self.__parent.createUserInterface()

      # Import/Load volume buttons
      self.__importDICOMBrowser = qt.QPushButton("Import DICOM folder")
      self.__layout.addRow(self.__importDICOMBrowser)
      self.__importDICOMBrowser.connect('clicked(bool)', self.importDICOMBrowser)

      self.__showDICOMBrowserButton = qt.QPushButton("Show DICOM browser")
      self.__layout.addRow(self.__showDICOMBrowserButton)
      self.__showDICOMBrowserButton.connect('clicked(bool)', self.showDICOMBrowser)

      self.__loadScrewButton = qt.QPushButton("Load spine CT from other file")
      self.__layout.addRow(self.__loadScrewButton)
      self.__loadScrewButton.connect('clicked(bool)', self.loadVolume)

      self.__loadSampleCtDataButton = qt.QPushButton("Load sample spine CT")
      self.__layout.addRow(self.__loadSampleCtDataButton)
      self.__loadSampleCtDataButton.connect('clicked(bool)', self.loadSampleVolume)

      # Volume selector
      self.activeText = qt.QLabel("Spine CT:")
      self.__layout.addRow(self.activeText)

      self.__inputSelector = slicer.qMRMLNodeComboBox()
      self.__inputSelector.nodeTypes = ( ("vtkMRMLScalarVolumeNode"), "" )
      self.__inputSelector.addEnabled = False
      self.__inputSelector.removeEnabled = False
      self.__inputSelector.setMRMLScene( slicer.mrmlScene )
      self.__layout.addRow(self.__inputSelector )

      # Add Window/Level sliders
      # Window slider
      self.windowSliderWidget = ctk.ctkSliderWidget()
      self.windowSliderWidget.minimum = 1
      self.windowSliderWidget.maximum = 4000
      self.windowSliderWidget.value = 1800
      self.windowSliderWidget.singleStep = 10
      self.windowSliderWidget.decimals = 0
      self.windowSliderWidget.setToolTip("Adjust the window width (WW).")
      self.__layout.addRow("Window Width (WW):", self.windowSliderWidget)
      self.windowSliderWidget.connect("valueChanged(double)", self.onWindowSliderChanged)

      # Level slider
      self.levelSliderWidget = ctk.ctkSliderWidget()
      self.levelSliderWidget.minimum = -1000
      self.levelSliderWidget.maximum = 1000
      self.levelSliderWidget.value = 400
      self.levelSliderWidget.singleStep = 10
      self.levelSliderWidget.decimals = 0
      self.levelSliderWidget.setToolTip("Adjust the window level (WL).")
      self.__layout.addRow("Window Level (WL):", self.levelSliderWidget)
      self.levelSliderWidget.connect("valueChanged(double)", self.onLevelSliderChanged)

      # Make sure sliders update if user changes selected volume
      self.__inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelectedVolumeChanged)

      # Cleanup or additional code
      qt.QTimer.singleShot(0, self.killButton)

      transform = slicer.vtkMRMLLinearTransformNode()
      transform.SetName("Camera Transform")
      slicer.mrmlScene.AddNode(transform)

      cam = slicer.mrmlScene.GetNodeByID('vtkMRMLCameraNode1')
      cam.SetAndObserveTransformNodeID('vtkMRMLLinearTransformNode4')

    # Volume and W/L callbacks
    def onSelectedVolumeChanged(self, node):
        """When user picks a volume in combo box, update the sliders to reflect the volume's WW/WL."""
        if node and node.GetScalarVolumeDisplayNode():
            displayNode = node.GetScalarVolumeDisplayNode()
            # Make sure auto W/L is off or you won't see manual changes
            displayNode.SetAutoWindowLevel(False)
            w = displayNode.GetWindow()
            l = displayNode.GetLevel()
            # Block signals so we don't trigger callbacks while setting slider
            self.windowSliderWidget.blockSignals(True)
            self.levelSliderWidget.blockSignals(True)
            self.windowSliderWidget.value = w
            self.levelSliderWidget.value = l
            self.windowSliderWidget.blockSignals(False)
            self.levelSliderWidget.blockSignals(False)

    def onWindowSliderChanged(self, newWindow):
        volumeNode = self.__inputSelector.currentNode()
        if volumeNode and volumeNode.GetScalarVolumeDisplayNode():
            displayNode = volumeNode.GetScalarVolumeDisplayNode()
            displayNode.SetAutoWindowLevel(False)
            displayNode.SetWindow(newWindow)

    def onLevelSliderChanged(self, newLevel):
        volumeNode = self.__inputSelector.currentNode()
        if volumeNode and volumeNode.GetScalarVolumeDisplayNode():
            displayNode = volumeNode.GetScalarVolumeDisplayNode()
            displayNode.SetAutoWindowLevel(False)
            displayNode.SetLevel(newLevel)

    # --------------------------------------------
    # Existing methods for loading and processing
    # --------------------------------------------
    def importDICOMBrowser(self):
      # If DICOM database is invalid then try to create a default one. If fails then show an error message.
      if slicer.modules.DICOMInstance.browserWidget is None:
        slicer.util.selectModule('DICOM')
        slicer.util.selectModule('PedicleScrewSimulator')
      # Make the DICOM browser disappear after loading data
      slicer.modules.DICOMInstance.browserWidget.browserPersistent = False
      if not slicer.dicomDatabase or not slicer.dicomDatabase.isOpen:
        # Try to create a database with default settings
        slicer.modules.DICOMInstance.browserWidget.dicomBrowser.createNewDatabaseDirectory()
        if not slicer.dicomDatabase or not slicer.dicomDatabase.isOpen:
          # Failed to create database
          # Show DICOM browser then display error message
          slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutDicomBrowserView)
          slicer.util.warningDisplay("Could not create a DICOM database with default settings. "
                                     "Please create a new database or update the existing incompatible "
                                     "database using options shown in DICOM browser.")
          return

      slicer.modules.dicom.widgetRepresentation().self().browserWidget.dicomBrowser.openImportDialog()
      slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutDicomBrowserView)

    def showDICOMBrowser(self):
      if slicer.modules.DICOMInstance.browserWidget is None:
        slicer.util.selectModule('DICOM')
        slicer.util.selectModule('PedicleScrewSimulator')
      # Make the DICOM browser disappear after loading data
      slicer.modules.DICOMInstance.browserWidget.browserPersistent = False
      slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutDicomBrowserView)

    def loadVolume(self):
        # Record what volume nodes exist before loading
        volumeNodesBefore = set(slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode'))

        # Opens the load data dialog
        slicer.util.openAddDataDialog()
        volumeNodesAfter = set(slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode'))
        newVolumeNodes = volumeNodesAfter - volumeNodesBefore

        # For each new volume:
        for volumeNode in newVolumeNodes:
            displayNode = volumeNode.GetScalarVolumeDisplayNode()
            if not displayNode:
                continue
            # Turn off auto window/level
            displayNode.SetAutoWindowLevel(False)
            # Set desired window/level (example: W=400, L=40)
            displayNode.SetWindow(1800)
            displayNode.SetLevel(400)

            self.__inputSelector.setCurrentNode(volumeNode)

    def loadSampleVolume(self):
      import SampleData
      sampleDataLogic = SampleData.SampleDataLogic()
      sampleDataLogic.downloadCTChest()

    def onEntry(self, comingFrom, transitionType):
      super(LoadDataStep, self).onEntry(comingFrom, transitionType)
      lm = slicer.app.layoutManager()
      lm.setLayout(3)
      qt.QTimer.singleShot(0, self.killButton)

    def validate( self, desiredBranchId ):
      self.__parent.validate( desiredBranchId )
      self.__baseline = self.__inputSelector.currentNode()

      pNode = self.parameterNode()
      if self.__baseline:
        baselineID = self.__baseline.GetID()
        if baselineID:
          pNode.SetNodeReferenceID('baselineVolume', baselineID)
          self.__parent.validationSucceeded(desiredBranchId)
      else:
        self.__parent.validationFailed(desiredBranchId, 'Error','Please load a volume before proceeding')

    def onExit(self, goingTo, transitionType):
      self.doStepProcessing()
      super(LoadDataStep, self).onExit(goingTo, transitionType)

    def doStepProcessing(self):
      coords = [0,0,0]
      coords = self.__baseline.GetOrigin()

      transformVolmat = vtk.vtkMatrix4x4()
      transformVolmat.SetElement(0,3,coords[0]*-1)
      transformVolmat.SetElement(1,3,coords[1]*-1)
      transformVolmat.SetElement(2,3,coords[2]*-1)

      transformVol = slicer.vtkMRMLLinearTransformNode()
      slicer.mrmlScene.AddNode(transformVol)
      transformVol.ApplyTransformMatrix(transformVolmat)

      self.__baseline.SetAndObserveTransformNodeID(transformVol.GetID())
      slicer.vtkSlicerTransformLogic.hardenTransform(self.__baseline)

      newCoords = [0,0,0,0,0,0]
      self.__baseline.GetRASBounds(newCoords)
      logging.debug(newCoords)
      shift = [0,0,0]
      shift[0] = 0.5*(newCoords[1] - newCoords[0])
      shift[1] = 0.5*(newCoords[3] - newCoords[2])
      shift[2] = 0.5*(newCoords[4] - newCoords[5])

      transformVolmat2 = vtk.vtkMatrix4x4()
      transformVolmat2.SetElement(0,3,shift[0])
      transformVolmat2.SetElement(1,3,shift[1])
      transformVolmat2.SetElement(2,3,shift[2])

      transformVol2 = slicer.vtkMRMLLinearTransformNode()
      slicer.mrmlScene.AddNode(transformVol2)
      transformVol2.ApplyTransformMatrix(transformVolmat2)

      self.__baseline.SetAndObserveTransformNodeID(transformVol2.GetID())
      slicer.vtkSlicerTransformLogic.hardenTransform(self.__baseline)

      slicer.mrmlScene.RemoveNode(transformVol)
      slicer.mrmlScene.RemoveNode(transformVol2)

      logging.debug('Done')
