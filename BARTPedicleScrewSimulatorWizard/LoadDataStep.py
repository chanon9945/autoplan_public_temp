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
      # First, translate volume so its origin is at (0,0,0)
      volume_origin = self.__baseline.GetOrigin()
      logging.debug(f"Original volume origin: {volume_origin}")
      
      # Create first transform to move origin to (0,0,0)
      transform_matrix1 = vtk.vtkMatrix4x4()
      transform_matrix1.Identity()
      transform_matrix1.SetElement(0, 3, -volume_origin[0])  # Negate for translation
      transform_matrix1.SetElement(1, 3, -volume_origin[1])
      transform_matrix1.SetElement(2, 3, -volume_origin[2])
      
      transform_node1 = slicer.vtkMRMLLinearTransformNode()
      transform_node1.SetName("OriginTransform")
      slicer.mrmlScene.AddNode(transform_node1)
      transform_node1.SetMatrixTransformToParent(transform_matrix1)
      
      # Apply first transform
      self.__baseline.SetAndObserveTransformNodeID(transform_node1.GetID())
      slicer.vtkSlicerTransformLogic.hardenTransform(self.__baseline)
      
      # Now get the bounds of the volume to determine its center
      bounds = [0, 0, 0, 0, 0, 0]
      self.__baseline.GetBounds(bounds)
      logging.debug(f"Volume bounds after origin translation: {bounds}")
      
      # Calculate center of the volume
      center = [
          (bounds[1] + bounds[0]) / 2.0,
          (bounds[3] + bounds[2]) / 2.0,
          (bounds[5] + bounds[4]) / 2.0
      ]
      logging.debug(f"Volume center: {center}")
      
      # Create second transform to center the volume at the origin
      transform_matrix2 = vtk.vtkMatrix4x4()
      transform_matrix2.Identity()
      transform_matrix2.SetElement(0, 3, -center[0])
      transform_matrix2.SetElement(1, 3, -center[1])
      transform_matrix2.SetElement(2, 3, -center[2])
      
      transform_node2 = slicer.vtkMRMLLinearTransformNode()
      transform_node2.SetName("CenteringTransform")
      slicer.mrmlScene.AddNode(transform_node2)
      transform_node2.SetMatrixTransformToParent(transform_matrix2)
      
      # Apply second transform
      self.__baseline.SetAndObserveTransformNodeID(transform_node2.GetID())
      slicer.vtkSlicerTransformLogic.hardenTransform(self.__baseline)
      
      # Verify the new bounds
      final_bounds = [0, 0, 0, 0, 0, 0]
      self.__baseline.GetBounds(final_bounds)
      final_center = [
          (final_bounds[1] + final_bounds[0]) / 2.0,
          (final_bounds[3] + final_bounds[2]) / 2.0,
          (final_bounds[5] + final_bounds[4]) / 2.0
      ]
      logging.debug(f"Final volume center: {final_center}")
      
      # Clean up transform nodes
      slicer.mrmlScene.RemoveNode(transform_node1)
      slicer.mrmlScene.RemoveNode(transform_node2)
      
      logging.debug('Volume centering completed')