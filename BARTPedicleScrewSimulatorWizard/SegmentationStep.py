import qt, ctk, vtk, slicer
from BARTPedicleScrewSimulatorWizard import PedicleScrewSimulatorStep
import logging
from .Helper import *
from slicer.ScriptedLoadableModule import *
from TotalSegmentator import TotalSegmentatorLogic

class SegmentationStep(PedicleScrewSimulatorStep):
    def __init__(self, stepid):
        self.initialize(stepid)
        self.setName('2. Segmentation Step')
        self.setDescription('Spine segmentation step')
        self.__parent = super(SegmentationStep, self)

    def killButton(self):
      # hide useless button
      bl = slicer.util.findChildren(text='Final')
      if len(bl):
        bl[0].hide()

    def createUserInterface(self):
        self.__layout = self.__parent.createUserInterface()

        self.infoLabel = qt.QLabel("This is an segmentation step to separate spine from surrounding tissues.")
        self.infoLabel.setWordWrap(True)
        self.__layout.addRow(self.infoLabel)

        # Input volume selector with label
        inputLabel = qt.QLabel("Input Volume:")
        inputLabel.setToolTip("Select the input volume to segment.")
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("Pick the input volume to segment.")
        self.__layout.addRow(inputLabel, self.inputSelector)

        # Output segmentation selector with label
        outputLabel = qt.QLabel("Output Segmentation:")
        outputLabel.setToolTip("Select or create an output segmentation.")
        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.outputSelector.selectNodeUponCreation = True
        self.outputSelector.addEnabled = True
        self.outputSelector.removeEnabled = True
        self.outputSelector.noneEnabled = False
        self.outputSelector.showHidden = False
        self.outputSelector.showChildNodeTypes = False
        self.outputSelector.setMRMLScene(slicer.mrmlScene)
        self.outputSelector.setToolTip("Pick or create an output segmentation.")
        self.__layout.addRow(outputLabel, self.outputSelector)

        # Task selector with label
        taskLabel = qt.QLabel("Task:")
        taskLabel.setToolTip("Select the segmentation task.")
        self.taskSelector = qt.QComboBox()
        self.taskSelector.addItem("Total Segmentation", "total")
        # Add other tasks as needed
        self.taskSelector.setToolTip("Select the segmentation task.")
        self.__layout.addRow(taskLabel, self.taskSelector)

        # Fast mode checkbox
        self.fastCheckBox = qt.QCheckBox("Fast Mode")
        self.fastCheckBox.checked = False
        self.fastCheckBox.setToolTip("Enable fast mode for quicker, lower-resolution results.")
        self.__layout.addRow(self.fastCheckBox)

        # CPU mode checkbox
        self.cpuCheckBox = qt.QCheckBox("CPU Mode")
        self.cpuCheckBox.checked = False
        self.cpuCheckBox.setToolTip("Enable CPU mode.")
        self.__layout.addRow(self.cpuCheckBox)

        # Add TotalSegmentator Button with color
        self.segmentButton = qt.QPushButton("Run TotalSegmentator")
        self.segmentButton.toolTip = "Segment anatomical structures using TotalSegmentator."
        self.segmentButton.setStyleSheet(
            "background-color: orange; font-weight: bold; border-radius: 3px; padding: 5px;")
        self.segmentButton.setFixedHeight(40)
        self.__layout.addRow(self.segmentButton)
        self.segmentButton.connect('clicked()', self.runTotalSegmentator)

        # Add segmentationShow3DButton using qMRMLSegmentationShow3DButton
        self.segmentationShow3DButton = slicer.qMRMLSegmentationShow3DButton()
        self.segmentationShow3DButton.setToolTip("Toggle 3D visibility of segmentation.")
        self.segmentationShow3DButton.setStyleSheet(
            "background-color: green; font-weight: bold; border-radius: 3px; padding: 5px;")
        self.segmentationShow3DButton.setFixedHeight(40)
        self.__layout.addRow(self.segmentationShow3DButton)

        # Add a button to save the segmentation as .nrrd
        self.saveSegmentationButton = qt.QPushButton("Save Segmentation")
        self.saveSegmentationButton.toolTip = "Save the segmentation as a .nrrd file."
        self.saveSegmentationButton.setStyleSheet(
            "background-color: blue; font-weight: bold; border-radius: 3px; padding: 5px;")
        self.saveSegmentationButton.setFixedHeight(40)
        self.__layout.addRow(self.saveSegmentationButton)

        # Connect the button to the save function
        self.saveSegmentationButton.clicked.connect(self.saveSegmentation)

    def runTotalSegmentator(self):
        """
        Executes TotalSegmentator logic when the button is clicked.
        """
        try:
            inputVolumeNode = self.inputSelector.currentNode()
            outputSegmentationNode = self.outputSelector.currentNode()

            if not inputVolumeNode or not outputSegmentationNode:
                slicer.util.errorDisplay("Please select valid input and output nodes.")
                return

            task = self.taskSelector.currentData
            fast = self.fastCheckBox.checked
            cpu = self.cpuCheckBox.checked

            # Directly use TotalSegmentatorLogic
            totalSegmentatorLogic = TotalSegmentatorLogic()

            # Run the segmentation process
            totalSegmentatorLogic.process(
                inputVolume=inputVolumeNode,
                outputSegmentation=outputSegmentationNode,
                fast=fast,
                cpu=cpu,
                task=task,
                interactive=True
            )
            # self.retainSpecificSegments(outputSegmentationNode,["vertebra","sacrum"])
            self.retainSpecificSegments(outputSegmentationNode,["L1 vertebra", "L2 vertebra", "L3 vertebra", "L4 vertebra", "L5 vertebra"])
            # Increase color contrast for the retained segments
            segmentation = outputSegmentationNode.GetSegmentation()
            Colors = {
                "L1 vertebra": (0.56, 0.93, 0.56),  # Red
                "L2 vertebra": (0.75, 0.40, 0.34),  # Green
                "L3 vertebra": (0.86, 0.96, 0.08),  # Blue
                "L4 vertebra": (0.3, 0.25, 0.0),  # Yellow
                "L5 vertebra": (1.0, 0.98, 0.86),  # Magenta
            }

            for segmentName, color in Colors.items():
                segmentID = segmentation.GetSegmentIdBySegmentName(segmentName)
                if segmentID:
                    segmentation.GetSegment(segmentID).SetColor(color)
            # Set the segmentation node for the 3D visibility button
            self.segmentationShow3DButton.setSegmentationNode(outputSegmentationNode)
            # maskVolumeNode = self.convertSegmentationToMask(outputSegmentationNode, inputVolumeNode)
            # pNode = self.parameterNode()
            # pNode.SetNodeReferenceID('segmentationMask', maskVolumeNode.GetID())
            slicer.util.infoDisplay("Segmentation completed successfully.")

        except Exception as e:
            slicer.util.errorDisplay(f"Error during segmentation: {str(e)}")

    def retainSpecificSegments(self, segmentationNode, keepKeywords):
        """
        Retain only specific segments in a segmentation node based on keywords.

        Parameters:
        - segmentationNode: vtkMRMLSegmentationNode to filter.
        - keepKeywords: List of keywords (e.g., "vertebra", "sacrum") to retain.
        """
        segmentation = segmentationNode.GetSegmentation()
        segmentIdsToRemove = []

        # Identify segments to remove
        for segmentIndex in range(segmentation.GetNumberOfSegments()):
            segmentId = segmentation.GetNthSegmentID(segmentIndex)
            segmentName = segmentation.GetSegment(segmentId).GetName()

            # Check if any keyword matches the segment name
            if not any(keyword.lower() in segmentName.lower() for keyword in keepKeywords):
                segmentIdsToRemove.append(segmentId)

        # Remove identified segments
        for segmentId in segmentIdsToRemove:
            segmentation.RemoveSegment(segmentId)

    def convertSegmentationToMask(self, segmentationNode, referenceVolumeNode):
        """
        Converts a segmentation node to a binary labelmap mask.

        Parameters:
        - segmentationNode: vtkMRMLSegmentationNode to convert.
        - referenceVolumeNode: vtkMRMLScalarVolumeNode to define the output geometry.

        Returns:
        - maskVolumeNode: vtkMRMLLabelMapVolumeNode containing the mask.
        """

        # Step 1: Create a new labelmap volume node
        maskVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "SegmentationMask")

        # Export segmentation to labelmap
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            segmentationNode, maskVolumeNode, referenceVolumeNode
        )

        return maskVolumeNode

    def saveSegmentation(self):
        # Get the first segmentation node in the scene
        segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        if not segmentationNode:
            qt.QMessageBox.warning(
                None,
                "No Segmentation Found",
                "No active segmentation found to save."
            )
            return

        # Prompt user to choose a .seg.nrrd file path
        filePath = qt.QFileDialog.getSaveFileName(None, "Save Segmentation", "", "Segmentation Files (*.seg.nrrd)")
        if not filePath:
            return  # user canceled

        # Create a temporary labelmap node
        labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "SegmentationLabel")

        # Optional: define a reference volume to match geometry
        refVolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")

        # Export visible segments to the labelmap
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            segmentationNode,
            labelmapNode,
            refVolume
        )

        # Save the node to the chosen file path
        success = slicer.util.saveNode(segmentationNode, filePath)
        if success:
            qt.QMessageBox.information(None, "Save Successful", f"Segmentation saved successfully to:\n{filePath}")
        else:
            qt.QMessageBox.critical(None, "Save Failed", "Failed to save segmentation.")

        # Remove the temporary node from the scene
        slicer.mrmlScene.RemoveNode(labelmapNode)

    def onEntry(self, comingFrom, transitionType):
        super(SegmentationStep, self).onEntry(comingFrom, transitionType)
        logging.debug('SegmentationStep: Entering step.')
        lm = slicer.app.layoutManager()
        lm.setLayout(3)

        # read scalar volume node ID from previous step
        pNode = self.parameterNode()
        self.__baselineVolume = pNode.GetNodeReference('baselineVolume')

        # if ROI was created previously, get its transformation matrix and update current ROI
        roiTransformNode = pNode.GetNodeReference('roiTransform')
        if not roiTransformNode:
            roiTransformNode = slicer.vtkMRMLLinearTransformNode()
            slicer.mrmlScene.AddNode(roiTransformNode)
            pNode.SetNodeReferenceID('roiTransform', roiTransformNode.GetID())

        dm = vtk.vtkMatrix4x4()
        self.__baselineVolume.GetIJKToRASDirectionMatrix(dm)
        dm.SetElement(0, 3, 0)
        dm.SetElement(1, 3, 0)
        dm.SetElement(2, 3, 0)
        dm.SetElement(0, 0, abs(dm.GetElement(0, 0)))
        dm.SetElement(1, 1, abs(dm.GetElement(1, 1)))
        dm.SetElement(2, 2, abs(dm.GetElement(2, 2)))
        roiTransformNode.SetMatrixTransformToParent(dm)

        Helper.SetBgFgVolumes(self.__baselineVolume.GetID())
        Helper.SetLabelVolume(None)

        qt.QTimer.singleShot(0, self.killButton)

    def validate(self, desiredBranchId):
        self.__parent.validate(desiredBranchId)
        self.__parent.validationSucceeded(desiredBranchId)

    def onExit(self, goingTo, transitionType):

        # pNode = self.parameterNode()
        #
        # pNode.SetParameter('vertebra', self.vSelector.currentText)
        # if self.showSidesSelector:
        #     pNode.SetParameter('sides', self.sSelector.currentText)
        # pNode.SetParameter('inst_length', self.iSelector.currentText)
        # pNode.SetParameter('approach', self.aSelector.currentText)

        super(SegmentationStep, self).onExit(goingTo, transitionType)
        logging.debug('SegmentationStep: Exiting step.')

