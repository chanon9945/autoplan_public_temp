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
        self.setDescription(
            "This step segments the spine by separating vertebrae from surrounding tissues. "
            "Choose the input volume, set the output segmentation, select a segmentation task, "
            "and adjust performance modes before running TotalSegmentator to obtain a refined segmentation."
        )
        self.__parent = super(SegmentationStep, self)

    def killButton(self):
        # Hide the 'Final' button (not used in our workflow)
        bl = slicer.util.findChildren(text='Final')
        if len(bl):
            bl[0].hide()

    def createUserInterface(self):
        self.__layout = self.__parent.createUserInterface()

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
        self.taskSelector.addItem("Total (L1-L5)", "total")
        self.taskSelector.addItem("Vertebrae Body", "vertebrae_body")
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

        # Add Run button
        self.segmentButton = qt.QPushButton("Run Spine Segmentation")
        self.segmentButton.toolTip = "Segment anatomical structures using TotalSegmentator."
        self.segmentButton.setStyleSheet(
            "background-color: orange; font-weight: bold; border-radius: 3px; padding: 5px;"
        )
        self.segmentButton.setFixedHeight(40)
        self.__layout.addRow(self.segmentButton)
        self.segmentButton.connect('clicked()', self.runTotalSegmentator)

        # 3D visibility button
        self.segmentationShow3DButton = slicer.qMRMLSegmentationShow3DButton()
        self.segmentationShow3DButton.setToolTip("Toggle 3D visibility of segmentation.")
        self.segmentationShow3DButton.setStyleSheet(
            "background-color: green; font-weight: bold; border-radius: 3px; padding: 5px;"
        )
        self.segmentationShow3DButton.setFixedHeight(40)
        self.__layout.addRow(self.segmentationShow3DButton)

        # Save segmentation button
        self.saveSegmentationButton = qt.QPushButton("Save Segmentation")
        self.saveSegmentationButton.toolTip = "Save the segmentation as a .nrrd file."
        self.saveSegmentationButton.setStyleSheet(
            "background-color: blue; font-weight: bold; border-radius: 3px; padding: 5px;"
        )
        self.saveSegmentationButton.setFixedHeight(40)
        self.__layout.addRow(self.saveSegmentationButton)
        self.saveSegmentationButton.clicked.connect(self.saveSegmentation)

    def runTotalSegmentator(self):
        try:
            inputVolumeNode = self.inputSelector.currentNode()
            outputSegmentationNode = self.outputSelector.currentNode()

            if not inputVolumeNode or not outputSegmentationNode:
                slicer.util.errorDisplay("Please select valid input and output nodes.")
                return

            task = self.taskSelector.currentData
            fast = self.fastCheckBox.checked
            cpu = self.cpuCheckBox.checked

            # Common dictionary for lumbar vertebra colors
            lumbarVertebraColors = {
                "L1 vertebra": (0.56, 0.93, 0.56),
                "L2 vertebra": (0.75, 0.40, 0.34),
                "L3 vertebra": (0.86, 0.96, 0.08),
                "L4 vertebra": (0.3, 0.25, 0.0),
                "L5 vertebra": (1.0, 0.98, 0.86),
            }
            vertebraColor = (0.76, 0.61, 0.83)

            totalSegmentatorLogic = TotalSegmentatorLogic()

            # -------------------------------------------------
            # Two-stage approach if user selected "vertebrae_body"
            # -------------------------------------------------
            if task == "vertebrae_body":
                # 1) Run "total" task into a temporary node
                tempSegNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSegmentationNode", "TempSegmentation"
                )
                totalSegmentatorLogic.process(
                    inputVolume=inputVolumeNode,
                    outputSegmentation=tempSegNode,
                    fast=fast,
                    cpu=cpu,
                    task="total",
                    interactive=True
                )
                # Retain only L1–L5
                self.retainSpecificSegments(
                    tempSegNode,
                    ["L1 vertebra", "L2 vertebra", "L3 vertebra", "L4 vertebra", "L5 vertebra"]
                )
                # Color them
                tempSeg = tempSegNode.GetSegmentation()
                for name, color in lumbarVertebraColors.items():
                    segID = tempSeg.GetSegmentIdBySegmentName(name)
                    if segID:
                        tempSeg.GetSegment(segID).SetColor(color)

                # 2) Run "vertebrae_body" task into the user's output node
                totalSegmentatorLogic.process(
                    inputVolume=inputVolumeNode,
                    outputSegmentation=outputSegmentationNode,
                    fast=fast,
                    cpu=cpu,
                    task="vertebrae_body",
                    interactive=False
                )
                # Retain only "vertebrae_body" and color it
                self.retainSpecificSegments(outputSegmentationNode, ["vertebrae_body"])
                finalSeg = outputSegmentationNode.GetSegmentation()
                for idx in range(finalSeg.GetNumberOfSegments()):
                    sid = finalSeg.GetNthSegmentID(idx)
                    finalSeg.GetSegment(sid).SetColor(vertebraColor)

                # 3) Merge the "vertebrae_body" segment into the *temp* node (which holds L1-L5)
                #    so we temporarily get the combined segmentation
                for i in range(finalSeg.GetNumberOfSegments()):
                    segId = finalSeg.GetNthSegmentID(i)
                    segObj = finalSeg.GetSegment(segId)
                    if segObj.GetName() == "vertebrae_body":
                        tempSeg.CopySegmentFromSegmentation(finalSeg, segId)
                        segObj.SetTag('TerminologyEntry', '')
                        break

                finalSeg.RemoveAllSegments()  # Clear any leftover

                # Manually copy each segment from tempSeg to finalSeg
                segmentIDs = vtk.vtkStringArray()
                tempSeg.GetSegmentIDs(segmentIDs)
                for i in range(segmentIDs.GetNumberOfValues()):
                    segID = segmentIDs.GetValue(i)
                    finalSeg.CopySegmentFromSegmentation(tempSeg, segID)

                # 5) Remove the temporary segmentation node from the scene
                slicer.mrmlScene.RemoveNode(tempSegNode)

            # -------------------------------------------
            # Single-stage approach for other tasks
            # -------------------------------------------
            else:
                totalSegmentatorLogic.process(
                    inputVolume=inputVolumeNode,
                    outputSegmentation=outputSegmentationNode,
                    fast=fast,
                    cpu=cpu,
                    task=task,
                    interactive=True
                )
                # If "total" was chosen (or any other custom task),
                # we’ll just keep the L1–L5 vertebra segments for this example
                seg = outputSegmentationNode.GetSegmentation()
                self.retainSpecificSegments(
                    outputSegmentationNode,
                    ["L1 vertebra", "L2 vertebra", "L3 vertebra", "L4 vertebra", "L5 vertebra"]
                )
                for name, color in lumbarVertebraColors.items():
                    segID = seg.GetSegmentIdBySegmentName(name)
                    if segID:
                        seg.GetSegment(segID).SetColor(color)

            # Link the final node to the 3D visualization button
            self.segmentationShow3DButton.setSegmentationNode(outputSegmentationNode)
            slicer.util.infoDisplay("Segmentation completed successfully.")

        except Exception as e:
            slicer.util.errorDisplay(f"Error during segmentation: {str(e)}")

    def retainSpecificSegments(self, segmentationNode, keepKeywords):
        """
        Remove any segments that do not contain one of the specified
        keyword strings (case-insensitive) in their name.
        """
        segmentation = segmentationNode.GetSegmentation()
        segmentIdsToRemove = []

        # Identify segments to remove
        for segmentIndex in range(segmentation.GetNumberOfSegments()):
            segmentId = segmentation.GetNthSegmentID(segmentIndex)
            segmentName = segmentation.GetSegment(segmentId).GetName()
            if not any(keyword.lower() in segmentName.lower() for keyword in keepKeywords):
                segmentIdsToRemove.append(segmentId)

        # Remove identified segments
        for segmentId in segmentIdsToRemove:
            segmentation.RemoveSegment(segmentId)

    def convertSegmentationToMask(self, segmentationNode, referenceVolumeNode):
        """
        Convert a segmentation to a labelmap mask, returned as a
        new vtkMRMLLabelMapVolumeNode in the scene.
        """
        maskVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "SegmentationMask")
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            segmentationNode, maskVolumeNode, referenceVolumeNode
        )
        return maskVolumeNode

    def saveSegmentation(self):
        """
        Saves the user-selected output segmentation node to a .seg.nrrd file.
        """
        segmentationNode = self.outputSelector.currentNode()
        if not segmentationNode:
            qt.QMessageBox.warning(None, "No Segmentation Found",
                                   "No active segmentation found to save.")
            return

        filePath = qt.QFileDialog.getSaveFileName(
            None, "Save Segmentation", "", "Segmentation Files (*.seg.nrrd)"
        )
        if not filePath:
            return  # user canceled

        # The simplest way to save a SegmentationNode to .seg.nrrd
        success = slicer.util.saveNode(segmentationNode, filePath)
        if success:
            qt.QMessageBox.information(None, "Save Successful",
                                       f"Segmentation saved successfully to:\n{filePath}")
        else:
            qt.QMessageBox.critical(None, "Save Failed",
                                    "Failed to save segmentation.")

    def onEntry(self, comingFrom, transitionType):
        super(SegmentationStep, self).onEntry(comingFrom, transitionType)
        logging.debug('SegmentationStep: Entering step.')
        lm = slicer.app.layoutManager()
        lm.setLayout(3)

        # Read scalar volume node ID from previous step
        pNode = self.parameterNode()
        self.__baselineVolume = pNode.GetNodeReference('baselineVolume')

        # If ROI was created previously, get its transformation matrix and update current ROI
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
        super(SegmentationStep, self).onExit(goingTo, transitionType)
        logging.debug('SegmentationStep: Exiting step.')
