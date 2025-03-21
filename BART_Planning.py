import logging
import os
import unittest
from typing import Annotated, Optional

import vtk, qt, ctk, slicer

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
import logging
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

import BARTPedicleScrewSimulatorWizard

#
# BART_Planning
#

class BART_Planning(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("BART_Planning")
        self.parent.categories = ["BART_LAB"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Krittanat Tanthuvanit (BART LAB, Mahidol University)",
            "Chinnagrit Junchaya (BART LAB, Mahidol University)",
            "Phubase Netrapathompornkij (BART LAB, Mahidol University)"
        ]
        self.parent.helpText = _("""
    This module has been modified by BART LAB for research purposes.
    For more details, documentation, and the latest updates, please visit:
    <a href="https://github.com/MonoJRz/BART_Planning.git">https://github.com/MonoJRz/BART_Planning.git</a>
    """)
        self.parent.acknowledgementText = _("""
    This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
    and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
    It has been modified by BART LAB for research purposes.
    """)

# BART_PlanningWidget
#

class BART_PlanningWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Instantiate and connect widgets ...

        self.workflow = ctk.ctkWorkflow()

        self.workflowWidget = ctk.ctkWorkflowStackedWidget()
        self.workflowWidget.setWorkflow(self.workflow)
        self.layout.addWidget(self.workflowWidget)

        # create all wizard steps
        self.loadDataStep = BARTPedicleScrewSimulatorWizard.LoadDataStep('LoadData')
        self.segmentationStep = BARTPedicleScrewSimulatorWizard.SegmentationStep('Segmentation')
        self.defineROIStep = BARTPedicleScrewSimulatorWizard.DefineROIStep('DefineROI', showSidesSelector=False)
        self.measurementsStep = BARTPedicleScrewSimulatorWizard.MeasurementsStep('Measurements')
        self.landmarksStep = BARTPedicleScrewSimulatorWizard.LandmarksStep('Landmarks')
        self.screwStep = BARTPedicleScrewSimulatorWizard.ScrewStep('Screw')
        self.gradeStep = BARTPedicleScrewSimulatorWizard.GradeStep('Grade')
        self.endStep = BARTPedicleScrewSimulatorWizard.EndStep('Final')

        # add the wizard steps to an array for convenience
        allSteps = []

        allSteps.append(self.loadDataStep)
        allSteps.append(self.segmentationStep)  # Add the new segmentation step
        allSteps.append(self.defineROIStep)
        allSteps.append(self.landmarksStep)
        allSteps.append(self.measurementsStep)
        allSteps.append(self.screwStep)
        allSteps.append(self.gradeStep)
        allSteps.append(self.endStep)

        # Add transition
        # Check if volume is loaded
        # Transition from LoadData to segmentationStep
        self.workflow.addTransition(self.loadDataStep, self.segmentationStep)

        # Transition from segmentationStep to DefineROI
        self.workflow.addTransition(self.segmentationStep, self.landmarksStep)

        # # Remaining transitions
        # self.workflow.addTransition(self.defineROIStep, self.landmarksStep, 'pass', ctk.ctkWorkflow.Bidirectional)
        # self.workflow.addTransition(self.defineROIStep, self.loadDataStep, 'fail', ctk.ctkWorkflow.Bidirectional)
        #
        # # Transition from SegmentationStep to DefineROI
        # self.workflow.addTransition(self.segmentationStep, self.defineROIStep)

        self.workflow.addTransition(self.landmarksStep, self.measurementsStep, 'pass', ctk.ctkWorkflow.Bidirectional)
        self.workflow.addTransition(self.landmarksStep, self.measurementsStep, 'fail', ctk.ctkWorkflow.Bidirectional)

        self.workflow.addTransition(self.measurementsStep, self.screwStep, 'pass', ctk.ctkWorkflow.Bidirectional)
        self.workflow.addTransition(self.measurementsStep, self.screwStep, 'fail', ctk.ctkWorkflow.Bidirectional)

        self.workflow.addTransition(self.screwStep, self.gradeStep, 'pass', ctk.ctkWorkflow.Bidirectional)
        self.workflow.addTransition(self.screwStep, self.gradeStep, 'fail', ctk.ctkWorkflow.Bidirectional)

        self.workflow.addTransition(self.gradeStep, self.endStep)

        nNodes = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLScriptedModuleNode')

        self.parameterNode = None
        for n in range(nNodes):
            compNode = slicer.mrmlScene.GetNthNodeByClass(n, 'vtkMRMLScriptedModuleNode')
            nodeid = None
            if compNode.GetModuleName() == 'BART_Planning':
                self.parameterNode = compNode
                logging.debug('Found existing BART_Planning parameter node')
                break
        if self.parameterNode == None:
            self.parameterNode = slicer.vtkMRMLScriptedModuleNode()
            self.parameterNode.SetModuleName('BART_Planning')
            slicer.mrmlScene.AddNode(self.parameterNode)

        for s in allSteps:
            s.setParameterNode(self.parameterNode)

        # restore workflow step
        currentStep = self.parameterNode.GetParameter('currentStep')

        if currentStep != '':
            logging.debug('Restoring workflow step to ' + currentStep)
            if currentStep == 'LoadData':
                self.workflow.setInitialStep(self.loadDataStep)
            elif currentStep == 'Segmentation':
                self.workflow.setInitialStep(self.segmentationStep)
            elif currentStep == 'DefineROI':
                self.workflow.setInitialStep(self.defineROIStep)
            elif currentStep == 'Measurements':
                self.workflow.setInitialStep(self.measurementsStep)
            elif currentStep == 'Landmarks':
                self.workflow.setInitialStep(self.landmarksStep)
            elif currentStep == 'Screw':
                self.workflow.setInitialStep(self.screwStep)
            elif currentStep == 'Grade':
                self.workflow.setInitialStep(self.gradeStep)
            elif currentStep == 'Final':
                self.workflow.setInitialStep(self.endStep)
        else:
            logging.debug('currentStep in parameter node is empty')

        # start the workflow and show the widget
        self.workflow.start()
        self.workflowWidget.visible = True
        self.layout.addWidget(self.workflowWidget)

        # compress the layout
        # self.layout.addStretch(1)

    def cleanup(self):
        pass

    def onReload(self):
        logging.debug("Reloading BART_Planning")

        packageName = 'BARTPedicleScrewSimulatorWizard'
        submoduleNames = ['PedicleScrewSimulatorStep',
                          'DefineROIStep',
                          'SegmentationStep',
                          'EndStep',
                          'GradeStep',
                          'Helper',
                          'LandmarksStep',
                          'LoadDataStep',
                          'MeasurementsStep',
                          'ScrewStep']

        import imp
        f, filename, description = imp.find_module(packageName)
        package = imp.load_module(packageName, f, filename, description)
        for submoduleName in submoduleNames:
            f, filename, description = imp.find_module(submoduleName, package.__path__)
            try:
                imp.load_module(packageName + '.' + submoduleName, f, filename, description)
            finally:
                f.close()

        ScriptedLoadableModuleWidget.onReload(self)

class BART_PlanningTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_PedicleScrewSimulator1()

    def test_PedicleScrewSimulator1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay('No test is implemented.')