#-----------------------------------------------------------------------------
set(MODULE_NAME BART_Planning)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  BARTPedicleScrewSimulatorWizard/__init__.py
  BARTPedicleScrewSimulatorWizard/DefineROIStep.py
  BARTPedicleScrewSimulatorWizard/EndStep.py
  BARTPedicleScrewSimulatorWizard/GradeStep.py
  BARTPedicleScrewSimulatorWizard/Helper.py
  BARTPedicleScrewSimulatorWizard/LandmarksStep.py
  BARTPedicleScrewSimulatorWizard/LoadDataStep.py
  BARTPedicleScrewSimulatorWizard/MeasurementsStep.py
  BARTPedicleScrewSimulatorWizard/PedicleScrewSimulatorStep.py
  BARTPedicleScrewSimulatorWizard/ScrewStep.py
  BARTPedicleScrewSimulatorWizard/SegmentationStep.py
  BARTPedicleScrewSimulatorWizard/TotalSegmentator.py
)
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/ScrewModels/3.5x30.stl
  Resources/ScrewModels/3.5x35.stl
  Resources/ScrewModels/3.5x40.stl
  Resources/ScrewModels/3.5x45.stl
  Resources/ScrewModels/3.5x50.stl
  Resources/ScrewModels/3.5x55.stl
  Resources/ScrewModels/3.5x60.stl
  Resources/ScrewModels/3.5x65.stl
  Resources/ScrewModels/3.5x70.stl
  Resources/ScrewModels/3.5x75.stl
  Resources/ScrewModels/3x30.stl
  Resources/ScrewModels/3x35.stl
  Resources/ScrewModels/3x40.stl
  Resources/ScrewModels/3x45.stl
  Resources/ScrewModels/3x50.stl
  Resources/ScrewModels/3x55.stl
  Resources/ScrewModels/3x60.stl
  Resources/ScrewModels/3x65.stl
  Resources/ScrewModels/3x70.stl
  Resources/ScrewModels/3x75.stl
  Resources/ScrewModels/4.5x30.stl
  Resources/ScrewModels/4.5x35.stl
  Resources/ScrewModels/4.5x40.stl
  Resources/ScrewModels/4.5x45.stl
  Resources/ScrewModels/4.5x50.stl
  Resources/ScrewModels/4.5x55.stl
  Resources/ScrewModels/4.5x60.stl
  Resources/ScrewModels/4.5x65.stl
  Resources/ScrewModels/4.5x70.stl
  Resources/ScrewModels/4.5x75.stl
  Resources/ScrewModels/4x30.stl
  Resources/ScrewModels/4x35.stl
  Resources/ScrewModels/4x40.stl
  Resources/ScrewModels/4x45.stl
  Resources/ScrewModels/4x50.stl
  Resources/ScrewModels/4x55.stl
  Resources/ScrewModels/4x60.stl
  Resources/ScrewModels/4x65.stl
  Resources/ScrewModels/4x70.stl
  Resources/ScrewModels/4x75.stl
  Resources/ScrewModels/screw head.vtk
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()