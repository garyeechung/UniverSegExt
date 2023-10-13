import logging
import os

import qt
import vtk

import slicer

from SegmentEditorEffects import *

logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

class SegmentEditorEffect(AbstractScriptedSegmentEditorEffect):
    """This effect uses Watershed algorithm to partition the input volume"""

    def __init__(self, scriptedEffect):
        scriptedEffect.name = 'UniverSeg'
        scriptedEffect.perSegment = True  # this effect operates on all segments at once (not on a single selected segment)
        scriptedEffect.requireSegments = True  # this effect requires segment(s) existing in the segmentation
        AbstractScriptedSegmentEditorEffect.__init__(self, scriptedEffect)

    def clone(self):
        # It should not be necessary to modify this method
        import qSlicerSegmentationsEditorEffectsPythonQt as effects
        clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
        clonedEffect.setPythonSource(__file__.replace('\\', '/'))
        return clonedEffect

    def icon(self):
        # It should not be necessary to modify this method
        iconPath = os.path.join(os.path.dirname(__file__), 'SegmentEditorEffect.png')
        if os.path.exists(iconPath):
            return qt.QIcon(iconPath)
        return qt.QIcon()

    def helpText(self):
        return "Given the support set (images + masks), it returns a mask as the same task in support set."

    def setupOptionsFrame(self):

        # Object scale slider
        self.objectScaleMmSlider = slicer.qMRMLSliderWidget()
        self.objectScaleMmSlider.setMRMLScene(slicer.mrmlScene)
        # self.objectScaleMmSlider.quantity = "length"  # get unit, precision, etc. from MRML unit node
        self.objectScaleMmSlider.minimum = 0
        self.objectScaleMmSlider.maximum = 100
        # self.objectScaleMmSlider.value = 50
        self.objectScaleMmSlider.setToolTip('Increasing this value smooths the segmentation and reduces leaks. This is the sigma used for edge detection.')
        self.scriptedEffect.addLabeledOptionsWidget("Threshold(%):", self.objectScaleMmSlider)
        self.objectScaleMmSlider.connect('valueChanged(double)', self.updateMRMLFromGUI)

        # Apply button
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.objectName = self.__class__.__name__ + 'Apply'
        self.applyButton.setToolTip("Accept previewed result")
        self.scriptedEffect.addOptionsWidget(self.applyButton)
        self.applyButton.connect('clicked()', self.onApply)

    def createCursor(self, widget):
        # Turn off effect-specific cursor for this effect
        return slicer.util.mainWindow().cursor

    def setMRMLDefaults(self):
        self.scriptedEffect.setParameterDefault("ObjectScaleMm", 50)

    def updateGUIFromMRML(self):
        objectScaleMm = self.scriptedEffect.doubleParameter("ObjectScaleMm")
        wasBlocked = self.objectScaleMmSlider.blockSignals(True)
        self.objectScaleMmSlider.value = abs(objectScaleMm)
        self.objectScaleMmSlider.blockSignals(wasBlocked)

    def updateMRMLFromGUI(self):
        self.scriptedEffect.setParameter("ObjectScaleMm", self.objectScaleMmSlider.value)

    def onApply(self):

        # Get list of visible segment IDs, as the effect ignores hidden segments.
        segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
        visibleSegmentIds = vtk.vtkStringArray()
        segmentationNode.GetDisplayNode().GetVisibleSegmentIDs(visibleSegmentIds)
        if visibleSegmentIds.GetNumberOfValues() == 0:
            logging.info("Smoothing operation skipped: there are no visible segments")
            return

        # This can be a long operation - indicate it to the user
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

        # Allow users revert to this state by clicking Undo
        self.scriptedEffect.saveStateForUndo()

        # Export source image data to temporary new volume node.
        # Note: Although the original source volume node is already in the scene, we do not use it here,
        # because the source volume may have been resampled to match segmentation geometry.
        sourceVolumeNode = slicer.vtkMRMLScalarVolumeNode()
        slicer.mrmlScene.AddNode(sourceVolumeNode)
        sourceVolumeNode.SetAndObserveTransformNodeID(segmentationNode.GetTransformNodeID())
        slicer.vtkSlicerSegmentationsModuleLogic.CopyOrientedImageDataToVolumeNode(self.scriptedEffect.sourceVolumeImageData(), sourceVolumeNode)
        # Generate merged labelmap of all visible segments, as the filter expects a single labelmap with all the labels.
        mergedLabelmapNode = slicer.vtkMRMLLabelMapVolumeNode()
        slicer.mrmlScene.AddNode(mergedLabelmapNode)
        slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(segmentationNode, visibleSegmentIds, mergedLabelmapNode, sourceVolumeNode)

        # Run segmentation algorithm
        import SimpleITK as sitk
        import sitkUtils
        from universeg import universeg
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.warning(f"{os.listdir()}")
        logging.warning(f"{device}")

        # Read input data from Slicer into SimpleITK
        # The labelImage is the segment layer that we added in Segment Editor in Slicer
        # Currenrly should be an empty mask, will be replace with predicted mask later.
        labelImage = sitk.ReadImage(sitkUtils.GetSlicerITKReadWriteAddress(mergedLabelmapNode.GetName()))
        backgroundImage = sitk.ReadImage(sitkUtils.GetSlicerITKReadWriteAddress(sourceVolumeNode.GetName()))

        # Convert SimpleITK.Image instance to numpy.Array
        # lab = sitk.GetArrayFromImage(labelImage)
        img = sitk.GetArrayFromImage(backgroundImage)

        # According to the official repo: https://github.com/JJGO/UniverSeg
        # TODO: resize img to (B, 1, 128, 128)
        # TODO: Normalize values to [0, 1]

        # TODO: instantiate UniverSeg model
        model = universeg(pretrained=True)
        model.to(device)

        # TODO: example dataset or user's own data, psudo code as follow

        # if example:
        #     use example dataset like oasis or wbc
        # else:
        #     read uploaded data

        # prediction = model(
        #     target_image,        # (B, 1, H, W)
        #     support_images,      # (B, S, 1, H, W)
        #     support_labels,      # (B, S, 1, H, W)
        # ) # -> (B, 1, H, W)

        # TODO: convert prob. to binary with the given threshold
        threshold = float(self.scriptedEffect.doubleParameter("ObjectScaleMm"))
        # lab = prediction >= threshold

        # TODO: resize the label mask to the image's original shape, e.g. (600, 512)
        # lab = resize(lab, original_shape)

        # TODO: replace labelImage with lab

        # Pixel type of watershed output is the same as the input. Convert it to int16 now.
        if labelImage.GetPixelID() != sitk.sitkInt16:
            labelImage = sitk.Cast(labelImage, sitk.sitkInt16)
        # Write result from SimpleITK to Slicer. This currently performs a deep copy of the bulk data.
        sitk.WriteImage(labelImage, sitkUtils.GetSlicerITKReadWriteAddress(mergedLabelmapNode.GetName()))
        mergedLabelmapNode.GetImageData().Modified()
        mergedLabelmapNode.Modified()

        # Update segmentation from labelmap node and remove temporary nodes
        slicer.vtkSlicerSegmentationsModuleLogic.ImportLabelmapToSegmentationNode(mergedLabelmapNode, segmentationNode, visibleSegmentIds)
        slicer.mrmlScene.RemoveNode(sourceVolumeNode)
        slicer.mrmlScene.RemoveNode(mergedLabelmapNode)

        qt.QApplication.restoreOverrideCursor()
