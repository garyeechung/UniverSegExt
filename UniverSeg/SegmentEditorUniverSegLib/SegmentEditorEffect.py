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
        self._support_dir = None

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

        # Select Directory button
        self.selectDirectoryButton = qt.QPushButton("Select Support Set Directory")
        self.selectDirectoryButton.objectName = self.__class__.__name__ + 'Select'
        self.selectDirectoryButton.setToolTip("Select a directory")
        self.scriptedEffect.addOptionsWidget(self.selectDirectoryButton)
        self.selectDirectoryButton.connect('clicked()', self.onSelect)

        # Upload File button
        self.uploadButton = qt.QPushButton("Upload File")
        self.uploadButton.objectName = self.__class__.__name__ + 'Upload'
        self.uploadButton.setToolTip("Upload a file")
        self.scriptedEffect.addOptionsWidget(self.uploadButton)
        self.uploadButton.connect('clicked()', self.onUpload)
    
    # Since images and masks should be paired, it is better to select a directory
    def onSelect(self):
        logging.info("Select button clicked")
        directory = qt.QFileDialog.getExistingDirectory(None, "Select Directory")
        if directory:
            logging.info(directory)
            self.selectDirectoryButton.setText(directory)
            self._support_dir = directory

    def onUpload(self):
        logging.info("Upload button clicked")
        options = qt.QFileDialog.Options()
        options |= qt.QFileDialog.DontUseNativeDialog
        fileName, _ = qt.QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            logging.info(fileName)
            self.uploadButton.setText(fileName)
            # TODO: Add code here to handle the file. For example, you could read its content
            # and perform some operation.
            # with open(fileName, 'r') as f:
            #     print(f.read())
    def activate(self):
        # Nothing to do here
        pass

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
        from torchvision import transforms
        from PIL import Image
        import numpy as np
        import glob
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.warning(f"{os.listdir()}")
        logging.warning(f"{device}")

        def resize_and_scale(image_np):
            # Convert NumPy array to PIL Image
            image_pil = Image.fromarray(image_np)

            # Resize the image
            image_pil = image_pil.resize((128, 128))

            # Convert back to NumPy array
            resized_image_np = np.array(image_pil)

            # Scale pixel values to [0, 1]
            scaled_image = resized_image_np.astype(np.float32) / 255.0

            return scaled_image

        # Read input data from Slicer into SimpleITK
        # The labelImage is the segment layer that we added in Segment Editor in Slicer
        # Currenrly should be an empty mask, will be replace with predicted mask later.
        labelImage = sitk.ReadImage(sitkUtils.GetSlicerITKReadWriteAddress(mergedLabelmapNode.GetName()))
        backgroundImage = sitk.ReadImage(sitkUtils.GetSlicerITKReadWriteAddress(sourceVolumeNode.GetName()))
        # print(sourceVolumeNode.GetName())
        # print(segmentationNode.GetName())

        # Convert SimpleITK.Image instance to numpy.Array
        # lab = sitk.GetArrayFromImage(labelImage)
        target_image = sitk.GetArrayFromImage(backgroundImage)
        # print(target_image.shape)
        target_image = resize_and_scale(target_image.reshape(target_image.shape[1],-1))
        # print(target_image.shape)

        target_image = torch.from_numpy(target_image)
        target_image = target_image.unsqueeze(0)
        target_image = target_image.to(device)
        print(target_image.shape)

        # Read and transform the support images
        def process_image(image_path):
            # Load image
            image = Image.open(image_path).convert('RGB')
            print(np.array(image).shape)

            # Check if the image is grayscale or RGB
            if image.mode == 'RGB':
                # Define a transformation pipeline for RGB images
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),  # Resize to 128x128
                    transforms.Grayscale(),         # Convert to grayscale
                    transforms.ToTensor(),          # Convert to tensor
                ])
            else:
                # Define a transformation pipeline for grayscale images
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),  # Resize to 128x128
                    transforms.Grayscale(),         # Convert to grayscale
                    transforms.ToTensor(),          # Convert to tensor
                ])

            # Apply transformations
            tensor_image = transform(image)
            print(tensor_image.shape)
            return tensor_image
        
        support_images = []
        support_labels = []
        for support_img in os.listdir(self._support_dir):
            support_images.append(process_image(os.path.join(self._support_dir, support_img, 'img.png')))
            support_labels.append(process_image(os.path.join(self._support_dir, support_img, 'seg.png')))
        print(support_images[0].shape)
        print(support_labels[0].shape)
        support_images = torch.stack(support_images).to(device)
        support_labels = torch.stack(support_labels).to(device)
        print(support_images.shape)
        # According to the official repo: https://github.com/JJGO/UniverSeg
        # TODO: resize img to (B, 1, 128, 128)
        # TODO: Normalize values to [0, 1]

        # TODO: instantiate UniverSeg model
        model = universeg(pretrained=True)
        model.to(device)

        # TODO: example dataset or user's own data, psudo code as follow
        # NOTE: failed to import example_data
        # from SegmentEditorUniverSegLib import wbc
        # d_support = wbc.WBCDataset('JTSC', split='support', label='cytoplasm')
        # logging.warning(d_support)

        # if example:
        #     use example dataset like oasis or wbc
        # else:
        #     read uploaded data

        prediction = model(
            target_image[None],        # (B, 1, H, W)
            support_images[None],      # (B, S, 1, H, W)
            support_labels[None],      # (B, S, 1, H, W)
        ) # -> (B, 1, H, W)
        print('done')
        print(prediction)

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
