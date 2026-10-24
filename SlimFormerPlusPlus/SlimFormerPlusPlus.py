# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import ctk
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

MODULE_DIR = Path(__file__).resolve().parent
LIB_DIR = MODULE_DIR / "SlimFormerPlusPlusLib"
MINIMUM_NNUNET = "2.6.2"
DISCLAIMER = "For research use only. Not for clinical diagnosis or treatment."

# Display colors (Slicer GenericAnatomyColors) by label name in the model's dataset.json.
SEGMENT_COLORS = {
    "spleen": (157, 108, 162),
    "right kidney": (185, 102, 83),
    "left kidney": (185, 102, 83),
    "gallbladder": (139, 150, 98),
    "esophagus": (211, 171, 143),
    "liver": (221, 130, 101),
    "stomach": (216, 132, 105),
    "aorta": (224, 97, 76),
    "inferior vena cava": (0, 151, 206),
    "pancreas": (249, 180, 111),
    "right adrenal gland": (244, 214, 49),
    "left adrenal gland": (244, 214, 49),
    "duodenum": (255, 253, 229),
    "bladder": (222, 154, 132),
    "prostate/uterus": (230, 158, 140),
}


class InstallError(Exception):
    def __init__(self, message, restartRequired=False):
        super().__init__(message)
        self.restartRequired = restartRequired


#
# SlimFormerPlusPlus
#


class SlimFormerPlusPlus(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SlimFormer++"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["WHHHHHY822"]
        self.parent.helpText = (
            "Automatic segmentation of 15 abdominal organs in CT images with SlimFormer++ "
            "(Mamba-free implementation, trained on AMOS2022 CT). "
            "The first run installs PyTorch and nnU-Net into Slicer and downloads the network weights.<br>"
            f"<b>{DISCLAIMER}</b><br>"
            'See the <a href="https://github.com/WHHHHHY822/SlicerSlimFormerPlusPlus">documentation</a>.'
        )
        self.parent.acknowledgementText = (
            "Network: SlimFormer++ (https://github.com/WHHHHHY822/SlimFormerPlusPlus). "
            "Pre- and post-processing: nnU-Net v2 (Isensee et al., Nature Methods 2021). "
            "Training data: AMOS2022 (Ji et al., NeurIPS 2022). "
            "Network weights: SlimFormer++ Noncommercial Research License 1.0."
        )


#
# SlimFormerPlusPlusWidget
#


class SlimFormerPlusPlusWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None
        self._waitCursor = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = SlimFormerPlusPlusLogic()
        self.logic.logCallback = self.addLog
        self.logic.progressCallback = self.setProgress

        # Inputs and outputs
        ioBox = ctk.ctkCollapsibleButton()
        ioBox.text = "Segmentation"
        self.layout.addWidget(ioBox)
        form = qt.QFormLayout(ioBox)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("CT volume to segment (abdomen).")
        form.addRow("Input CT volume:", self.inputSelector)

        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.outputSelector.addEnabled = True
        self.outputSelector.renameEnabled = True
        self.outputSelector.removeEnabled = True
        self.outputSelector.noneEnabled = True
        self.outputSelector.noneDisplay = "(Create new segmentation)"
        self.outputSelector.setMRMLScene(slicer.mrmlScene)
        self.outputSelector.setToolTip("Existing segments of this segmentation are replaced.")
        form.addRow("Output segmentation:", self.outputSelector)

        self.applyButton = qt.QPushButton("Segment")
        self.applyButton.toolTip = "Run SlimFormer++ on the input volume."
        form.addRow(self.applyButton)

        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.visible = False
        form.addRow(self.progressBar)

        disclaimer = qt.QLabel(DISCLAIMER)
        disclaimer.wordWrap = True
        disclaimer.setStyleSheet("font-weight: bold")
        form.addRow(disclaimer)

        # Advanced
        advancedBox = ctk.ctkCollapsibleButton()
        advancedBox.text = "Advanced"
        advancedBox.collapsed = True
        self.layout.addWidget(advancedBox)
        advanced = qt.QFormLayout(advancedBox)

        self.deviceCombo = qt.QComboBox()
        self.deviceCombo.addItem("Automatic (GPU if available)", "auto")
        self.deviceCombo.addItem("GPU (CUDA)", "cuda")
        self.deviceCombo.addItem("CPU", "cpu")
        self.deviceCombo.setToolTip("Computation device. CPU works everywhere but is slow.")
        advanced.addRow("Device:", self.deviceCombo)

        self.speedCombo = qt.QComboBox()
        self.speedCombo.addItem("Standard (50% patch overlap)", 0.5)
        self.speedCombo.addItem("Faster (25% patch overlap)", 0.75)
        self.speedCombo.setToolTip("Faster uses about 2.5 times fewer patches, at a small accuracy cost. "
                                   "Useful on the CPU.")
        advanced.addRow("Speed:", self.speedCombo)

        self.cropBodyCheck = qt.QCheckBox()
        self.cropBodyCheck.checked = True
        self.cropBodyCheck.setToolTip("Segment only the bounding box of the patient's body, not the air and table "
                                      "around it: about 40% fewer patches on typical CT scans, about the same accuracy.")
        advanced.addRow("Crop to body:", self.cropBodyCheck)

        self.ttaCheck = qt.QCheckBox()
        self.ttaCheck.setToolTip("Mirroring test-time augmentation: about 8 times slower, slightly smoother results.")
        advanced.addRow("Test-time augmentation:", self.ttaCheck)

        self.threadsSpin = qt.QSpinBox()
        self.threadsSpin.setRange(0, 256)
        self.threadsSpin.setSpecialValueText("automatic")
        self.threadsSpin.setToolTip("Number of CPU threads used on the CPU device.")
        advanced.addRow("CPU threads:", self.threadsSpin)

        self.show3DCheck = qt.QCheckBox()
        self.show3DCheck.checked = True
        advanced.addRow("Show in 3D:", self.show3DCheck)

        self.keepTempCheck = qt.QCheckBox()
        self.keepTempCheck.setToolTip("Keep the temporary input/output files (for troubleshooting).")
        advanced.addRow("Keep temporary files:", self.keepTempCheck)

        self.modelStatusLabel = qt.QLabel()
        self.modelStatusLabel.wordWrap = True
        advanced.addRow("Model:", self.modelStatusLabel)

        buttons = qt.QHBoxLayout()
        self.installDepsButton = qt.QPushButton("Install dependencies")
        self.installDepsButton.toolTip = "Install PyTorch and nnU-Net into Slicer's Python environment."
        buttons.addWidget(self.installDepsButton)
        self.installZipButton = qt.QPushButton("Install model from file...")
        self.installZipButton.toolTip = "Install the network weights from a downloaded model .zip file."
        buttons.addWidget(self.installZipButton)
        advanced.addRow(buttons)

        # Log
        logBox = ctk.ctkCollapsibleButton()
        logBox.text = "Log"
        self.layout.addWidget(logBox)
        logLayout = qt.QVBoxLayout(logBox)
        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumHeight(220)
        logLayout.addWidget(self.statusText)

        self.layout.addStretch(1)

        self.applyButton.connect("clicked(bool)", self.onApply)
        self.installDepsButton.connect("clicked(bool)", self.onInstallDependencies)
        self.installZipButton.connect("clicked(bool)", self.onInstallModelFromFile)
        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateButtons)
        self.restoreSettings()
        self.updateButtons()
        self.updateModelStatus()

    def restoreSettings(self):
        settings = qt.QSettings()
        index = self.deviceCombo.findData(settings.value("SlimFormerPlusPlus/Device", "auto"))
        self.deviceCombo.currentIndex = max(index, 0)
        self.ttaCheck.checked = str(settings.value("SlimFormerPlusPlus/TTA", "false")).lower() == "true"
        self.speedCombo.currentIndex = max(self.speedCombo.findData(float(settings.value("SlimFormerPlusPlus/StepSize", 0.5))), 0)
        self.cropBodyCheck.checked = str(settings.value("SlimFormerPlusPlus/CropBody", "true")).lower() == "true"
        self.threadsSpin.value = int(settings.value("SlimFormerPlusPlus/Threads", 0))

    def saveSettings(self):
        settings = qt.QSettings()
        settings.setValue("SlimFormerPlusPlus/Device", self._device())
        settings.setValue("SlimFormerPlusPlus/TTA", "true" if self.ttaCheck.checked else "false")
        settings.setValue("SlimFormerPlusPlus/StepSize", self._stepSize())
        settings.setValue("SlimFormerPlusPlus/CropBody", "true" if self.cropBodyCheck.checked else "false")
        settings.setValue("SlimFormerPlusPlus/Threads", self.threadsSpin.value)

    def _device(self):
        return self.deviceCombo.itemData(self.deviceCombo.currentIndex)

    def _stepSize(self):
        return float(self.speedCombo.itemData(self.speedCombo.currentIndex))

    def updateButtons(self, *unused):
        self.applyButton.enabled = self.inputSelector.currentNode() is not None

    def updateModelStatus(self):
        key = self.logic.defaultModelKey()
        if self.logic.isModelInstalled(key):
            self.modelStatusLabel.text = f"{self.logic.models[key]['title']}: installed in {self.logic.modelDir(key)}"
        else:
            self.modelStatusLabel.text = f"{self.logic.models[key]['title']}: not installed yet (downloaded on first use)"

    def addLog(self, text):
        logging.info(text)
        self.statusText.appendPlainText(text)
        slicer.app.processEvents()

    def setProgress(self, fraction):
        self.progressBar.visible = fraction is not None
        if fraction is not None:
            self.progressBar.setValue(int(round(100 * fraction)))
        slicer.app.processEvents()

    def _busy(self, busy):
        self.applyButton.enabled = not busy and self.inputSelector.currentNode() is not None
        self.installDepsButton.enabled = not busy
        self.installZipButton.enabled = not busy
        if busy and not self._waitCursor:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        elif not busy and self._waitCursor:
            qt.QApplication.restoreOverrideCursor()
        self._waitCursor = busy

    def _handleInstallError(self, error):
        if error.restartRequired:
            if slicer.util.confirmOkCancelDisplay(
                    f"{error}\n\nRestart 3D Slicer now?", "Restart required"):
                slicer.util.restart()
        else:
            slicer.util.errorDisplay(f"Installation failed: {error}")

    def onInstallDependencies(self):
        self._busy(True)
        try:
            self.logic.setupPythonRequirements()
            self.addLog("Dependencies are installed.")
        except InstallError as e:
            self._busy(False)
            self._handleInstallError(e)
            return
        except Exception as e:
            logging.exception("Dependency installation failed")
            slicer.util.errorDisplay(f"Installation failed: {e}")
        finally:
            self._busy(False)

    def onInstallModelFromFile(self):
        path = qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(), "Select model file", "", "Model (*.zip)")
        if not path:
            return
        self._busy(True)
        try:
            self.logic.installModelFromZip(self.logic.defaultModelKey(), path)
            self.updateModelStatus()
        except Exception as e:
            logging.exception("Model installation failed")
            slicer.util.errorDisplay(f"Model installation failed: {e}")
        finally:
            self._busy(False)

    def onApply(self):
        self.saveSettings()
        inputVolume = self.inputSelector.currentNode()
        outputSegmentation = self.outputSelector.currentNode()
        if outputSegmentation is None:
            outputSegmentation = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", f"{inputVolume.GetName()} SlimFormer++")
            self.outputSelector.setCurrentNode(outputSegmentation)
        self.statusText.clear()
        self._busy(True)
        try:
            self.logic.setupPythonRequirements()
            self.logic.process(inputVolume, outputSegmentation,
                               device=self._device(),
                               tta=self.ttaCheck.checked,
                               stepSize=self._stepSize(),
                               cropBody=self.cropBodyCheck.checked,
                               threads=self.threadsSpin.value,
                               keepTemporaryFiles=self.keepTempCheck.checked)
            if self.show3DCheck.checked:
                outputSegmentation.CreateClosedSurfaceRepresentation()
            self.updateModelStatus()
        except InstallError as e:
            self._busy(False)
            self._handleInstallError(e)
        except Exception as e:
            logging.exception("SlimFormer++ segmentation failed")
            slicer.util.errorDisplay(f"Segmentation failed: {e}\n\nSee the log for details.")
        finally:
            self._busy(False)
            self.setProgress(None)


#
# SlimFormerPlusPlusLogic
#


class SlimFormerPlusPlusLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.models = json.loads((LIB_DIR / "models.json").read_text())
        self.logCallback = None
        self.progressCallback = None

    def log(self, text):
        if self.logCallback:
            self.logCallback(text)
        else:
            logging.info(text)

    def progress(self, fraction):
        if self.progressCallback:
            self.progressCallback(fraction)

    # Models

    def defaultModelKey(self):
        return next(iter(self.models))

    def modelDir(self, key):
        return Path(slicer.app.cachePath) / "SlimFormerPlusPlus" / f"{key}-v{self.models[key]['version']}"

    def isModelInstalled(self, key):
        d = self.modelDir(key)
        return all((d / f).is_file() for f in ("plans.json", "dataset.json", "fold_0/checkpoint_final.pth"))

    @staticmethod
    def _sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def installModelFromZip(self, key, zipPath, expectedSha256=None):
        """Verify and unpack a model archive (plans.json, dataset.json, fold_0/checkpoint_final.pth, SHA256SUMS)."""
        if expectedSha256 is None and self.models[key]["sha256"] not in ("", "TO_BE_DECIDED"):
            expectedSha256 = self.models[key]["sha256"]
        if expectedSha256 and self._sha256(zipPath) != expectedSha256:
            raise ValueError("The model file does not match the expected checksum (corrupted or wrong file).")
        target = self.modelDir(key)
        staging = target.with_name(target.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        with zipfile.ZipFile(zipPath) as z:
            names = z.namelist()
            if any(n.startswith("/") or ".." in Path(n).parts for n in names):
                raise ValueError("Unsafe paths in model archive")
            z.extractall(staging)
        sums = staging / "SHA256SUMS"
        if not sums.is_file():
            raise ValueError("SHA256SUMS is missing from the model archive")
        for line in sums.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            if self._sha256(staging / name) != digest:
                raise ValueError(f"Checksum mismatch for {name} in the model archive")
        shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
        if not self.isModelInstalled(key):
            raise ValueError("The model archive does not contain a complete model")
        self.log(f"Model installed in {target}")

    def downloadModel(self, key):
        import urllib.request
        model = self.models[key]
        if model["url"] in ("", "TO_BE_DECIDED"):
            raise RuntimeError("No download address is configured for this model yet. "
                               "Use 'Install model from file...' in the Advanced section.")
        self.log(f"Downloading model weights from {model['url']} ...")
        downloadDir = Path(slicer.util.tempDirectory())
        zipPath = downloadDir / "model.zip"
        try:
            try:
                with urllib.request.urlopen(model["url"]) as response, open(zipPath, "wb") as out:
                    total = int(response.headers.get("Content-Length") or 0)
                    done = 0
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
                        done += len(block)
                        if total:
                            self.progress(done / total)
            except OSError as e:  # urllib's URLError/HTTPError are OSErrors
                raise RuntimeError(
                    f"Could not download the model weights from {model['url']} ({e}). Check the internet "
                    "connection, or download the file with a web browser and use 'Install model from file...' "
                    "in the Advanced section.") from e
            finally:
                self.progress(None)
            self.installModelFromZip(key, zipPath, model["sha256"])
        finally:
            shutil.rmtree(downloadDir, ignore_errors=True)

    # Python dependencies

    def setupPythonRequirements(self):
        """PyTorch through the PyTorch extension, nnU-Net through the NNUNet extension."""
        try:
            import PyTorchUtils  # noqa: F401
        except ModuleNotFoundError:
            raise InstallError("This module needs the 'PyTorch' extension. "
                               "Install it in the Extensions Manager and restart Slicer.")
        try:
            import SlicerNNUNetLib
        except ModuleNotFoundError:
            raise InstallError("This module needs the 'NNUNet' extension. "
                               "Install it in the Extensions Manager and restart Slicer.")
        from packaging.requirements import Requirement

        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        nnunetLogic = SlicerNNUNetLib.InstallLogic(doAskConfirmation=False)
        requirement = f"nnunetv2>={MINIMUM_NNUNET}"
        missing = []
        if not torchLogic.torchInstalled():
            missing.append("PyTorch")
        if not nnunetLogic.isPackageInstalledAndCompatible(Requirement(requirement)):
            missing.append("nnU-Net v2")
        if not missing:
            return
        if not slicer.util.confirmOkCancelDisplay(
                "SlimFormer++ needs additional Python packages. Installation needs a network connection "
                "and may take several minutes. Click OK to proceed.",
                "Install Python packages", detailedText="Packages: " + ", ".join(missing)):
            raise InstallError("Installation was cancelled.")
        nnunetLogic.progressInfo.connect(self.log)
        if "nnU-Net v2" in missing:
            # Also installs a PyTorch build that matches the GPU driver, through the PyTorch extension.
            if not nnunetLogic.setupPythonRequirements(requirement):
                raise InstallError("Installing nnU-Net failed. See the log for details.")
        if not torchLogic.torchInstalled():
            self.log("Installing PyTorch (this may take several minutes) ...")
            if torchLogic.installTorch(askConfirmation=False, torchVersionRequirement=">=2.1.2") is None:
                raise InstallError("Installing PyTorch failed. Use the 'PyTorch Utils' module to install it.")
        if nnunetLogic.needsRestart:
            raise InstallError("Python packages were installed. 3D Slicer must be restarted before "
                               "segmenting.", restartRequired=True)

    # Segmentation

    def process(self, inputVolume, outputSegmentation, device="auto", tta=False, threads=0,
                keepTemporaryFiles=False, modelKey=None, stepSize=0.5, cropBody=True):
        import numpy as np

        modelKey = modelKey or self.defaultModelKey()
        if not self.isModelInstalled(modelKey):
            self.downloadModel(modelKey)
        modelDir = self.modelDir(modelKey)
        start = time.time()

        tempDir = Path(slicer.util.tempDirectory())
        try:
            inputPath, outputPath = tempDir / "input.npz", tempDir / "output.npz"
            image = slicer.util.arrayFromVolume(inputVolume)
            self.log(f"Input: {inputVolume.GetName()}, {image.shape[::-1]} voxels, "
                     f"spacing {tuple(round(s, 3) for s in inputVolume.GetSpacing())} mm")
            if image.ndim != 3:
                raise ValueError("The input must be a single-component 3D volume")
            if image.min() >= 0:
                self.log("Warning: the image has no negative values, so it is probably not in Hounsfield units. "
                         "The model expects CT intensities in HU; results may be poor.")
            np.savez(inputPath, image=image, ijk_to_ras=self.ijkToWorld(inputVolume))

            python = shutil.which("PythonSlicer")
            if not python:
                raise RuntimeError("PythonSlicer executable was not found")
            cmd = [python, str(LIB_DIR / "predict.py"), "--model-dir", str(modelDir),
                   "--input", str(inputPath), "--output", str(outputPath), "--device", device]
            if tta:
                cmd.append("--tta")
            if stepSize != 0.5:
                cmd += ["--step-size", str(stepSize)]
            if not cropBody:
                cmd.append("--no-crop-body")
            if threads:
                cmd += ["--threads", str(threads)]
            # nnU-Net warns when these are unset; they are not used for prediction.
            env = {name: str(tempDir) for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")}
            env["nnUNet_compile"] = "false"
            self.log("Running SlimFormer++ ...")
            logging.debug(f"Command: {cmd}")
            proc = slicer.util.launchConsoleProcess(cmd, updateEnvironment=env)
            self._followProcess(proc, outputPath)

            labels = np.load(outputPath)["labels"]
            self._importLabels(labels, inputVolume, outputSegmentation, modelDir)
        finally:
            if keepTemporaryFiles:
                self.log(f"Temporary files kept in {tempDir}")
            else:
                shutil.rmtree(tempDir, ignore_errors=True)
        self.log(f"Segmentation finished in {time.time() - start:.0f} s")

    @staticmethod
    def ijkToWorld(volumeNode):
        """IJK-to-RAS matrix including a linear parent transform."""
        import numpy as np
        m = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(m)
        ijkToRas = slicer.util.arrayFromVTKMatrix(m)
        transformNode = volumeNode.GetParentTransformNode()
        if transformNode is not None and transformNode.IsTransformToWorldLinear():
            toWorld = vtk.vtkMatrix4x4()
            transformNode.GetMatrixTransformToWorld(toWorld)
            ijkToRas = slicer.util.arrayFromVTKMatrix(toWorld) @ ijkToRas
        return np.asarray(ijkToRas, dtype=float)

    def _followProcess(self, proc, outputPath):
        """Forward the process output to the log and progress bar; raise if it fails."""
        finished = False
        lastLog = 0.0
        while True:
            try:
                line = proc.stdout.readline()
            except UnicodeDecodeError:
                continue
            if not line:
                break
            line = line.rstrip()
            if line.startswith("PROGRESS "):
                done, total = (int(v) for v in line.split()[1:3])
                self.progress(done / max(total, 1))
                if time.time() - lastLog > 10 or done == total:
                    self.log(f"  sliding window: {done}/{total} patches")
                    lastLog = time.time()
                else:
                    slicer.app.processEvents()
                continue
            self.log(line)
            if line.startswith("Done in"):
                finished = True
                # On macOS a finished Python process can keep stdout open; do not wait for it.
                break
        if finished:
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
        else:
            proc.wait()
        self.progress(None)
        if not (finished and outputPath.is_file()):
            raise RuntimeError(f"SlimFormer++ inference failed (exit code {proc.returncode}). See the log.")

    def _importLabels(self, labels, inputVolume, segmentationNode, modelDir):
        datasetJson = json.loads((Path(modelDir) / "dataset.json").read_text())
        names = {int(v): k for k, v in datasetJson["labels"].items() if int(v) > 0}

        colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", "SlimFormerPlusPlusColors")
        colorNode.SetTypeToUser()
        colorNode.SetNumberOfColors(max(names) + 1)
        colorNode.SetColor(0, "background", 0, 0, 0, 0)
        for value, name in names.items():
            r, g, b = SEGMENT_COLORS.get(name, (128, 128, 128))
            colorNode.SetColor(value, name[:1].upper() + name[1:], r / 255, g / 255, b / 255, 1.0)

        labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "SlimFormerPlusPlusLabels")
        try:
            labelmap.CopyOrientation(inputVolume)
            labelmap.SetAndObserveTransformNodeID(inputVolume.GetTransformNodeID())
            slicer.util.updateVolumeFromArray(labelmap, labels)
            labelmap.CreateDefaultDisplayNodes()
            labelmap.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())
            segmentationNode.GetSegmentation().RemoveAllSegments()
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
            segmentationNode.SetAndObserveTransformNodeID(inputVolume.GetTransformNodeID())
            if not slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap, segmentationNode):
                raise RuntimeError("Importing the labels into the segmentation failed")
        finally:
            slicer.mrmlScene.RemoveNode(labelmap)
            slicer.mrmlScene.RemoveNode(colorNode)
        found = segmentationNode.GetSegmentation().GetNumberOfSegments()
        self.log(f"{found} of {len(names)} structures found")


#
# SlimFormerPlusPlusTest
#


class SlimFormerPlusPlusTest(ScriptedLoadableModuleTest):
    """Fast tests that need neither network access nor the network weights."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_reorientation_round_trip()
        self.test_model_registry()

    def test_reorientation_round_trip(self):
        import numpy as np
        sys.path.insert(0, str(MODULE_DIR))
        from SlimFormerPlusPlusLib.reorient import from_model_axes, to_model_axes
        array = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
        lps_like = np.diag([-0.8, -0.8, 2.5, 1.0])     # j runs posterior, as in most DICOM series
        model, spacing, mapping = to_model_axes(array, lps_like)
        self.assertEqual(mapping["flip"], (False, True, False))
        self.assertTrue(np.array_equal(from_model_axes(model, mapping), array))
        self.delayDisplay("Reorientation round trip passed")

    def test_model_registry(self):
        logic = SlimFormerPlusPlusLogic()
        key = logic.defaultModelKey()
        self.assertIn("url", logic.models[key])
        self.assertTrue(str(logic.modelDir(key)).endswith(f"{key}-v{logic.models[key]['version']}"))
        self.delayDisplay("Model registry test passed")
