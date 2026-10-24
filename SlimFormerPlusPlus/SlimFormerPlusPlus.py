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

# Sliding-window step as a fraction of the patch size: 20% overlap of neighbouring patches.
STEP_SIZE = 0.8

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

# Step messages of predict.py and the status shown for them (patch progress is reported separately).
PREDICTOR_STEPS = {
    "Preprocessing ...": "Preprocessing ...",
    "Predicting ...": "Segmenting ...",
    "Resampling to the original image grid ...": "Resampling to the original image grid ...",
}


class InstallError(Exception):
    def __init__(self, message, restartRequired=False):
        super().__init__(message)
        self.restartRequired = restartRequired


class SegmentationCancelled(Exception):
    pass


#
# SlimFormerPlusPlus
#


class SlimFormerPlusPlus(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SlimFormer++"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Yang Hong (University of Exeter)"]
        self.parent.helpText = (
            "Automatic segmentation of 15 abdominal organs in CT images with SlimFormer++ "
            "(Mamba-free implementation, trained on AMOS2022 CT). "
            "<b>Standard</b> mode works on 1.5 mm voxels and is the most accurate. <b>Fast</b> mode works on 3 mm "
            "voxels; it is several times faster and less accurate, mostly on small organs. "
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
        self._running = False
        self._startTime = 0.0
        self._status = ""

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = SlimFormerPlusPlusLogic()
        self.logic.logCallback = self.addLog
        self.logic.statusCallback = self.setStatus

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

        # One mode per model in models.json (Standard: 1.5 mm, Fast: 3 mm).
        self.modeCombo = qt.QComboBox()
        for key, model in self.logic.models.items():
            self.modeCombo.addItem(f"{model['mode']} ({model['spacing_mm']:g} mm voxels)", key)
        self.modeCombo.setToolTip("Standard: most accurate (default). Fast: several times faster, less accurate, "
                                  "mostly on small organs.")
        form.addRow("Mode:", self.modeCombo)

        self.applyButton = qt.QPushButton("Segment")
        self.applyButton.setMinimumHeight(32)
        form.addRow(self.applyButton)

        # Current step and elapsed time, shown while segmenting.
        self.statusLabel = qt.QLabel()
        self.statusLabel.visible = False
        form.addRow(self.statusLabel)

        self.progressBar = qt.QProgressBar()
        self.progressBar.setTextVisible(False)
        self.progressBar.visible = False
        form.addRow(self.progressBar)

        disclaimer = qt.QLabel(DISCLAIMER)
        disclaimer.wordWrap = True
        font = disclaimer.font
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * 0.9)
            disclaimer.setFont(font)
        form.addRow(disclaimer)

        # Advanced
        advancedBox = ctk.ctkCollapsibleButton()
        advancedBox.text = "Advanced"
        advancedBox.collapsed = True
        self.layout.addWidget(advancedBox)
        advanced = qt.QFormLayout(advancedBox)

        self.deviceCombo = qt.QComboBox()
        self.deviceCombo.addItem("Automatic (GPU if available)", "auto")
        self.deviceCombo.addItem("CPU", "cpu")
        self.deviceCombo.setToolTip("Automatic uses an NVIDIA GPU (CUDA) if PyTorch finds one, otherwise the CPU.")
        advanced.addRow("Device:", self.deviceCombo)

        self.threadsSpin = qt.QSpinBox()
        self.threadsSpin.setRange(0, 256)
        self.threadsSpin.setSpecialValueText("automatic")
        self.threadsSpin.setToolTip("Number of CPU threads used on the CPU device.")
        advanced.addRow("CPU threads:", self.threadsSpin)

        self.cropBodyCheck = qt.QCheckBox()
        self.cropBodyCheck.checked = True
        self.cropBodyCheck.setToolTip("Segment only the bounding box of the patient's body, not the air and table "
                                      "around it: about 40% fewer patches on typical CT scans, about the same accuracy.")
        advanced.addRow("Crop to body:", self.cropBodyCheck)

        self.show3DCheck = qt.QCheckBox()
        self.show3DCheck.checked = True
        advanced.addRow("Show in 3D:", self.show3DCheck)

        modelRow = qt.QHBoxLayout()
        self.modelStatusLabel = qt.QLabel()
        modelRow.addWidget(self.modelStatusLabel, 1)
        self.installZipButton = qt.QPushButton("Install from file...")
        self.installZipButton.toolTip = ("Install the network weights from a downloaded model .zip file "
                                         "(for computers without internet access).")
        modelRow.addWidget(self.installZipButton)
        advanced.addRow("Model:", modelRow)

        # Log
        self.logBox = ctk.ctkCollapsibleButton()
        self.logBox.text = "Log"
        self.logBox.collapsed = True
        self.layout.addWidget(self.logBox)
        logLayout = qt.QVBoxLayout(self.logBox)
        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumHeight(220)
        logLayout.addWidget(self.statusText)

        self.layout.addStretch(1)

        self.elapsedTimer = qt.QTimer()
        self.elapsedTimer.setInterval(1000)
        self.elapsedTimer.connect("timeout()", self._showStatus)

        self.applyButton.connect("clicked(bool)", self.onApply)
        self.installZipButton.connect("clicked(bool)", self.onInstallModelFromFile)
        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateButtons)
        self.modeCombo.connect("currentIndexChanged(int)", self.updateModelStatus)
        self.restoreSettings()
        self._setRunning(False)
        self.updateModelStatus()

    def cleanup(self):
        if self._running:
            self.logic.cancel()
        self.elapsedTimer.stop()

    def restoreSettings(self):
        settings = qt.QSettings()
        index = self.deviceCombo.findData(settings.value("SlimFormerPlusPlus/Device", "auto"))
        self.deviceCombo.currentIndex = max(index, 0)
        self.modeCombo.currentIndex = max(self.modeCombo.findData(settings.value("SlimFormerPlusPlus/Model", "")), 0)
        self.cropBodyCheck.checked = str(settings.value("SlimFormerPlusPlus/CropBody", "true")).lower() == "true"
        self.threadsSpin.value = int(settings.value("SlimFormerPlusPlus/Threads", 0))

    def saveSettings(self):
        settings = qt.QSettings()
        settings.setValue("SlimFormerPlusPlus/Device", self._device())
        settings.setValue("SlimFormerPlusPlus/Model", self._modelKey())
        settings.setValue("SlimFormerPlusPlus/CropBody", "true" if self.cropBodyCheck.checked else "false")
        settings.setValue("SlimFormerPlusPlus/Threads", self.threadsSpin.value)

    def _device(self):
        return self.deviceCombo.itemData(self.deviceCombo.currentIndex)

    def _modelKey(self):
        return self.modeCombo.itemData(self.modeCombo.currentIndex)

    def updateButtons(self, *unused):
        self.applyButton.enabled = self._running or self.inputSelector.currentNode() is not None

    def updateModelStatus(self, *unused):
        key = self._modelKey()
        model = self.logic.models[key]
        if self.logic.isModelInstalled(key):
            self.modelStatusLabel.text = f"{model['mode']}: installed"
            self.modelStatusLabel.toolTip = f"{model['title']}\n{self.logic.modelDir(key)}"
        else:
            self.modelStatusLabel.text = f"{model['mode']}: not installed yet"
            self.modelStatusLabel.toolTip = f"{model['title']}\nDownloaded when it is first used."

    def addLog(self, text):
        logging.info(text)
        self.statusText.appendPlainText(text)
        slicer.app.processEvents()

    def setStatus(self, text, fraction=None, cancellable=True):
        """Show the current step; a busy indicator if ``fraction`` is None."""
        self._status = text
        self.applyButton.enabled = cancellable and not self.logic.cancelRequested
        if fraction is None:
            self.progressBar.setRange(0, 0)
        else:
            self.progressBar.setRange(0, 1000)
            self.progressBar.setValue(int(round(1000 * fraction)))
        self._showStatus()
        slicer.app.processEvents()

    def _showStatus(self):
        minutes, seconds = divmod(int(time.time() - self._startTime), 60)
        self.statusLabel.text = f"{self._status}   {minutes}:{seconds:02d}"

    def _setRunning(self, running):
        self._running = running
        self.applyButton.text = "Cancel" if running else "Segment"
        self.applyButton.toolTip = "Stop the segmentation." if running else "Run SlimFormer++ on the input volume."
        for widget in (self.inputSelector, self.outputSelector, self.modeCombo, self.deviceCombo,
                       self.threadsSpin, self.cropBodyCheck, self.show3DCheck, self.installZipButton):
            widget.enabled = not running
        self.statusLabel.visible = running
        self.progressBar.visible = running
        if running:
            self._startTime = time.time()
            self.elapsedTimer.start()
        else:
            self.elapsedTimer.stop()
        self.updateButtons()

    def _handleInstallError(self, error):
        if error.restartRequired:
            if slicer.util.confirmOkCancelDisplay(
                    f"{error}\n\nRestart 3D Slicer now?", "Restart required"):
                slicer.util.restart()
        else:
            slicer.util.errorDisplay(f"Installation failed: {error}")

    def onInstallModelFromFile(self):
        path = qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(), "Select model file", "", "Model (*.zip)")
        if not path:
            return
        try:
            with slicer.util.WaitCursor():
                key, sha256 = self.logic.modelKeyForZip(path)
                self.logic.installModelFromZip(key, path, sha256)
            self.modeCombo.currentIndex = self.modeCombo.findData(key)
            self.updateModelStatus()
        except Exception as e:
            logging.exception("Model installation failed")
            slicer.util.errorDisplay(f"Model installation failed: {e}")

    def onApply(self):
        if self._running:
            self.logic.cancel()
            self.setStatus("Cancelling ...", cancellable=False)
            return
        self.saveSettings()
        inputVolume = self.inputSelector.currentNode()
        outputSegmentation = self.outputSelector.currentNode()
        createdOutput = outputSegmentation is None
        if createdOutput:
            outputSegmentation = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", f"{inputVolume.GetName()} SlimFormer++")
            self.outputSelector.setCurrentNode(outputSegmentation)
        self.statusText.clear()
        self.logic.cancelRequested = False
        self._setRunning(True)
        error = None
        try:
            self.setStatus("Checking Python packages ...")
            self.logic.setupPythonRequirements()
            self.logic.process(inputVolume, outputSegmentation,
                               device=self._device(),
                               modelKey=self._modelKey(),
                               cropBody=self.cropBodyCheck.checked,
                               threads=self.threadsSpin.value)
            if self.show3DCheck.checked:
                self.setStatus("Creating 3D surfaces ...", cancellable=False)
                outputSegmentation.CreateClosedSurfaceRepresentation()
        except SegmentationCancelled:
            self.addLog("Segmentation cancelled.")
        except Exception as e:
            if not isinstance(e, InstallError):
                logging.exception("SlimFormer++ segmentation failed")
            error = e
        finally:
            self._setRunning(False)
            self.updateModelStatus()
        if createdOutput and outputSegmentation.GetSegmentation().GetNumberOfSegments() == 0:
            slicer.mrmlScene.RemoveNode(outputSegmentation)
        if isinstance(error, InstallError):
            self._handleInstallError(error)
        elif error is not None:
            self.logBox.collapsed = False
            slicer.util.errorDisplay(f"Segmentation failed: {error}\n\nSee the log for details.")


#
# SlimFormerPlusPlusLogic
#


class SlimFormerPlusPlusLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.models = json.loads((LIB_DIR / "models.json").read_text())
        self.logCallback = None
        self.statusCallback = None
        self.cancelRequested = False

    def log(self, text):
        if self.logCallback:
            self.logCallback(text)
        else:
            logging.info(text)

    def status(self, text, fraction=None, cancellable=True):
        """Report the current step, its completed fraction if known, and whether it can be cancelled."""
        if self.statusCallback:
            self.statusCallback(text, fraction, cancellable)

    def cancel(self):
        """Stop a running download or segmentation at the next opportunity."""
        self.cancelRequested = True

    def _checkCancel(self):
        if self.cancelRequested:
            raise SegmentationCancelled()

    # Models

    def defaultModelKey(self):
        return next(iter(self.models))

    def modelDir(self, key):
        return Path(slicer.app.cachePath) / "SlimFormerPlusPlus" / f"{key}-v{self.models[key]['version']}"

    def checkpointName(self, key):
        return self.models[key].get("checkpoint", "checkpoint_final.pth")

    def isModelInstalled(self, key):
        d = self.modelDir(key)
        return all((d / f).is_file() for f in ("plans.json", "dataset.json", f"fold_0/{self.checkpointName(key)}"))

    @staticmethod
    def _sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def modelKeyForZip(self, zipPath):
        """The models.json key whose checksum matches the archive, and the archive's SHA256."""
        digest = self._sha256(zipPath)
        for key, model in self.models.items():
            if model["sha256"] == digest:
                return key, digest
        raise ValueError("This file is not one of the model files of this version of the module: "
                         + ", ".join(f"{m['mode']} ({m['title']})" for m in self.models.values()))

    def installModelFromZip(self, key, zipPath, expectedSha256=None):
        """Verify and unpack a model archive (plans.json, dataset.json, fold_0/<checkpoint>, SHA256SUMS)."""
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
                               "Use 'Install from file...' in the Advanced section.")
        self.log(f"Downloading model weights from {model['url']} ...")
        self.status("Downloading the model weights ...")
        downloadDir = Path(slicer.util.tempDirectory())
        zipPath = downloadDir / "model.zip"
        try:
            try:
                with urllib.request.urlopen(model["url"]) as response, open(zipPath, "wb") as out:
                    total = int(response.headers.get("Content-Length") or 0)
                    done = 0
                    while True:
                        self._checkCancel()
                        block = response.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
                        done += len(block)
                        if total:
                            self.status("Downloading the model weights ...", done / total)
            except OSError as e:  # urllib's URLError/HTTPError are OSErrors
                raise RuntimeError(
                    f"Could not download the model weights from {model['url']} ({e}). Check the internet "
                    "connection, or download the file with a web browser and use 'Install from file...' "
                    "in the Advanced section.") from e
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
        self.status("Installing Python packages (this may take several minutes) ...", cancellable=False)
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

    def process(self, inputVolume, outputSegmentation, device="auto", threads=0, modelKey=None,
                stepSize=STEP_SIZE, cropBody=True):
        import numpy as np

        modelKey = modelKey or self.defaultModelKey()
        if not self.isModelInstalled(modelKey):
            self.downloadModel(modelKey)
        modelDir = self.modelDir(modelKey)
        self.log(f"Mode: {self.models[modelKey]['mode']} ({self.models[modelKey]['title']})")
        start = time.time()

        tempDir = Path(slicer.util.tempDirectory())
        try:
            self.status("Exporting the input volume ...")
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
            self._checkCancel()

            python = shutil.which("PythonSlicer")
            if not python:
                raise RuntimeError("PythonSlicer executable was not found")
            cmd = [python, str(LIB_DIR / "predict.py"), "--model-dir", str(modelDir),
                   "--checkpoint", self.checkpointName(modelKey),
                   "--input", str(inputPath), "--output", str(outputPath), "--device", device,
                   "--step-size", str(stepSize)]
            if not cropBody:
                cmd.append("--no-crop-body")
            if threads:
                cmd += ["--threads", str(threads)]
            # nnU-Net warns when these are unset; they are not used for prediction.
            env = {name: str(tempDir) for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")}
            env["nnUNet_compile"] = "false"
            self.status("Loading the network ...")
            self.log("Running SlimFormer++ ...")
            logging.debug(f"Command: {cmd}")
            self._followProcess(self._launch(cmd, env), outputPath)

            self.status("Importing the segments ...", cancellable=False)
            labels = np.load(outputPath)["labels"]
            self._importLabels(labels, inputVolume, outputSegmentation, modelDir)
        finally:
            shutil.rmtree(tempDir, ignore_errors=True)
            self.cancelRequested = False
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

    @staticmethod
    def _launch(cmd, env):
        """slicer.util.launchConsoleProcess in a new process group, so that cancelling also stops the Python
        process that the PythonSlicer launcher starts."""
        import subprocess
        environment = slicer.util.startupEnvironment()
        environment.update(env)
        options = dict(env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        if os.name == "nt":
            info = subprocess.STARTUPINFO()
            info.dwFlags = subprocess.STARTF_USESHOWWINDOW
            info.wShowWindow = 0  # hide the console window
            return subprocess.Popen(cmd, startupinfo=info, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, **options)
        return subprocess.Popen(cmd, start_new_session=True, **options)

    @staticmethod
    def _stop(proc):
        """Stop the process and its children."""
        import signal
        import subprocess
        if proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        proc.wait()

    def _followProcess(self, proc, outputPath):
        """Forward the process output to the log and status, keeping the GUI responsive while waiting.

        Raises SegmentationCancelled if cancelled, RuntimeError if the process fails.
        """
        import queue
        import threading
        lines = queue.Queue()

        def read():  # in a thread: readline blocks until the process writes a line
            while True:
                try:
                    line = proc.stdout.readline()
                except UnicodeDecodeError:
                    continue
                except ValueError:  # stdout closed
                    break
                if not line:
                    break
                lines.put(line)
            lines.put(None)

        threading.Thread(target=read, daemon=True).start()
        finished = False
        lastLog = 0.0
        try:
            while True:
                self._checkCancel()
                try:
                    line = lines.get(timeout=0.05)
                except queue.Empty:
                    slicer.app.processEvents()
                    continue
                if line is None:
                    break
                line = line.rstrip()
                if line.startswith("PROGRESS "):
                    done, total = (int(v) for v in line.split()[1:3])
                    self.status(f"Segmenting: {done} / {total} patches", done / max(total, 1))
                    if time.time() - lastLog > 10 or done == total:
                        self.log(f"  sliding window: {done}/{total} patches")
                        lastLog = time.time()
                    continue
                self.log(line)
                if line in PREDICTOR_STEPS:
                    self.status(PREDICTOR_STEPS[line])
                if line.startswith("Done in"):
                    finished = True
                    # On macOS a finished Python process can keep stdout open; do not wait for it.
                    break
        except SegmentationCancelled:
            self._stop(proc)
            raise
        if finished:
            try:
                proc.wait(timeout=30)
            except Exception:
                self._stop(proc)
        else:
            proc.wait()
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
        self.assertEqual([m["mode"] for m in logic.models.values()], ["Standard", "Fast"])
        self.assertEqual(logic.models[logic.defaultModelKey()]["mode"], "Standard")
        for key, model in logic.models.items():
            for field in ("url", "sha256", "checkpoint", "spacing_mm"):
                self.assertIn(field, model)
            self.assertTrue(str(logic.modelDir(key)).endswith(f"{key}-v{model['version']}"))
        self.delayDisplay("Model registry test passed")
