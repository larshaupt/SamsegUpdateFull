# %%
import numpy as np
import pickle
import sys
sys.path.append("/scratch/users/lhaupt/ext/freesurfer_build/freesurfer/python/packages")
import os
import surfa as sf
import importlib
import freesurfer.samseg
import importlib
importlib.reload(freesurfer.samseg)
import pdb
imageFileNamesList = [['/scratch/users/lhaupt/data/NCANDA/MRIs/NCANDA_S00118_baseline_brain.nii.gz'],
 ['/scratch/users/lhaupt/data/NCANDA/MRIs/NCANDA_S00118_followup_1y_brain.nii.gz'],
 ['/scratch/users/lhaupt/data/NCANDA/MRIs/NCANDA_S00118_followup_2y_brain.nii.gz'],
 ['/scratch/users/lhaupt/data/NCANDA/MRIs/NCANDA_S00118_followup_3y_brain.nii.gz'],
 ['/scratch/users/lhaupt/data/NCANDA/MRIs/NCANDA_S00118_followup_4y_brain.nii.gz'],
 ['/scratch/users/lhaupt/data/NCANDA/MRIs/NCANDA_S00118_followup_5y_brain.nii.gz'],
 ['/scratch/users/lhaupt/data/NCANDA/MRIs/NCANDA_S00118_followup_6y_brain.nii.gz'],
 ['/scratch/users/lhaupt/data/NCANDA/MRIs/NCANDA_S00118_followup_7y_brain.nii.gz'],
 ['/scratch/users/lhaupt/data/NCANDA/MRIs/NCANDA_S00118_followup_8y_brain.nii.gz']]
updateImageFileNamesList = imageFileNamesList[2:]
imageYears = [1,2,3,4,5]
defaultAtlas = '20Subjects_smoothing2_down2_smoothingForAffine2'
atlasDir = os.path.join(os.environ.get('FREESURFER_HOME'), 'average', 'samseg', defaultAtlas)
savePath = "/scratch/users/lhaupt/data/NCANDA/labels_exp/NCANDA_S00118_n"
probabilisticAtlas = freesurfer.samseg.ProbabilisticAtlas()
# %%

s = freesurfer.samseg.SamsegLongitudinalUpdate(
    imageFileNamesList = imageFileNamesList[:4],
    updateFileNamesList = imageFileNamesList[2:4],
    atlasDir = atlasDir,
    savePath = savePath,
    numberOfIterations=1,
    saveHistory=True,
    saveMesh=True
    )
#%%
s.saveHistory = True

s.constructAndRegisterSubjectSpecificTemplate()
s.constructSstModel()
s.preProcess()
s.loadProcessedFiles()


# %%
s.sstModel.optimizationOptions["multiResolutionSpecification"][0]["maximumNumberOfIterations"] = 1
s.sstModel.optimizationOptions["multiResolutionSpecification"][1]["maximumNumberOfIterations"] = 1
s.fitModel()
s.postProcess()

# What do we need to load?
# All elements from s.processedFileNamesList
#
# timepointmodel.deformation

# from .io import kvlReadCompressionLookupTable, kvlReadSharedGMMParameters
# kvlReadSharedGMMParameters
# timepointmodel.gmm.mean
# timepointmode.gmm.variance
# timepointmodel.gmm.mixtureWeights
#deformation = probabilisticAtlas.getMesh("/scratch/users/lhaupt/data/NCANDA/labels_samseg/NCANDA_S00057/latentAtlases/latentAtlas_iteration_01.mgz.gz")
#gmm = kvlReadSharedGMMParameters("/scratch/users/lhaupt/ext/freesurfer_build/freesurfer/average/samseg/20Subjects_smoothing2_down2_smoothingForAffine2/exvivo.lh.sharedGMMParameters.txt")

    
    
# s.latentDeformation \/ saved
# s.sstModel.biasField.getBiasFields( s.sstModel.mask) x not saved
# s.sstModel.mask x not saved
# s.latentDeformationAtlasFileName = s.sstModel.deformationAtlasFileName
# s.latentMeans = s.sstModel.gmm.means.copy() x not saved
# s.latentVariances = s.sstModel.gmm.variances.copy() x not saved
# s.latentMixtureWeights = s.sstModel.gmm.mixtureWeights.copy() x not saved
# s.sstModel.gmm.numberOfGaussiansPerClass 
# s.sstModel.modelSpecifications.K

# %%

# What do we need to load?
# All elements from s.processedFileNamesList
#
# timepointmodel.deformation
# timepointmodel.gmm.mean
# timepointmode.gmm.variance
# timepointmodel.gmm.mixtureWeights
#   --> doesnt seem to be saved
#
# s.latentDeformation
# s.sstModel.biasField.getBiasFields( s.sstModel.mask)
# s.sstModel.mask
# s.latentDeformation = s.sstModel.deformation.copy()
# s.latentDeformationAtlasFileName = s.sstModel.deformationAtlasFileName
# s.latentMeans = s.sstModel.gmm.means.copy()
# s.latentVariances = s.sstModel.gmm.variances.copy()
# s.latentMixtureWeights = s.sstModel.gmm.mixtureWeights.copy()
# s.sstModel.gmm.numberOfGaussiansPerClass
# s.sstModel.modelSpecifications.K

# easiest solution: save all these things in a pickle file and load them
finalStateDir = os.path.join(s.savePath, 'finalState.p')
with open(finalStateDir, "rb") as infile:
    finalState = pickle.load(infile)

s.sstModel.initializeGMM()
s.sstModel.initializeBiasField()
s.sstModel.gmm.means = finalState["latentMeans"]
s.sstModel.gmm.variances = finalState["latentVariances"]
s.sstModel.gmm.mixtureWeights = finalState["latentMixtureWeights"]
s.sstModel.biasField.coefficients = finalState["sstBiasFieldCoefficients"]
s.sstModel.deformation = finalState["latentDeformation"]
s.sstModel.deformationAtlasFileName = finalState["latentDeformationAtlasFileName"]
s.sstModel.optimizationSummary = finalState["sstOptimizationSummary"]
s.sstModel.optimizationHistory = finalState["sstOptimizationHistory"]
s.sstModel.modelSpecifications.FreeSurferLabels = finalState["labels"]
s.sstModel.modelSpecifications.names = finalState["names"]
s.sstModel.numberOfGaussiansPerClass = finalState["sstNumberOfGaussiansPerClass"]
s.timepointVolumesInCubicMm = finalState["timepointVolumesInCubicMm"]
s.optimizationSummary = finalState["optimizationSummary"]


#%%
# s.latentMeans = finalState["latentMeans"]
# s.latentVariances = finalState["latentVariances"]
# s.latentMixtureWeights = finalState["latentMixtureWeights"]
# s.latentDeformation = finalState["latentDeformation"]
# s.latentDeformationAtlasFileName = finalState["latentDeformationAtlasFileName"]

for timepointModel in finalState["timepointModel"]:
    # find the matching timpoints
    timepointNumbers = [i for i in range(s.numberOfTimepoints) if s.imageFileNamesList[i] == timepointModel["timepointFileName"]]
    assert len(timepointNumbers) < 2
    # if the file is matching our file, then we add it to the list
    # if the file does not match our files, we discard it
    # the file should not match multiple files
    if len(timepointNumbers) == 0 or timepointModel["timepointFileName"] not in s.processedFileNamesList:
        continue

    timepointNumber = timepointNumbers[0]

    s.timepointNumberProcessed.append(timepointNumber)

    s.timepointModels[timepointNumber].initializeGMM()
    s.timepointModels[timepointNumber].gmm.means = timepointModel["timepointMeans"]
    s.timepointModels[timepointNumber].gmm.variances = timepointModel["timepointVariances"]
    s.timepointModels[timepointNumber].gmm.mixtureWeights = timepointModel["timepointMixtureWeights"]
    s.timepointModels[timepointNumber].deformation = timepointModel["timepointDeformations"]
    s.timepointModels[timepointNumber].deformationAtlasFileName = timepointModel["timepointDeformationAtlasFileNames"]

s.latentInitialized = True

s.timpointNumberUpdate = [t for t in range(s.numberOfTimepoints) if t not in s.timepointNumberProcessed]
    
# %%
