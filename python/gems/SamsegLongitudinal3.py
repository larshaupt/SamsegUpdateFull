import os
import numpy as np
import pickle

from .SamsegUtility import *
from .utilities import requireNumpyArray
from .figures import initVisualizer
from .Affine import Affine
from .ProbabilisticAtlas import ProbabilisticAtlas
from .Samseg import Samseg
from .SubjectSpecificAtlas import SubjectSpecificAtlas
from . import gems
import pdb


"""
Longitudinal version of samsegment
The idea is based on the generative model in the paper

  Iglesias, Juan Eugenio, et al.
  Bayesian longitudinal segmentation of hippocampal substructures in brain MRI using subject-specific atlases.
  Neuroimage 141 (2016): 542-555,
in which a subject-specific atlas is obtained by generating a random warp from the usual population atlas, and
subsequently each time point is again randomly warped from this subject-specific atlas. The intermediate
subject-specific atlas is effectively a latent variable in the model, and it's function is to encourage the
various time points to have atlas warps that are similar between themselves, without having to define a priori
what these warps should be similar to. In the implementation provided here, the stiffness of the first warp
(denoted by K0, as in the paper) is taken as the stiffness used in ordinary samseg, and the stiffness of the
second warp (denoted by K1) is controlled by the setting

  strengthOfLatentDeformationHyperprior

so that K1 = strengthOfLatentDeformationHyperprior * K0. In the Iglesias paper, the setting

  strengthOfLatentDeformationHyperprior = 1.0

was used.

The overall idea is extended here by adding also latent variables encouraging corresponding Gaussian mixture
models across time points to be similar across time -- again without having to define a priori what exactly
they should look like. For given values of these latent variables, they effectively act has hyperparameters
on the mixture model parameters, the strength of which is controlled through the setting

  strengthOfLatentGMMHyperprior

which weighs the relative strength of this hyperprior relative to the relevant (expected) data term in the
estimation procedures of the mixture model parameters at each time point. This aspect can be switched off by
setting

  strengthOfLatentGMMHyperprior = 0.0

NOTE: The general longitudinal pipeline in FreeSurfer 6.0 is described in the paper

  Reuter, Martin, et al.
  Within-subject template estimation for unbiased longitudinal image analysis.
  Neuroimage 61.4 (2012): 1402-1418,

which is based on the idea of retaining some temporal consistency by simply initializing the model fitting
procedure across all time points in exactly the same way. This is achieved by first creating a subject-specific
template that is subsequently analyzed and the result of which is then used as a (very good) initialization
in each time point. This behavior can be mimicked by setting

  initializeLatentDeformationToZero = True
  numberOfIterations = 1
  strengthOfLatentGMMHyperprior = 0.0
  strengthOfLatentDeformationHyperprior = 1.0

in the implementation provided here.
"""

eps = np.finfo( float ).eps

class SamsegLongitudinal3:
    def __init__(self,
        imageFileNamesList,
        imageTimePoints,
        atlasDir,
        savePath,
        hiddenState = None,
        userModelSpecifications={},
        userOptimizationOptions={},
        visualizer=None, 
        saveHistory=False,
        saveMesh=None,
        targetIntensity=None,
        targetSearchStrings=None,
        numberOfIterations=5,
        strengthOfLatentGMMHyperprior=0.5,
        strengthOfLatentDeformationHyperprior=20.0,
        saveSSTResults=True,
        updateLatentMeans=True,
        updateLatentVariances=True,
        updateLatentMixtureWeights=True,
        updateLatentDeformation=True,
        initializeLatentDeformationToZero=False,
        threshold=None,
        n_neighbors:int = 1,
        thresholdSearchString=None,
        modeNames=None,
        pallidumAsWM=True,
        savePosteriors=False
        ):

        # Store input parameters as class variables
        self.imageFileNamesList = imageFileNamesList
        self.imageTimePoints = imageTimePoints
        self.hiddenState = hiddenState #should be a mesh like atlas
        self.numberOfTimepoints = len(self.imageFileNamesList)
        self.savePath = savePath
        self.atlasDir = atlasDir
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.thresholdSearchString = thresholdSearchString
        self.targetIntensity = targetIntensity
        self.targetSearchStrings = targetSearchStrings
        self.numberOfIterations = numberOfIterations
        self.strengthOfLatentGMMHyperprior = strengthOfLatentGMMHyperprior
        self.strengthOfLatentDeformationHyperprior = strengthOfLatentDeformationHyperprior
        self.modeNames = modeNames
        self.pallidumAsWM = pallidumAsWM
        self.savePosteriors = savePosteriors
        self.latentAtlasFileName = [None] * self.numberOfTimepoints

        # Initialize some objects
        self.probabilisticAtlas = ProbabilisticAtlas()

        # Get full model specifications and optimization options (using default unless overridden by user)
        self.userModelSpecifications = userModelSpecifications
        self.userOptimizationOptions = userOptimizationOptions

        # Setup a null visualizer if necessary
        if visualizer is None:
            self.visualizer = initVisualizer(False, False)
        else:
            self.visualizer = visualizer

        self.saveHistory = saveHistory
        self.saveMesh = saveMesh
        self.saveSSTResults = saveSSTResults
        self.updateLatentMeans = updateLatentMeans
        self.updateLatentVariances = updateLatentVariances
        self.updateLatentMixtureWeights = updateLatentMixtureWeights
        self.updateLatentDeformation = updateLatentDeformation
        self.initializeLatentDeformationToZero = initializeLatentDeformationToZero

        # Make sure we can write in the target/results directory
        os.makedirs(savePath, exist_ok=True)

        # Here some class variables that will be defined later
        self.sstModel = None
        self.timepointModels = None
        self.imageBuffersList = None
        self.sstFileNames = None
        self.combinedImageBuffers = None
        self.latentDeformation = [None] * self.numberOfTimepoints
        self.latentDeformationAtlasFileName = [None] * self.numberOfTimepoints
        self.latentMeans = [None] * self.numberOfTimepoints
        self.latentVariances = [None] * self.numberOfTimepoints
        self.latentMixtureWeights = [None] * self.numberOfTimepoints
        self.latentMeansNumberOfMeasurements = None
        self.latentVariancesNumberOfMeasurements = None
        self.latentMixtureWeightsNumberOfMeasurements = None
        self.timepointVolumesInCubicMm = []
        self.optimizationSummary = None
        self.history = None
        self.historyOfTotalCost = None
        self.historyOfTotalTimepointCost = None
        self.historyOfLatentAtlasCost = None
        # New code

        if len(self.imageTimePoints) != self.numberOfTimepoints:
            print(f"Did not provide the correct amount of weights: {self.imageTimePoints}. It should have the length {len(self.imageFileNamesList)}")
            self.imageTimePoints = [1]*self.numberOfTimepoints

        if not all([isinstance(el, float) or isinstance(el,int) for el in self.imageTimePoints]):
            if not all([el.replace(".", "").isnumeric() for el in self.imageTimePoints]):
                print(f"Did not provide the correct weights: {self.imageTimePoints}. It should be floats")
                self.imageTimePoints = [1]*self.numberOfTimepoints
            else:
                self.imageTimePoints = [float(el) for el in self.imageTimePoints]
        self.RBFLength = 1
        self.RBFScale = 1.1
        self.RBFintercept = 0.1
        def rbf_kernel(a,b):
            return np.exp(-np.square(a-b)/self.RbfLength) * self.RBFScale + self.RBFintercept

        self.weightKernel = rbf_kernel

    def segment(self, saveWarp=False):
        # =======================================================================================
        #
        # Main function that runs the whole longitudinal segmentation pipeline
        #
        # =======================================================================================
        self.sstDir = sstDir = os.path.join(self.savePath, 'base')

        self.sstModelObject = SubjectSpecificAtlas(
            imageFileNamesList = self.imageFileNamesList,
            atlasDir = self.atlasDir,
            sstDir = self.sstDir,
            userModelSpecifications = self.userModelSpecifications,
            userOptimizationOptions = self.userOptimizationOptions,
            visualizer = self.visualizer,
            targetIntensity = self.targetIntensity,
            targetSearchStrings = self.targetSearchStrings,
            modeNames = self.modeNames,
            pallidumAsWM = self.pallidumAsWM,
            saveHistory = self.saveHistory
        )

        self.sstModelObject.fitModel()
        self.sstModel = self.sstModelObject.getSubjectSpecificTemplate()
        self.imageToImageTransformMatrix = self.sstModelObject.imageToImageTransformMatrix
        self.imageBuffersList = self.sstModelObject.imageBuffersList
        if self.saveHistory:
            self.history.update(self.sstModelObject.history)
        self.preProcess()
        self.fitModel()
        return self.postProcess(saveWarp=saveWarp)

    def preProcess(self):

        # =======================================================================================
        #
        # Preprocessing (reading and masking of data)
        #
        # =======================================================================================                                                                  (timepointNumber + 2) * numberOfContrasts]

        # construct timepoint models
        self.constructTimepointModels()


    def fitModel(self):


        # =======================================================================================
        #
        # Iterative parameter vs. latent variables estimation, using SST result for initialization
        # and/or anchoring of hyperprior strength
        #
        # =======================================================================================

        # Initialization of the time-specific model parameters
        for timepointNumber in range(self.numberOfTimepoints):
            self.timepointModels[timepointNumber].initializeGMM()
            self.timepointModels[timepointNumber].gmm.means = self.sstModelObject.sstModel.gmm.means.copy()
            self.timepointModels[timepointNumber].gmm.variances = self.sstModelObject.sstModel.gmm.variances.copy()
            self.timepointModels[timepointNumber].gmm.mixtureWeights = self.sstModelObject.sstModel.gmm.mixtureWeights.copy()
            self.timepointModels[timepointNumber].initializeBiasField()

            # Initialization of the latent variables, acting as hyperparameters when viewed from the model parameters' perspective
            
            self.latentDeformation[timepointNumber] = self.sstModelObject.sstModel.deformation.copy()
            self.latentDeformationAtlasFileName[timepointNumber] = self.sstModelObject.sstModel.deformationAtlasFileName
            self.latentMeans[timepointNumber] = self.sstModelObject.sstModel.gmm.means.copy()
            self.latentVariances[timepointNumber] = self.sstModelObject.sstModel.gmm.variances.copy()
            self.latentMixtureWeights[timepointNumber] = self.sstModelObject.sstModel.gmm.mixtureWeights.copy()

        if self.initializeLatentDeformationToZero:
            for timepointNumber in range(self.numberOfTimepoints):
                self.timepointModels[timepointNumber].deformation = self.latentDeformation.copy()
                self.timepointModels[timepointNumber].latentDeformationAtlasFileName = self.latentDeformationAtlasFileName
                self.latentDeformation[timepointNumber] = 0

        # Strength of the hyperprior (i.e., how much the latent variables control the conditional posterior of the parameters)
        # is user-controlled.
        #
        # For the GMM part, I'm using the *average* number of voxels assigned to the components in each mixture (class) of the
        # SST segmentation, so that all the components in each mixture are well-regularized (and tiny components don't get to do
        # whatever they want)
        # TAG: 
        K0 = self.sstModelObject.sstModel.modelSpecifications.K  # Stiffness population -> latent position
        K1 = self.strengthOfLatentDeformationHyperprior * K0  # Stiffness latent position -> each time point
        sstEstimatedNumberOfVoxelsPerGaussian = np.sum(self.sstModel.optimizationHistory[-1]['posteriorsAtEnd'], axis=0) * \
                                                np.prod(self.sstModel.optimizationHistory[-1]['downSamplingFactors'])
        numberOfClasses = len(self.sstModelObject.sstModel.gmm.numberOfGaussiansPerClass)
        numberOfGaussians = sum(self.sstModelObject.sstModel.gmm.numberOfGaussiansPerClass)
        self.latentMeansNumberOfMeasurements = np.zeros(numberOfGaussians)
        self.latentVariancesNumberOfMeasurements = np.zeros(numberOfGaussians)
        self.latentMixtureWeightsNumberOfMeasurements = np.zeros(numberOfClasses)
        for classNumber in range(numberOfClasses):
            #
            numberOfComponents = self.sstModel.gmm.numberOfGaussiansPerClass[classNumber]
            gaussianNumbers = np.array(np.sum(self.sstModel.gmm.numberOfGaussiansPerClass[:classNumber]) +
                                       np.array(range(numberOfComponents)), dtype=np.uint32)
            sstEstimatedNumberOfVoxelsInClass = np.sum(sstEstimatedNumberOfVoxelsPerGaussian[gaussianNumbers])

            self.latentMixtureWeightsNumberOfMeasurements[
                classNumber] = self.strengthOfLatentGMMHyperprior * sstEstimatedNumberOfVoxelsInClass

            averageSizeOfComponents = sstEstimatedNumberOfVoxelsInClass / numberOfComponents
            self.latentMeansNumberOfMeasurements[gaussianNumbers] = self.strengthOfLatentGMMHyperprior * averageSizeOfComponents
            self.latentVariancesNumberOfMeasurements[gaussianNumbers] = self.strengthOfLatentGMMHyperprior * averageSizeOfComponents

        # Estimating the mode of the latentVariance posterior distribution (which is Wishart) requires a stringent condition
        # on latentVariancesNumberOfMeasurements so that the mode is actually defined
        threshold = (self.sstModel.gmm.numberOfContrasts + 2) + 1e-6
        self.latentVariancesNumberOfMeasurements[self.latentVariancesNumberOfMeasurements < threshold] = threshold

        # No point in updating latent GMM parameters if the GMM hyperprior has zero weight. The latent variances are also
        # a bit tricky, as they're technically driven to zero in that scenario -- let's try not to go there...
        if self.strengthOfLatentGMMHyperprior == 0:
            self.updateLatentMeans, self.updateLatentVariances, self.updateLatentMixtureWeights = False, False, False

        # Loop over all iterations
        self.historyOfTotalCost, self.historyOfTotalTimepointCost, self.historyOfLatentAtlasCost = [], [], []
        progressPlot = None
        iterationNumber = 0
        if self.saveHistory:
            self.history = {**self.history,
                       **{
                           "timepointMeansEvolution": [],
                           "timepointVariancesEvolution": [],
                           "timepointMixtureWeightsEvolution": [],
                           "timepointBiasFieldCoefficientsEvolution": [],
                           "timepointDeformationsEvolution": [],
                           "timepointDeformationAtlasFileNamesEvolution": [],
                           "latentMeansEvolution": [],
                           "latentVariancesEvolution": [],
                           "latentMixtureWeightsEvolution": [],
                           "latentDeformationEvolution": [],
                           "latentDeformationAtlasFileNameEvolution": []
                       }
                       }

        # Make latent atlas directory
        latentAtlasDirectory = os.path.join(self.savePath, 'latentAtlases')
        os.makedirs(latentAtlasDirectory, exist_ok=True)

        while True:
            
            # =======================================================================================
            #
            # Update parameters for each time point using the current latent variable estimates
            #
            # =======================================================================================
            totalTimepointCost = 0
            for timepointNumber in range(self.numberOfTimepoints):
                # Create a new atlas that will be the basis to deform the individual time points from
                self.latentAtlasFileName[timepointNumber] = os.path.join(latentAtlasDirectory, 'latentAtlas_%02d_iteration_%02d.mgz' % (timepointNumber, iterationNumber + 1))
                self.probabilisticAtlas.saveDeformedAtlas(self.latentDeformationAtlasFileName[timepointNumber], self.latentAtlasFileName[timepointNumber], self.latentDeformation[timepointNumber], True)

                # Only use the last resolution level, and with the newly created atlas as atlas
                self.timepointModels[timepointNumber].optimizationOptions = self.sstModel.optimizationOptions
                self.timepointModels[timepointNumber].optimizationOptions['multiResolutionSpecification'] = [
                    self.timepointModels[timepointNumber].optimizationOptions['multiResolutionSpecification'][-1]]
                self.timepointModels[timepointNumber].optimizationOptions['multiResolutionSpecification'][0]['atlasFileName'] = self.latentAtlasFileName[timepointNumber]
                print(self.timepointModels[timepointNumber].optimizationOptions)

                # Loop over all time points
                
                # TODO add the specific weights here
                self.timepointModels[timepointNumber].modelSpecifications.K = K1 
                self.timepointModels[timepointNumber].gmm.hyperMeans = self.latentMeans[timepointNumber]
                self.timepointModels[timepointNumber].gmm.hyperVariances = self.latentVariances[timepointNumber]
                self.timepointModels[timepointNumber].gmm.hyperMixtureWeights = self.latentMixtureWeights[timepointNumber]
                self.timepointModels[timepointNumber].gmm.fullHyperMeansNumberOfMeasurements = self.latentMeansNumberOfMeasurements.copy()
                self.timepointModels[timepointNumber].gmm.fullHyperVariancesNumberOfMeasurements = self.latentVariancesNumberOfMeasurements.copy()
                self.timepointModels[timepointNumber].gmm.fullHyperMixtureWeightsNumberOfMeasurements = self.latentMixtureWeightsNumberOfMeasurements.copy()
                # This is where the optimization is done
                self.timepointModels[timepointNumber].estimateModelParameters(
                                            initialBiasFieldCoefficients=self.timepointModels[timepointNumber].biasField.coefficients,
                                            initialDeformation=self.timepointModels[timepointNumber].deformation,
                                            initialDeformationAtlasFileName=self.timepointModels[timepointNumber].deformationAtlasFileName,
                                            skipBiasFieldParameterEstimationInFirstIteration=False,
                                            skipGMMParameterEstimationInFirstIteration=(iterationNumber == 0)
                                            )

                totalTimepointCost += self.timepointModels[timepointNumber].optimizationHistory[-1]['historyOfCost'][-1]

                print('=================================')
                print('\n')
                print('timepointNumber: ', timepointNumber)
                print('perVoxelCost: ', self.timepointModels[timepointNumber].optimizationSummary[-1]['perVoxelCost'])
                print('\n')
                print('=================================')
                if hasattr(self.visualizer, 'show_flag'):
                    import matplotlib.pyplot as plt  # avoid importing matplotlib by default
                    plt.ion()
                    self.timepointModels[timepointNumber].biasField.downSampleBasisFunctions([1, 1, 1])
                    timepointBiasFields = self.timepointModels[timepointNumber].biasField.getBiasFields(self.sstModel.mask)
                    timepointData = self.imageBuffersList[timepointNumber][self.sstModel.mask, :] - timepointBiasFields[self.sstModel.mask, :]
                    for contrastNumber in range(self.sstModel.gmm.numberOfContrasts):
                        axs = axsList[contrastNumber]
                        ax = axs.ravel()[2 + timepointNumber]
                        ax.clear()
                        ax.hist(timepointData[:, contrastNumber], bins)
                        ax.grid()
                        ax.set_title('time point ' + str(timepointNumber))
                    plt.draw()

                    # End loop over time points

                # =======================================================================================
                #
                # Check for convergence.
                # =======================================================================================

                # In order to also measure the deformation from the population atlas -> latent position,
                # create:
                #   (1) a mesh collection with as reference position the population reference position, and as positions
                #       the currently estimated time point positions.
                #   (2) a mesh with the current latent position
                # Note that in (1) we don't need those time positions now, but these will come in handy very soon to
                # optimize the latent position
                #
                # The parameter estimation happens in a (potentially) downsampled image grid, so it's import to work in the same space
                # when measuring and updating the latentDeformation
                #pdb.set_trace()

            for timepointNumber in range(self.numberOfTimepoints):

                transformUsedForEstimation = gems.KvlTransform(
                    requireNumpyArray(self.sstModelObject.sstModel.optimizationHistory[-1]['downSampledTransformMatrix']))
                mesh_collection = gems.KvlMeshCollection()
                mesh_collection.read(self.latentDeformationAtlasFileName[timepointNumber])
                mesh_collection.transform(transformUsedForEstimation)
                referencePosition = mesh_collection.reference_position

                # TODO add hidden state to mesh_collection
                #pdb.set_trace()
                timepointPositions = []
                TimepointForLatentNumbers = [timepointNumber]

                if timepointNumber-1 >= 0:
                    TimepointForLatentNumbers.append(timepointNumber-1)
                if timepointNumber+1 < self.numberOfTimepoints:
                    TimepointForLatentNumbers.append(timepointNumber+1)
                timePointWeights = [self.weightKernel(self.imageTimePoints[timepointNumber], self.imageTimePoints[el]) for el in TimepointForLatentNumbers]

                for timepointNumberNeighbours in TimepointForLatentNumbers:
                    positionInTemplateSpace = self.probabilisticAtlas.mapPositionsFromSubjectToTemplateSpace(referencePosition,
                                                                                    transformUsedForEstimation) + \
                                            self.latentDeformation[timepointNumber] + self.timepointModels[timepointNumberNeighbours].deformation
                    timepointPositions.append(
                        self.probabilisticAtlas.mapPositionsFromTemplateToSubjectSpace(positionInTemplateSpace, transformUsedForEstimation))
                mesh_collection.set_positions(referencePosition, timepointPositions)

                # Read mesh in sst warp
                # TODO: Fix this
                mesh = self.probabilisticAtlas.getMesh(self.latentAtlasFileName[timepointNumber], transformUsedForEstimation)

                #
                calculator = gems.KvlCostAndGradientCalculator(mesh_collection, K0, 0.0, transformUsedForEstimation)
                latentAtlasCost, _ = calculator.evaluate_mesh_position(mesh)

                #
                totalCost = totalTimepointCost + latentAtlasCost
                print('*' * 100 + '\n')
                print('iterationNumber: ', iterationNumber)
                print('totalCost: ', totalCost)
                print('   latentAtlasCost: ', latentAtlasCost)
                print('   totalTimepointCost: ', totalTimepointCost)
                print('*' * 100 + '\n')
                self.historyOfTotalCost.append(totalCost)
                self.historyOfTotalTimepointCost.append(totalTimepointCost)
                self.historyOfLatentAtlasCost.append(latentAtlasCost)

                if hasattr(self.visualizer, 'show_flag'):
                    import matplotlib.pyplot as plt  # avoid importing matplotlib by default
                    plt.ion()
                    if progressPlot is None:
                        plt.figure()
                        progressPlot = plt.subplot()
                    progressPlot.clear()
                    progressPlot.plot(self.historyOfTotalCost, color='k')
                    progressPlot.plot(self.historyOfTotalTimepointCost, linestyle='-.', color='b')
                    progressPlot.plot(self.historyOfLatentAtlasCost, linestyle='-.', color='r')
                    progressPlot.grid()
                    progressPlot.legend(['total', 'timepoints', 'latent atlas deformation'])
                    plt.draw()

                if self.saveHistory:
                    self.history["timepointMeansEvolution"].append(self.timepointModels[timepointNumber].gmm.means.copy())
                    self.history["timepointVariancesEvolution"].append(self.timepointModels[timepointNumber].gmm.variances.copy())
                    self.history["timepointMixtureWeightsEvolution"].append(self.timepointModels[timepointNumber].gmm.mixtureWeights.copy())
                    self.history["timepointBiasFieldCoefficientsEvolution"].append(self.timepointModels[timepointNumber].biasField.coefficients.copy())
                    self.history["timepointDeformationsEvolution"].append(self.timepointModels[timepointNumber].deformation)
                    self.history["timepointDeformationAtlasFileNamesEvolution"].append(self.timepointModels[timepointNumber].deformationAtlasFileName)
                    self.history["latentMeansEvolution"].append(self.latentMeans.copy())
                    self.history["latentVariancesEvolution"].append(self.latentVariances.copy())
                    self.history["latentMixtureWeightsEvolution"].append(self.latentMixtureWeights.copy())
                    self.history["latentDeformationEvolution"].append(self.latentDeformation.copy())
                    self.history["latentDeformationAtlasFileNameEvolution"].append(self.latentDeformationAtlasFileName)


                if iterationNumber >= (self.numberOfIterations - 1):
                    print('Stopping')
                    break


                # =======================================================================================
                #
                # Update the latent variables based on the current parameter estimates
                #
                # =======================================================================================
                # mesh_collection - mesh collection of all segmentation
                # mesh  -  mesh of registered probabilistic atlas

                self.updateLatentDeformationAtlas(mesh_collection, mesh, K0, timePointWeights, transformUsedForEstimation, timepointNumber)
                self.updateGMMLatentVariables(timepointNumber)

            iterationNumber += 1
            # End loop over parameter and latent variable estimation iterations

    def updateLatentDeformationAtlas(self, mesh_collection, mesh, K0, K1s, transformUsedForEstimation, latentTimePointNumber):
        # Update the latentDeformation
        if self.updateLatentDeformation:
            # Set up calculator
            calculator = gems.KvlCostAndGradientCalculator(mesh_collection, K0, K1s, transformUsedForEstimation)

            # Get optimizer and plug calculator in it
            optimizerType = 'L-BFGS'
            optimizationParameters = {
                'Verbose': self.sstModel.optimizationOptions['verbose'],
                'MaximalDeformationStopCriterion': self.sstModel.optimizationOptions['maximalDeformationStopCriterion'],
                'LineSearchMaximalDeformationIntervalStopCriterion': self.sstModel.optimizationOptions[
                    'lineSearchMaximalDeformationIntervalStopCriterion'],
                'MaximumNumberOfIterations': self.sstModel.optimizationOptions['maximumNumberOfDeformationIterations'],
                'BFGS-MaximumMemoryLength': self.sstModel.optimizationOptions['BFGSMaximumMemoryLength']
            }
            optimizer = gems.KvlOptimizer(optimizerType, mesh, calculator, optimizationParameters)

            # Run deformation optimization
            historyOfDeformationCost = []
            historyOfMaximalDeformation = []
            nodePositionsBeforeDeformation = mesh.points
            while True:
                minLogLikelihoodTimesDeformationPrior, maximalDeformation = optimizer.step_optimizer_samseg()
                print("maximalDeformation=%.4f minLogLikelihood=%.4f" % (
                    maximalDeformation, minLogLikelihoodTimesDeformationPrior))
                historyOfDeformationCost.append(minLogLikelihoodTimesDeformationPrior)
                historyOfMaximalDeformation.append(maximalDeformation)
                if maximalDeformation == 0:
                    break

            nodePositionsAfterDeformation = mesh.points
            maximalDeformationApplied = np.sqrt(
                np.max(np.sum((nodePositionsAfterDeformation - nodePositionsBeforeDeformation) ** 2, 1)))

            #
            nodePositionsBeforeDeformation = self.probabilisticAtlas.mapPositionsFromSubjectToTemplateSpace(
                nodePositionsBeforeDeformation,
                transformUsedForEstimation)
            nodePositionsAfterDeformation = self.probabilisticAtlas.mapPositionsFromSubjectToTemplateSpace(
                nodePositionsAfterDeformation,
                transformUsedForEstimation)
            estimatedUpdate = nodePositionsAfterDeformation - nodePositionsBeforeDeformation
            self.latentDeformation[latentTimePointNumber] += estimatedUpdate

    def updateGMMLatentVariables(self, latentTimePointNumber):

        # Update latentMeans
        if self.updateLatentMeans:
            numberOfGaussians = np.sum(self.sstModel.gmm.numberOfGaussiansPerClass)
            numberOfContrasts = self.latentMeans[latentTimePointNumber].shape[-1]
            for gaussianNumber in range(numberOfGaussians):
                # Set up linear system
                lhs = np.zeros((numberOfContrasts, numberOfContrasts))
                rhs = np.zeros((numberOfContrasts, 1))
                for timepointNumber in range(self.numberOfTimepoints):
                    mean = np.expand_dims(self.timepointModels[timepointNumber].gmm.means[gaussianNumber], 1)
                    variance = self.timepointModels[timepointNumber].gmm.variances[gaussianNumber]

                    lhs += np.linalg.inv(variance)
                    rhs += np.linalg.solve(variance, mean)

                # Solve linear system
                latentMean = np.linalg.solve(lhs, rhs)
                self.latentMeans[latentTimePointNumber][gaussianNumber, :] = latentMean.T

        # Update latentVariances
        if self.updateLatentVariances:
            numberOfGaussians = np.sum(self.sstModel.gmm.numberOfGaussiansPerClass)
            numberOfContrasts = self.latentMeans[latentTimePointNumber].shape[-1]
            for gaussianNumber in range(numberOfGaussians):
                # Precision is essentially averaged
                averagePrecision = np.zeros((numberOfContrasts, numberOfContrasts))
                for timepointNumber in range(self.numberOfTimepoints):
                    variance = self.timepointModels[timepointNumber].gmm.variances[gaussianNumber]

                    averagePrecision += np.linalg.inv(variance)
                averagePrecision /= self.numberOfTimepoints

                latentVarianceNumberOfMeasurements = self.latentVariancesNumberOfMeasurements[gaussianNumber]
                latentVariance = np.linalg.inv(averagePrecision) * \
                                 (latentVarianceNumberOfMeasurements - numberOfContrasts - 2) / latentVarianceNumberOfMeasurements
                self.latentVariances[latentTimePointNumber][gaussianNumber] = latentVariance

        # Update latentMixtureWeights
        if self.updateLatentMixtureWeights:
            numberOfClasses = len(self.sstModel.gmm.numberOfGaussiansPerClass)
            for classNumber in range(numberOfClasses):
                numberOfComponents = self.sstModel.gmm.numberOfGaussiansPerClass[classNumber]
                averageInLogDomain = np.zeros(numberOfComponents)
                for componentNumber in range(numberOfComponents):
                    gaussianNumber = sum(self.sstModel.gmm.numberOfGaussiansPerClass[:classNumber]) + componentNumber
                    for timepointNumber in range(self.numberOfTimepoints):
                        mixtureWeight = self.timepointModels[timepointNumber].gmm.mixtureWeights[gaussianNumber]
                        averageInLogDomain[componentNumber] += np.log(mixtureWeight + eps)
                    averageInLogDomain[componentNumber] /= self.numberOfTimepoints

                # Solution is normalized version
                solution = np.exp(averageInLogDomain)
                solution /= np.sum(solution + eps)

                #
                for componentNumber in range(numberOfComponents):
                    gaussianNumber = sum(self.sstModel.gmm.numberOfGaussiansPerClass[:classNumber]) + componentNumber
                    self.latentMixtureWeights[latentTimePointNumber][gaussianNumber] = solution[componentNumber]

    def postProcess(self, saveWarp=False):

        # =======================================================================================
        #
        # Using estimated parameters, segment and write out results for each time point
        #
        # =======================================================================================

        self.timepointVolumesInCubicMm = []
        for timepointNumber in range(self.numberOfTimepoints):
            #
            timepointModel = self.timepointModels[timepointNumber]

            #
            timepointModel.deformation = self.latentDeformation[timepointNumber] + timepointModel.deformation
            timepointModel.deformationAtlasFileName = self.latentDeformationAtlasFileName[timepointNumber]
            posteriors, biasFields, nodePositions, _, _ = timepointModel.computeFinalSegmentation()

            #
            timepointDir = os.path.join(self.savePath, 'tp%03d' % (timepointNumber + 1))
            os.makedirs(timepointDir, exist_ok=True)

            #
            timepointModel.savePath = timepointDir
            volumesInCubicMm = timepointModel.writeResults(biasFields, posteriors)

            # Save the timepoint->template warp
            if saveWarp:
                timepointModel.saveWarpField(os.path.join(timepointDir, 'template.m3z'))

            # Save the final mesh collection
            if self.saveMesh:
                print('Saving the final mesh in template space')
                deformedAtlasFileName = os.path.join(timepointModel.savePath, 'mesh.txt')
                timepointModel.probabilisticAtlas.saveDeformedAtlas(timepointModel.modelSpecifications.atlasFileName,
                                                                    deformedAtlasFileName, nodePositions)

            # Save the history of the parameter estimation process
            if self.saveHistory:
                history = {'input': {
                    'imageFileNames': timepointModel.imageFileNames,
                    'imageToImageTransformMatrix': timepointModel.imageToImageTransformMatrix,
                    'modelSpecifications': timepointModel.modelSpecifications,
                    'optimizationOptions': timepointModel.optimizationOptions,
                    'savePath': timepointModel.savePath
                }, 'imageBuffers': timepointModel.imageBuffers, 'mask': timepointModel.mask,
                    'cropping': timepointModel.cropping,
                    'transform': timepointModel.transform.as_numpy_array,
                    'historyWithinEachMultiResolutionLevel': timepointModel.optimizationHistory,
                    "labels": timepointModel.modelSpecifications.FreeSurferLabels, "names": timepointModel.modelSpecifications.names,
                    "volumesInCubicMm": volumesInCubicMm, "optimizationSummary": timepointModel.optimizationSummary}
                with open(os.path.join(timepointModel.savePath, 'history.p'), 'wb') as file:
                    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)
            

            self.timepointVolumesInCubicMm.append(volumesInCubicMm)

        #
        self.optimizationSummary = {
            "historyOfTotalCost": self.historyOfTotalCost,
            "historyOfTotalTimepointCost": self.historyOfTotalTimepointCost,
            "historyOfLatentAtlasCost": self.historyOfLatentAtlasCost
        }

        if self.saveHistory:
            self.history["labels"] = self.sstModel.modelSpecifications.FreeSurferLabels
            self.history["names"] = self.sstModel.modelSpecifications.names
            self.history["timepointVolumesInCubicMm"] = self.timepointVolumesInCubicMm
            self.history["optimizationSummary"] = self.optimizationSummary
            with open(os.path.join(self.savePath, 'history.p'), 'wb') as file:
                pickle.dump(self.history, file, protocol=pickle.HIGHEST_PROTOCOL)

    
    def constructTimepointModels(self):

        self.timepointModels = []

        # Construction of the cross sectional model for each time point
        for timepointNumber in range(self.numberOfTimepoints):
            self.timepointModels.append(Samseg(
                imageFileNames=self.imageFileNamesList[timepointNumber],
                atlasDir=self.atlasDir,
                savePath=self.savePath,
                imageToImageTransformMatrix=self.imageToImageTransformMatrix,
                userModelSpecifications=self.userModelSpecifications,
                userOptimizationOptions=self.userOptimizationOptions,
                visualizer=self.visualizer,
                saveHistory=True,
                targetIntensity=self.targetIntensity,
                targetSearchStrings=self.targetSearchStrings,
                modeNames=self.modeNames,
                pallidumAsWM=self.pallidumAsWM,
                savePosteriors=self.savePosteriors
            ))
            self.timepointModels[timepointNumber].mask = self.sstModelObject.sstModel.mask
            self.timepointModels[timepointNumber].imageBuffers = self.imageBuffersList[timepointNumber]
            self.timepointModels[timepointNumber].voxelSpacing = self.sstModelObject.sstModel.voxelSpacing
            self.timepointModels[timepointNumber].transform = self.sstModelObject.sstModel.transform
            self.timepointModels[timepointNumber].cropping = self.sstModelObject.sstModel.cropping
