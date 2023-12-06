package org.deeplearning4j.examples.advanced.modelling.sequencepredictionMT;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.arbiter.ComputationGraphSpace;
import org.deeplearning4j.arbiter.conf.updater.AdamSpace;
import org.deeplearning4j.arbiter.layers.GravesLSTMLayerSpace;
import org.deeplearning4j.arbiter.layers.RnnOutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.ROCScoreFunction;
import org.deeplearning4j.arbiter.scoring.impl.ROCScoreFunction.ROCType;
import org.deeplearning4j.arbiter.task.ComputationGraphTaskCreator;
import org.deeplearning4j.arbiter.ui.listener.ArbiterStatusListener;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ClinicalArbiterTimeSeriesLSTM {

	private static final long MAX_OPTIMIZATION_TIME = 30; // Max optimization time in minutes
	private static final int MAX_OPTIMIZATION_COUNT = 120; // Max optimization count
	
	private static String DATA_PATH = "c:\\Users\\Dell\\Downloads\\physionet2012\\";
	private static int NB_TRAIN_EXAMPLES = 800; // number of training examples
	private static int NB_TEST_EXAMPLES = 160; // number of testing examples

	// Set neural network parameters
	private static int NB_INPUTS = 86;
	private static int NB_EPOCHS = 10;
	private static int RANDOM_SEED = 1234;
	private static double LEARNING_RATE = 0.005;
	private static int BATCH_SIZE = 32;
	private static int LSTM_LAYER_SIZE = 200;
	private static int NUM_LABEL_CLASSES = 2;

	public ClinicalArbiterTimeSeriesLSTM() {
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		String featureBaseDir = DATA_PATH + "sequence"; // set feature directory
		String mortalityBaseDir = DATA_PATH + "mortality";// set label directory
		
		Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
		
		
		// Load training data
		CSVSequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1, ",");
		trainFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "\\%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

		CSVSequenceRecordReader trainLabels = new CSVSequenceRecordReader();
		trainLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "\\%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

		SequenceRecordReaderDataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures,
				trainLabels, BATCH_SIZE, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

		// Load testing data
		CSVSequenceRecordReader testFeatures = new CSVSequenceRecordReader(1, ",");
		testFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "\\%d.csv", NB_TRAIN_EXAMPLES,
				NB_TRAIN_EXAMPLES + NB_TEST_EXAMPLES - 1));

		CSVSequenceRecordReader testLabels = new CSVSequenceRecordReader();
		testLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "\\%d.csv", NB_TRAIN_EXAMPLES,
				NB_TRAIN_EXAMPLES + NB_TEST_EXAMPLES - 1));

		SequenceRecordReaderDataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels,
				BATCH_SIZE, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
		
		// Hyperparameters variables
//		ContinuousParameterSpace learningRateHyperparam = new ContinuousParameterSpace(0.0001, 0.1);
//		IntegerParameterSpace layerSizeHyperparam = new IntegerParameterSpace(16, 256);
		
		ParameterSpace<Double> learningRateParam = new ContinuousParameterSpace(0.0001, 0.01);
		ParameterSpace<Integer> layerSizeParam = new IntegerParameterSpace(10, 250);
		
		// Neural Network Configuration
		ComputationGraphSpace conf = new ComputationGraphSpace.Builder()
				.seed(RANDOM_SEED)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new AdamSpace(learningRateParam))
				.weightInit(WeightInit.XAVIER)
				.dropOut(0.25)
				.addInputs("trainFeatures")
				.setOutputs("predictMortality")
				.validateOutputLayerConfig(false)
				.layer("L0", new GravesLSTMLayerSpace.Builder()
						.nIn(NB_INPUTS)
						.nOut(layerSizeParam)
						.forgetGateBiasInit(1)
						.activation(Activation.TANH)
						.build(), "trainFeatures")
//				.addLayer("L1", new LSTM.Builder()
//						.nIn(LSTM_LAYER_SIZE)
//						.nOut(100)
//						.activation(Activation.TANH)
//						.build(), "L0")
				.layer("predictMortality", new RnnOutputLayerSpace.Builder()
						.activation(Activation.SOFTMAX)
						.nIn(layerSizeParam)
						.nOut(NUM_LABEL_CLASSES)
						.lossFunction(LossFunctions.LossFunction.XENT)
						.build(), "L0")
				.setInputTypes(InputType.recurrent(NB_INPUTS))
				.build();
		
		CandidateGenerator candidateGenerator = new RandomSearchGenerator(conf);
//		CandidateGenerator candidateGenerator = new GridSearchCandidateGenerator(conf, 4,
//				 GridSearchCandidateGenerator.Mode.Sequential);
		
		
		@SuppressWarnings("deprecation")
		ScoreFunction scoreFunction = new ROCScoreFunction(ROCType.ROC,ROCScoreFunction.Metric.AUC);
//		ScoreFunction scoreFunction = new EvaluationScoreFunction(Metric.ACCURACY);

		String baseSaveDirectory = System.getProperty("user.dir") + "\\res\\save";
		File f = new File(baseSaveDirectory);

		if (f.exists())
			f.delete();
		f.mkdir();

		FileModelSaver modelSaver = new FileModelSaver(baseSaveDirectory);

		TerminationCondition[] terminationConditions = {
				new MaxTimeCondition(MAX_OPTIMIZATION_TIME, TimeUnit.MINUTES),
				new MaxCandidatesCondition(MAX_OPTIMIZATION_COUNT)
		};
		
		Properties dataSourceProperties = new Properties();
		dataSourceProperties.put("minibatchSize", BATCH_SIZE);
		dataSourceProperties.put("trainData", trainData);
		dataSourceProperties.put("testData", testData);

		OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
				 	.candidateGenerator(candidateGenerator)
				 	.dataSource(ExampleDataSource.class, dataSourceProperties)
	                .modelSaver(modelSaver)
	                .scoreFunction(scoreFunction)
	                .terminationConditions(terminationConditions)
	                .build();
		 
		IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new ComputationGraphTaskCreator());
		
		
		// UI server
		StatsStorage storage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"),"HyperParamOptimizationStatsModel.dl4j"));
		runner.addListeners(new ArbiterStatusListener(storage));
		UIServer.getInstance().attach(storage);
		
		// Start the hyperparameter optimization
//		runner.addListeners(new LoggingStatusListener());
		runner.execute();

		String s = "Best score: " + runner.bestScore() + "\n" + "Index of model with best score: "
				+ runner.bestScoreCandidateIndex() + "\n" + "Number of configurations evaluated: "
				+ runner.numCandidatesCompleted() + "\n";
		System.out.println(s);

		// Get all results, and print out details of the best result:
		int indexOfBestResult = runner.bestScoreCandidateIndex();
		List<ResultReference> allResults = runner.getResults();

		OptimizationResult bestResult = allResults.get(indexOfBestResult).getResult();
		Model bestModel = (Model) bestResult.getResultReference().getResultModel();
		
		System.out.println("\n\nConfiguration of best model:\n");
		 String  json = null;
		    if (bestModel instanceof MultiLayerNetwork) {
		      json = ((MultiLayerNetwork)bestModel).getLayerWiseConfigurations().toJson();
		    } else if (bestModel instanceof ComputationGraph) {
		      json = ((ComputationGraph)bestModel).getConfiguration().toJson();
		    }
		    
		    System.out.println(json);
//		System.out.println(bestModel.getLayerWiseConfigurations().toJson());

//		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .seed(12345)
//                .weightInit(WeightInit.XAVIER)
//                .updater(new AdaGrad(0.005))
//                .list()
//                .layer(0, new ConvolutionLayer.Builder(kernelSize, kernelSize)
//                        .nIn(1) //1 channel
//                        .nOut(7)
//                        .stride(2, 2)
//                        .activation(Activation.RELU)
//                        .build())
//                .layer(1, new LSTM.Builder()
//                        .activation(Activation.SOFTSIGN)
//                        .nIn(84)
//                        .nOut(200)
//                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
//                        .gradientNormalizationThreshold(10)
//                        .build())
//                .layer(2, new RnnOutputLayer.Builder(LossFunction.MSE)
//                        .activation(Activation.IDENTITY)
//                        .nIn(200)
//                        .nOut(52)
//                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
//                        .gradientNormalizationThreshold(10)
//                        .build())
//                .inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, numChannels))
//                .inputPreProcessor(1, new CnnToRnnPreProcessor(6, 2, 7 ))
//                .build();
//		
//		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		

//		UIServer uiServer = UIServer.getInstance();
//		StatsStorage statsStorage = new FileStatsStorage(
//				new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
//		TimeIterationListener timeListener = new TimeIterationListener(10);
//		uiServer.attach(statsStorage);
		
		// training
//		model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));

		// Model training
//		model.fit(trainData, NB_EPOCHS);

		// Model evaluation
//		org.nd4j.evaluation.classification.ROC roc = new org.nd4j.evaluation.classification.ROC(100);

//		while (testData.hasNext()) {
//			DataSet batch = testData.next();
//			INDArray[] output = model.output(batch.getFeatures());
//			roc.evalTimeSeries(batch.getLabels(), output[0]);
//		}
//
//		System.out.println("FINAL TEST AUC: " + roc.calculateAUC());
	}
	public static class ExampleDataSource implements DataSource {
		
		private int minibatchSize;
		private SequenceRecordReaderDataSetIterator trainData;
		private SequenceRecordReaderDataSetIterator testData;

		public ExampleDataSource() {
			super();
		}

//		public ExampleDataSource(int minibatchSize) {
//			super();
//			this.minibatchSize = minibatchSize;
//		}
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;

		@Override
		public void configure(Properties properties) {
			this.minibatchSize = Integer.parseInt(properties.getProperty("minibatchSize","32"));
			this.trainData = (SequenceRecordReaderDataSetIterator) properties.get("trainData");
			this.testData = (SequenceRecordReaderDataSetIterator) properties.get("testData");
			
		}

		@Override
		public Object trainData() {
			return trainData;
		}

		@Override
		public Object testData() {
			return testData;
		}

		@Override
		public Class<?> getDataType() {
			return SequenceRecordReaderDataSetIterator.class;
		}
		
	}
	
}

