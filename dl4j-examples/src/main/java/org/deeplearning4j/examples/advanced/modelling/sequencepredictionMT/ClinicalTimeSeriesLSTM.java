package org.deeplearning4j.examples.advanced.modelling.sequencepredictionMT;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation.Metric;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.listeners.TimeIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ClinicalTimeSeriesLSTM {

	private static String DATA_PATH = "c:\\Users\\Dell\\Downloads\\physionet2012\\";
	private static int NB_TRAIN_EXAMPLES = 10; // number of training examples
	private static int NB_TEST_EXAMPLES = 10; // number of testing examples

	// Set neural network parameters
	private static int NB_INPUTS = 86;
	private static int NB_EPOCHS = 10;
	private static int RANDOM_SEED = 1234;
	private static double LEARNING_RATE = 0.005;
	private static int BATCH_SIZE = 32;
	private static int LSTM_LAYER_SIZE = 200;
	private static int NUM_LABEL_CLASSES = 2;

	public ClinicalTimeSeriesLSTM() {
		// TODO Auto-generated constructor stub
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
		
		// Neural Network Configuration
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(RANDOM_SEED)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Adam(LEARNING_RATE))
				.weightInit(WeightInit.XAVIER)
				.dropOut(0.25)
				.graphBuilder()
				.addInputs("trainFeatures")
				.setOutputs("predictMortality")
				.validateOutputLayerConfig(false)
				.addLayer("L0", new GravesLSTM.Builder()
						.nIn(NB_INPUTS)
						.nOut(LSTM_LAYER_SIZE)
						.forgetGateBiasInit(1)
						.activation(Activation.TANH)
						.build(), "trainFeatures")
//				.addLayer("L1", new LSTM.Builder()
//						.nIn(LSTM_LAYER_SIZE)
//						.nOut(100)
//						.activation(Activation.TANH)
//						.build(), "L0")
				.addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.XENT)
						.activation(Activation.SOFTMAX)
						.nIn(LSTM_LAYER_SIZE)
						.nOut(NUM_LABEL_CLASSES)
						.build(), "L0")
				.setInputTypes(InputType.recurrent(NB_INPUTS))
				.build();
		ComputationGraph model = new ComputationGraph(conf);
		model.init();
		
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
		

		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new FileStatsStorage(
				new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
		TimeIterationListener timeListener = new TimeIterationListener(10);
		uiServer.attach(statsStorage);
		
		// training
		model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));

		// Model training
		model.fit(trainData, NB_EPOCHS);

		// Model evaluation
		org.nd4j.evaluation.classification.ROC roc = new org.nd4j.evaluation.classification.ROC(100);

		while (testData.hasNext()) {
			DataSet batch = testData.next();
			INDArray[] output = model.output(batch.getFeatures());
			roc.evalTimeSeries(batch.getLabels(), output[0]);
		}
		EvaluationScoreFunction scoreFunction = new EvaluationScoreFunction(Metric.ACCURACY);
		System.out.println("Score metric: " + scoreFunction.score(model, testData));
		
		System.out.println("FINAL TEST AUC: " + roc.calculateAUC());
	}

}
