package org.deeplearning4j.examples.advanced.modelling.sequencepredictionMT;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ClinicalTimeSeriesLSTM {

	private static String DATA_PATH = FilenameUtils.concat("c:\\Users\\Dell\\Downloads\\physionet2012\\", "");
	private static int NB_TRAIN_EXAMPLES = 3200; // number of training examples
	private static int NB_TEST_EXAMPLES = 800; // number of testing examples

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
		String featureBaseDir = FilenameUtils.concat(DATA_PATH, "sequence"); // set feature directory
		String mortalityBaseDir = FilenameUtils.concat(DATA_PATH, "mortality");// set label directory

		// Load training data
		CSVSequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1, ",");
		trainFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

		CSVSequenceRecordReader trainLabels = new CSVSequenceRecordReader();
		trainLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

		SequenceRecordReaderDataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures,
				trainLabels, 32, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

		// Load testing data
		CSVSequenceRecordReader testFeatures = new CSVSequenceRecordReader(1, ",");
		testFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", NB_TRAIN_EXAMPLES,
				NB_TRAIN_EXAMPLES + NB_TEST_EXAMPLES - 1));

		CSVSequenceRecordReader testLabels = new CSVSequenceRecordReader();
		testLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", NB_TRAIN_EXAMPLES,
				NB_TRAIN_EXAMPLES + NB_TEST_EXAMPLES - 1));

		SequenceRecordReaderDataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels,
				32, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

		// Neural Network Configuration
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(RANDOM_SEED)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam(LEARNING_RATE))
				.weightInit(WeightInit.XAVIER).dropOut(0.25).graphBuilder().addInputs("trainFeatures")
				.setOutputs("predictMortality")
				.validateOutputLayerConfig(false)
				.inputPreProcessor(0, new CnnToRnnPreProcessor(6, 2, 7 ))
				.addLayer("L1",
						new GravesLSTM.Builder().nIn(NB_INPUTS).nOut(LSTM_LAYER_SIZE).forgetGateBiasInit(1)
								.activation(Activation.TANH).build(),
						"trainFeatures")
				.addLayer(
						"predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.XENT)
								.activation(Activation.SOFTMAX).nIn(LSTM_LAYER_SIZE).nOut(NUM_LABEL_CLASSES).build(),
						"L1")
				.build();

		ComputationGraph model = new ComputationGraph(conf);

		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new FileStatsStorage(
				new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
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

		System.out.println("FINAL TEST AUC: " + roc.calculateAUC());
	}

}
