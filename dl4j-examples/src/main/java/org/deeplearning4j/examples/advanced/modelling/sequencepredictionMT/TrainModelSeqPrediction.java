package org.deeplearning4j.examples.advanced.modelling.sequencepredictionMT;

import java.io.File;
import java.util.Scanner;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class TrainModelSeqPrediction {
    private static int batchSize = 64;
    private static long seed = 123;
    private static int numEpochs = 3;
    private static boolean modelType = true;

    public static String dataLocalPath;

	public TrainModelSeqPrediction() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) throws Exception {
		 // Define path to your CSV file
        String csvFile = System.getProperty("user.dir") + "\\res\\generated_data.csv";
        
        // Save model file
        File modelFile = new File(System.getProperty("user.dir"), "seqPredictModel.json");
        
        System.out.println(csvFile);
        
        Scanner scanner = new Scanner(new File(csvFile));
        
        System.out.println(scanner.nextLine());

        
        
        
        RecordReader rr = new CSVSequenceRecordReader(0,",");
        rr.initialize(new FileSplit(new File(csvFile)));
        
     // Define configuration for DataSetIterator
        int miniBatchSize = 1;
        int numFeatures = 10; // Number of features in each time step
        int numLabels = 4; // Number of labels in each time step
        int totalColumns = numFeatures + numLabels; // Total number of columns in the CSV

//        // Create empty lists to store features and labels
//        List<INDArray> features = new ArrayList<>();
//        List<INDArray> labels = new ArrayList<>();
//
//        // Read and separate features and labels manually
//        while (rr.hasNext()) {
//            List<Writable> line = rr.next();
//
//            // Extract features and labels
//            INDArray featureArray = Nd4j.create(line.subList(0, numFeatures).stream()
//                    .map(Writable::toDouble)
//                    .mapToDouble(Double::doubleValue)
//                    .toArray(), new int[]{1, numFeatures});
//
//            INDArray labelArray = Nd4j.create(line.subList(numFeatures, totalColumns).stream()
//                    .map(Writable::toDouble)
//                    .mapToDouble(Double::doubleValue)
//                    .toArray(), new int[]{1, numLabels});
//
//            features.add(featureArray);
//            labels.add(labelArray);
//        }
//
//        // Stack the features and labels to create 3D arrays
//        INDArray features3D = Nd4j.vstack(features.toArray(new INDArray[0]));
//        INDArray labels3D = Nd4j.vstack(labels.toArray(new INDArray[0]));
//        
//        DataSet dataSet = new DataSet(features3D,labels3D);
//
//        // Print the 3D arrays
//        System.out.println("Original 3D Features:\n" + features3D);
//        System.out.println("Original 3D Labels:\n" + labels3D);

        // Optionally, preprocess your data if needed
        // Normalization or other transformations can be applied here

        // Your RNN network configuration goes here

        // Train your model using the DataSetIterator
        DataSetIterator iter = new SeqMTDataSetIterator(csvFile, batchSize, modelType);
//        DataSetIterator iter = new RecordReaderDataSetIterator.Builder(rr, batchSize)
//        //Specify the columns that the regression labels/targets appear in. Note that all other columns will be
//        // treated as features. Columns indexes start at 0
//        .regression(9, 13)
//        .build();
        
        
        MultiLayerNetwork model = getNetModel(iter.inputColumns(), iter.totalOutcomes());        
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.attach(statsStorage);
        
        long startTime = System.currentTimeMillis();
        model.fit(iter, numEpochs);
        long endTime = System.currentTimeMillis();
        System.out.println("=============run time=====================" + (endTime - startTime));

        // save model to disk
        model.save(modelFile, true);
	
      

//        // Set up record reader
//        RecordReader recordReader = new CSVRecordReader(0, ",");
//        recordReader.initialize(new FileSplit(new File(csvFile)));
//
//        // Define configuration for DataSetIterator
//        int miniBatchSize = batchSize;
//        int numFeatures = 10; // Number of features in each line
//        int numLabels = 4; // Number of labels in each line
//
//        // Create empty lists to store features and labels
//        List<INDArray> features = new ArrayList<>();
//        List<INDArray> labels = new ArrayList<>();
//
//        // Read and separate features and labels manually
//        while (recordReader.hasNext()) {
//            List<Writable> line = recordReader.next();
//
//            // Extract features and labels
//            INDArray featureArray = RecordConverter.toArray(line.subList(0, numFeatures));
//            INDArray labelArray = RecordConverter.toArray(line.subList(numFeatures, numFeatures + numLabels));
//
//            features.add(featureArray);
//            labels.add(labelArray);
//        }
//
//        // Create ND4J DataSet
//        INDArray featuresArray = Nd4j.vstack(features.toArray(new INDArray[0]));
//        INDArray labelsArray = Nd4j.vstack(labels.toArray(new INDArray[0]));
//
////        // Normalize features
////        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
////        normalizer.fit(featuresArray);
////        normalizer.transform(featuresArray);
//
//        DataSetIterator dataSetIterator = new org.nd4j.linalg.dataset.api.iterator.IteratorDataSetIterator(
//                new org.nd4j.linalg.dataset.DataSet(featuresArray, labelsArray), miniBatchSize);

        // Your LSTM network configuration goes here

        // Train your model using the DataSetIterator

	}
	
	 //create the neural network
    private static MultiLayerNetwork getNetModel(int inputNum, int outputNum) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
            .seed(seed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .weightInit(WeightInit.XAVIER)
            .updater(new RmsProp.Builder().rmsDecay(0.95).learningRate(1e-2).build())
            .list()
            .layer(new LSTM.Builder().name("lstm1")
                .activation(Activation.TANH).nIn(inputNum).nOut(100).build())
            .layer(new LSTM.Builder().name("lstm2")
                .activation(Activation.TANH).nOut(80).build())
            .layer(new RnnOutputLayer.Builder().name("output")
                .activation(Activation.SOFTMAX).nOut(outputNum).lossFunction(LossFunctions.LossFunction.MSE)
                .build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }

}
