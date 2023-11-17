package org.deeplearning4j.examples.advanced.modelling.sequencepredictionMT;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

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
        
        System.out.println(csvFile);
        
        
        
        RecordReader rr = new CSVRecordReader(0);
        rr.initialize(new FileSplit(new File(csvFile)));
        
        DataSetIterator iter = new RecordReaderDataSetIterator.Builder(rr, batchSize)
        //Specify the columns that the regression labels/targets appear in. Note that all other columns will be
        // treated as features. Columns indexes start at 0
        .regression(9, 13)
        .build();
        
		List<String> labels = new ArrayList<String>();
		labels = iter.getLabels();
		
		
		if (!labels.isEmpty()) {
			for (String label : labels) {
				System.out.println(label);
			}
		} else {
			System.out.println("Labels null");
		}
      

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

}
