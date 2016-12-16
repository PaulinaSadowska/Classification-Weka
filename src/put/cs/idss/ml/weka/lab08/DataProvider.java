package put.cs.idss.ml.weka.lab08;

import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

/**
 * Created by palka on 16.12.2016.
 */
public class DataProvider {

    private static final String DATA_FOLDER_PATH = "data/";
    private String dataSetName;
    private long seed;

    public DataProvider(String dataSetName){
        this(dataSetName, 1);
    }

    public DataProvider(String dataSetName, long seed){
        this.dataSetName = dataSetName;
        this.seed = seed;
    }

    public Instances getTrainSet(double partOfDataSet) throws IOException {
        BufferedReader readerTrain = new BufferedReader(new FileReader(DATA_FOLDER_PATH+dataSetName+"-train.arff"));
        Instances trainSetTmp = new Instances(readerTrain);
        int newTrainSetSize = (int)((double)trainSetTmp.numInstances() * partOfDataSet);
        trainSetTmp.randomize(new Random(seed));
        Instances trainSet = new Instances(trainSetTmp, 0, newTrainSetSize);
        readerTrain.close();
        return trainSet;
    }

    public Instances getTestSet() throws IOException {
        BufferedReader readerTest = new BufferedReader(new FileReader(DATA_FOLDER_PATH+dataSetName+"-test.arff"));
        Instances testSet = new Instances(readerTest);
        testSet.randomize(new Random(seed));
        readerTest.close();
        return testSet;
    }

}
