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
    private Instances trainSet;
    private Instances testSet;

    private int trainSetSize;

    public DataProvider(String dataSetName) {
        this(dataSetName, 1);
    }

    public DataProvider(String dataSetName, long seed) {
        this.dataSetName = dataSetName;
        this.seed = seed;
    }

    private void setClassIndex(Instances set) {
        if (set.classIndex() == -1) set.setClassIndex(set.numAttributes() - 1);
    }

    public Instances getTrainSet(double partOfDataSet) throws IOException {
        if(trainSet==null) {
            BufferedReader readerTrain = new BufferedReader(new FileReader(DATA_FOLDER_PATH + dataSetName + "-train.arff"));
            trainSet = new Instances(readerTrain);
            readerTrain.close();
            trainSet.randomize(new Random(seed));
        }
        trainSetSize = (int) ((double) trainSet.numInstances() * partOfDataSet);
        Instances smallerTrainSet = new Instances(trainSet, 0, trainSetSize);
        setClassIndex(smallerTrainSet);
        return smallerTrainSet;
    }

    public int getTrainSetSize(){
        return trainSetSize;
    }

    public Instances getTestSet() throws IOException {
        if (testSet == null) {
            BufferedReader readerTest = new BufferedReader(new FileReader(DATA_FOLDER_PATH + dataSetName + "-test.arff"));
            testSet = new Instances(readerTest);
            testSet.randomize(new Random(seed));
            readerTest.close();
            setClassIndex(testSet);
        }
        return testSet;
    }

}
