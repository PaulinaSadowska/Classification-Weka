package put.cs.idss.ml.weka.lab08;

import weka.classifiers.Classifier;
import weka.core.Instances;

import java.io.IOException;

/**
 * Created by palka on 16.12.2016.
 */
public class ClassifierTrainer {

    public Instances trainSet;

    public ClassifierTrainer(DataProvider dataProvider, double partOfDataSet) throws IOException {
        trainSet = dataProvider.getTrainSet(partOfDataSet);
    }

    public Long buildClassifier(Classifier classifier) throws Exception {
        String classifierName = classifier.getClass().getSimpleName();
        //System.out.println("Training " + classifierName + "...");
        long trainingTimeStart = System.currentTimeMillis();
        classifier.buildClassifier(trainSet);
        long trainingTime = System.currentTimeMillis() - trainingTimeStart;
        //System.out.println(" - training time: " + trainingTime);
        return trainingTime;
    }
}
