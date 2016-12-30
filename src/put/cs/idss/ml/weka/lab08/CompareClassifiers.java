package put.cs.idss.ml.weka.lab08;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesianLogisticRegression;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;

public class CompareClassifiers {

    private static final String DATA_SET_NAME = "spambase"; // badges2 / credit-a-mod / credit-a

    static ArrayList<Classifier> classifiers;

    private static void initializeClassifiers() {
        //classifiers that we want to comparezzzzz
        classifiers = new ArrayList<>();
        classifiers.add(new NaiveBayes());
        classifiers.add(new Logistic());
    }

    public static void main(String[] args) throws Exception {

        initializeClassifiers();
        HashMap<String, Double> loss01 = new HashMap<>();
        HashMap<String, Double> squaredError = new HashMap<>();

        for (Classifier classifier : classifiers) {

            HashMap<Integer, Double> loss01ForParts = new HashMap<>();
            HashMap<Integer, Double> squaredErrorForParts = new HashMap<>();

            DataProvider dataProvider = new DataProvider(DATA_SET_NAME);
            Instances testSet = dataProvider.getTestSet();
            String classifierName = classifier.getClass().getSimpleName();


            for (int j = 1; j < 100; j+=1) {
                double partOfDataSet = 0.01*j; // part of randomized train set (0 .. 1)

                loss01.remove(classifierName);
                squaredError.remove(classifierName);

                ClassifierTrainer classifierTrainer = new ClassifierTrainer(dataProvider, partOfDataSet);
                classifierTrainer.buildClassifier(classifier);

                for (int i = 0; i < testSet.numInstances(); i++) {
                    Instance instance = testSet.instance(i);
                    int truth = (int) instance.classValue();

                    double[] distribution = classifier.distributionForInstance(instance);
                    int prediction = distribution[1] >= distribution[0] ? 1 : 0;

                    double _loss01 = truth == prediction ? 0 : 1;
                    double _squaredError = Math.pow(1.0 - distribution[truth], 2);

                    if (loss01.containsKey(classifierName)) {
                        _loss01 += loss01.get(classifierName);
                        _squaredError += squaredError.get(classifierName);
                    }
                    loss01.put(classifierName, _loss01);
                    squaredError.put(classifierName, _squaredError);
                }
                int testSetSize = testSet.numInstances();
                loss01ForParts.put(dataProvider.getTrainSetSize(), loss01.get(classifierName)/testSetSize);
                squaredErrorForParts.put(dataProvider.getTrainSetSize(), squaredError.get(classifierName)/testSetSize);
            }
            new DataExporter(classifierName, loss01ForParts, squaredErrorForParts).print(DATA_SET_NAME+"_" + classifierName).saveToFile(DATA_SET_NAME+"_" + classifierName);
        }
    }
}