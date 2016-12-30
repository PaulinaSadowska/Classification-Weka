package put.cs.idss.ml.weka.lab08;

import weka.classifiers.Classifier;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by palka on 16.12.2016.
 */
public class ResultPrinter {

    private ArrayList<Classifier> classifiers;

    public ResultPrinter(ArrayList<Classifier> classifiers) {
        this.classifiers = classifiers;
    }

    public String roundDouble(double x, int n) {
        String format = "%." + n + "f";
        return String.format(format, x);
    }

    public void print(HashMap<String, Long> trainingTimes, HashMap<String, Long> testingTimes, HashMap<String, Double> loss01, HashMap<String, Double> squaredError, double sum) {
        System.out.println("\nRESULTS:\n");
        for (Classifier classifier : classifiers) {
            String classifierName = classifier.getClass().getSimpleName();
            long trainingTime = trainingTimes.get(classifierName);
            long testingTime = testingTimes.get(classifierName);
            double _loss01 = loss01.get(classifierName) / sum;
            double _squaredError = squaredError.get(classifierName) / sum;
            System.out.println(classifierName + " :");
            System.out.println(" - training time:  " + trainingTime);
            System.out.println(" - testing time:   " + testingTime);
            System.out.println(" - 0/1 loss:       " + roundDouble(_loss01, 4));
            System.out.println(" - squared-error:  " + roundDouble(_squaredError, 4));
            System.out.println();
        }
    }
}
