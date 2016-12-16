package put.cs.idss.ml.weka.lab07;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class NaiveBayesClassifier extends Classifier {

    /**
     *
     */
    private static final long serialVersionUID = 7550409893545527343L;

    /**
     * number of classes
     */
    protected int numClasses;

    /**
     * discCounts, means, standard deviations, priors.....
     */
    protected double prioriProbabilities[][][];
    protected double classProbabilities[];

    protected double means[][];
    protected double stdDevs[][];

    public NaiveBayesClassifier() {
        // TODO Auto-generated constructor stub
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {

        numClasses = data.numClasses();
        int numInstances = data.numInstances();

        double[][][] discCounts = new double[numClasses][data.numAttributes() - 1][];
        prioriProbabilities = new double[numClasses][data.numAttributes() - 1][];
        classProbabilities = new double[numClasses];

        double[][] attrValueSums = new double[numClasses][data.numAttributes()];
        means = new double[numClasses][data.numAttributes()];
        stdDevs = new double[numClasses][data.numAttributes()];

        // remove instances with missing class
        data.deleteWithMissingClass();

		/* 1. Initialize arrays of discCounts for nominal attributes,
         * means and std.devs. for numeric attributes,
		 * and a priori probabilities of the classes. */
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            Attribute attribute = data.attribute(i);
            for (int j = 0; j < numClasses; j++) {
                if (attribute.isNominal()) {
                    discCounts[j][i] = new double[attribute.numValues()];
                    prioriProbabilities[j][i] = new double[attribute.numValues()];
                }
            }
        }

        double classOccurrences[] = new double[numClasses];

        // 2. compute discCounts and sums.
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            int classValue = (int) instance.classValue();
            classOccurrences[classValue] += 1;
            for (int j = 0; j < data.numAttributes() - 1; j++) {
                Attribute attribute = data.attribute(j);
                double value = instance.value(j);
                if (attribute.isNominal()) {
                    discCounts[classValue][j][(int) value]++;
                } else {
                    attrValueSums[classValue][j] += value;
                }
            }
        }

        for (int i = 0; i < numClasses; i++) {
            classProbabilities[i] = classOccurrences[i] / numInstances;
        }

        // 3. Compute means.

        // 5. normalize discCounts and a priori probabilities

        for (int i = 0; i < data.numAttributes() - 1; i++) {
            Attribute attribute = data.attribute(i);
            for (int j = 0; j < numClasses; j++) {
                if (attribute.isNominal()) {
                    for (int k = 0; k < attribute.numValues(); k++) {
                        prioriProbabilities[j][i][k] = discCounts[j][i][k] / numInstances;
                    }
                } else {
                    means[j][i] = attrValueSums[j][i] / classOccurrences[j];
                }
            }
        }

        // 4. Compute standard deviations.

        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            int classValue = (int) instance.classValue();
            for (int j = 0; j < data.numAttributes() - 1; j++) {
                if (!data.attribute(j).isNominal()) {
                    stdDevs[classValue][j] += Math.pow(instance.value(j) - means[classValue][j], 2);
                }
            }
        }


        for (int i = 0; i < data.numAttributes() - 1; i++) {
            Attribute attribute = data.attribute(i);
            for (int j = 0; j < data.numClasses(); j++) {
                if (!attribute.isNominal()) {
                    stdDevs[j][i] = Math.sqrt(stdDevs[j][i] / classOccurrences[j]);
                }
            }
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] distribution = new double[numClasses];
        double[] P_X = new double[numClasses];
        double denominator = 0.0;

        for (int i = 0; i < numClasses; i++) {
            P_X[i] = 1.0;
            for (int j = 0; j < instance.numAttributes() - 1; j++) {
                Attribute attribute = instance.attribute(j);
                if (attribute.isNominal()) {
                    double value = instance.value(j);
                    P_X[i] *= prioriProbabilities[i][j][(int) value] / classProbabilities[i];
                } else {
                    double mean = means[i][j];
                    double stdDev = stdDevs[i][j];
                    double x = instance.value(j);

                    double power = (-(Math.pow((x - mean), 2))) / (2 * Math.pow(stdDev, 2));
                    P_X[i] *= Math.pow(Math.E, power) / (stdDev * Math.sqrt(2 * Math.PI));
                }
            }
            distribution[i] = P_X[i] * classProbabilities[i];
            denominator += distribution[i];
        }
        for (int i = 0; i < numClasses; i++) {
            distribution[i] = distribution[i] / denominator;
        }


        return distribution;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double classValue = 0.0;
        double max = Double.MIN_VALUE;
        double[] dist = distributionForInstance(instance);

        for (int i = 0; i < dist.length; i++) {
            if (dist[i] > max) {
                classValue = i;
                max = dist[i];
            }
        }

        return classValue;
    }

}
