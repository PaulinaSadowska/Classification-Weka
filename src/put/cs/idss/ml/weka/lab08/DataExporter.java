package put.cs.idss.ml.weka.lab08;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by Paulina Sadowska on 16.12.2016.
 */
class DataExporter {

    private String classifierName;
    private HashMap<Integer, Double> loss01;
    private HashMap<Integer, Double> squaredError;


    DataExporter(String classifierName, HashMap<Integer, Double> loss01, HashMap<Integer, Double> squaredError) {
        this.classifierName = classifierName;
        this.loss01 = loss01;
        this.squaredError = squaredError;
    }


    DataExporter saveToFile(String fileName) throws IOException {
        saveLoss(Paths.get("output/" + fileName + "_loss.txt"));
        saveSquaredError(Paths.get("output/" + fileName + "_square.txt"));
        return this;
    }

    private void saveSquaredError(Path path) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(path))
        {
            writer.write(classifierName+"\n");
            writer.write("LOSS 01\n");
            for (Integer trainSize : loss01.keySet().stream().sorted().collect(Collectors.toList())) {
                writer.write(trainSize + "\t" + printDouble(loss01.get(trainSize)) + "\n");
            }
        }
    }

    private void saveLoss(Path path) throws IOException{
        try (BufferedWriter writer = Files.newBufferedWriter(path))
        {
            writer.write(classifierName+"\n");
            writer.write("Squared Error\n");
            for (Integer trainSize : squaredError.keySet().stream().sorted().collect(Collectors.toList())) {
                writer.write(trainSize + "\t" + printDouble(squaredError.get(trainSize)) + "\n");
            }
        }
    }

    private String printDouble(double x) {
        String format = "%,10f";
        return String.format(format, x);
    }

    DataExporter print(String dataSetName) {
        System.out.println(dataSetName);
        System.out.println(classifierName);
        System.out.println("LOSS 01");
        for (Integer trainSize : loss01.keySet().stream().sorted().collect(Collectors.toList())) {
            System.out.println(trainSize + " " + loss01.get(trainSize));
        }
        System.out.println("Squared Error");
        for (Integer trainSize : squaredError.keySet().stream().sorted().collect(Collectors.toList())) {
            System.out.println(trainSize + " " + squaredError.get(trainSize));
        }
        return this;
    }
}
