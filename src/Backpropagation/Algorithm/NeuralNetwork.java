package Backpropagation.Algorithm;

import java.util.*;

public class NeuralNetwork {

    private final Random rand = new Random();
    private final ArrayList<Neuron> inputLayer = new ArrayList<>();
    private final ArrayList<Neuron> hiddenLayer = new ArrayList<>();
    private final ArrayList<Neuron> outputLayer = new ArrayList<>();

    private double momentum;
    private double learningRate;

    private ArrayList<Double[]> inputs = new ArrayList<>();
    private ArrayList<Double> outputKinds = new ArrayList<>();

    public NeuralNetwork(ArrayList<Double[]> inputs, ArrayList<Double> outputKinds, int hidden, double momentum,
                         double learningRate, double threshold, double minRange, double maxRange) {
        this.inputs = inputs;
        this.outputKinds = outputKinds;
        int[] layers = new int[]{inputs.get(0).length - 1, hidden, 1};
        this.momentum = momentum;
        this.learningRate = learningRate;
        for (int i = 0; i < layers.length; i++) {
            if (i == 0) { // input layer
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    inputLayer.add(neuron);
                }
            } else if (i == 1) { // hidden layer
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    neuron.addConnections(inputLayer);
                    hiddenLayer.add(neuron);
                }
            } else if (i == 2) { // output layer
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    neuron.addConnections(hiddenLayer);
                    outputLayer.add(neuron);
                }
            } else {
                System.out.println("!Error NeuralNetwork init");
            }
        }

        // Initialize random weights
        for (Neuron neuron : hiddenLayer) {
            ArrayList<Connection> connections = neuron.getAllConnections();
            for (Connection conn : connections) {
                double newWeight = getRandomNumber(minRange, maxRange);
                conn.setWeight(newWeight);
            }
            connections.get(0).setWeight(threshold);
        }
        for (Neuron neuron : outputLayer) {
            ArrayList<Connection> connections = neuron.getAllConnections();
            for (Connection conn : connections) {
                double newWeight = getRandomNumber(minRange, maxRange);
                conn.setWeight(newWeight);
            }
        }

        // Reset id counters
        Neuron.counter = 0;
        Connection.counter = 0;
    }

    private Double getRandomNumber(Double minRange, Double maxRange) {
        return minRange + (maxRange - minRange) * rand.nextDouble();
    }

    private void setInput(Double inputs[]) {
        for (int i = 0; i < inputLayer.size(); i++) {
            inputLayer.get(i).setOutput(inputs[i]);
        }
    }

    private Double[] getOutput() {
        Double[] outputs = new Double[outputLayer.size()];
        for (int i = 0; i < outputLayer.size(); i++)
            outputs[i] = outputLayer.get(i).getOutput();
        return outputs;
    }

    private void activate() {
        hiddenLayer.forEach(Neuron::calculateOutput);
        outputLayer.forEach(Neuron::calculateOutput);
    }

    private void applyBackpropagation(Double expectedOutput[]) {
        int i = 0;
        for (Neuron n : outputLayer) {
            ArrayList<Connection> connections = n.getAllConnections();
            for (Connection connection : connections) {
                double pervY = connection.leftNeuron.getOutput();
                double y = n.getOutput();
                double dy = expectedOutput[i];
                double partialDerivative = (dy - y) * y * (1 - y);
                double deltaWeight = learningRate * partialDerivative * pervY;
                double newWeight = connection.getWeight() + deltaWeight;
                connection.setDeltaWeight(deltaWeight);
                connection.setWeight(newWeight + momentum * connection.getPrevDeltaWeight());
            }
            i++;
        }

        for (Neuron n : hiddenLayer) {
            ArrayList<Connection> connections = n.getAllConnections();
            for (Connection con : connections) {
                double pervY = con.leftNeuron.getOutput();
                double y = n.getOutput();
                double sumOutputs = 0;
                int j = 0;
                for (Neuron outputN : outputLayer) {
                    double wjk = outputN.getConnection(n.id).getWeight();
                    double dy = expectedOutput[j];
                    double yk = outputN.getOutput();
                    sumOutputs += (dy - yk) * yk * (1 - yk) * wjk;
                    j++;
                }
                double partialDerivative = y * (1 - y) * sumOutputs;
                double deltaWeight = learningRate * partialDerivative * pervY;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
            }
        }
    }

    public String run(int maxSteps, double minError) {
        int i;
        // Train neural network until minError reached or maxSteps exceeded
        double error = 1;
        int correct = 0;
        for (i = 0; i < maxSteps && error > minError; i++) {
            error = 0;
            correct = 0;
            for (Double[] input : inputs) {
                setInput(input);
                activate();
                Double[] output = getOutput();
                Double[] expectedOutput = new Double[]{input[input.length - 1]};
                for (int j = 0; j < expectedOutput.length; j++) {
                    double err = Math.pow(expectedOutput[j] - output[j], 2) / 2;
                    error += err;
                }
                double distance = Math.abs(outputKinds.get(0) - output[0]);
                int idx = 0;
                for (int j = 1; j < outputKinds.size(); j++) {
                    double newDistance = Math.abs(outputKinds.get(j) - output[0]);
                    if (newDistance < distance) {
                        idx = j;
                        distance = newDistance;
                    }
                }
                double y = outputKinds.get(idx);
                if (y == expectedOutput[0]) ++correct;
                applyBackpropagation(expectedOutput);
            }
        }
        printAllWeights();
        // result = runTimes + MSE + trainRate
        return String.valueOf(i) + " " + error + " " + (double) correct / inputs.size() * 100 + "%";
    }

    public String test(ArrayList<Double[]> inputs, int maxSteps, double minError) {
        int i;
        double error = 1;
        int correct = 0;
        for (i = 0; i < maxSteps && error > minError; i++) {
            error = 0;
            correct = 0;
            for (Double[] input : inputs) {
                setInput(input);
                activate();
                Double[] output = getOutput();
                Double[] expectedOutput = new Double[]{input[input.length - 1]};
                for (int j = 0; j < expectedOutput.length; j++) {
                    double err = Math.pow(expectedOutput[j] - output[j], 2) / 2;
                    error += err;
                }
                double distance = Math.abs(outputKinds.get(0) - output[0]);
                int idx = 0;
                for (int j = 1; j < outputKinds.size(); j++) {
                    double newDistance = Math.abs(outputKinds.get(j) - output[0]);
                    if (newDistance < distance) {
                        idx = j;
                        distance = newDistance;
                    }
                }
                double y = outputKinds.get(idx);
                if (y == expectedOutput[0]) ++correct;
            }
        }
        // result = runTimes + MSE + trainRate
        return (double) correct / inputs.size() * 100 + "%";
    }

    private void printAllWeights() {
        System.out.println("printAllWeights");
        // weights for the hidden layer
        for (Neuron n : hiddenLayer) {
            ArrayList<Connection> connections = n.getAllConnections();
            for (Connection con : connections) {
                double w = con.getWeight();
                System.out.println("n=" + n.id + " c=" + con.id + " w=" + w);
            }
        }
        // weights for the output layer
        for (Neuron n : outputLayer) {
            ArrayList<Connection> connections = n.getAllConnections();
            for (Connection con : connections) {
                double w = con.getWeight();
                System.out.println("n=" + n.id + " c=" + con.id + " w=" + w);
            }
        }
        System.out.println();
    }
}
