package Backpropagation.Algorithm;

import java.util.*;

public class NeuralNetwork {

    private final Random rand = new Random();
    private final ArrayList<Neuron> inputLayer = new ArrayList<>();
    private final ArrayList<Neuron[]> hiddenLayers = new ArrayList<>();
    private final ArrayList<Neuron> outputLayer = new ArrayList<>();

    private double momentum;
    private double learningRate;

    private ArrayList<Double[]> inputs = new ArrayList<>();
    private ArrayList<Double> outputKinds = new ArrayList<>();

    public NeuralNetwork(ArrayList<Double[]> inputs, ArrayList<Double> outputKinds, String hidden, double momentum,
                         double learningRate, double threshold, double minRange, double maxRange) {
        this.inputs = inputs;
        this.outputKinds = outputKinds;
        this.momentum = momentum;
        this.learningRate = learningRate;
        int inputNeuron = inputs.get(0).length - 1;
        int[] hiddenNeurons = Arrays.stream(hidden.split(",")).mapToInt(Integer::parseInt).toArray();
        int outputNeuron = 1;
        // input layer
        for (int i = 0; i < inputNeuron; i++) {
            Neuron neuron = new Neuron();
            inputLayer.add(neuron);
        }
        // hidden layers
        for (int i = 0; i < hiddenNeurons.length; i++) {
            Neuron neurons[] = new Neuron[hiddenNeurons[i]];
            if (i == 0) {
                for (int j = 0; j < neurons.length; j++) {
                    neurons[j] = new Neuron();
                    neurons[j].addConnections(inputLayer);
                }
            } else {
                for (int j = 0; j < neurons.length; j++) {
                    neurons[j] = new Neuron();
                    neurons[j].addConnections(hiddenLayers.get(i - 1));
                }
            }
            hiddenLayers.add(neurons);
        }
        // output layer
        for (int i = 0; i < outputNeuron; i++) {
            Neuron neuron = new Neuron();
            neuron.addConnections(hiddenLayers.get(hiddenLayers.size() - 1));
            outputLayer.add(neuron);
        }

        // Initialize random weights
        for (Neuron[] neurons : hiddenLayers) {
            for (Neuron neuron : neurons) {
                ArrayList<Connection> connections = neuron.getAllConnections();
                for (Connection conn : connections) {
                    double newWeight = getRandomNumber(minRange, maxRange);
                    conn.setWeight(newWeight);
                }
                connections.get(0).setWeight(threshold);
            }
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
        for (Neuron[] neurons : hiddenLayers) {
            for (Neuron neuron : neurons) {
                neuron.calculateOutput();
            }
        }
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
        double[] pervPartialDerivatives = new double[0];
        for (int j = hiddenLayers.size() - 1; j >= 0; j--) {
            Neuron[] neurons = hiddenLayers.get(j);
            double[] nowPartialDerivatives = new double[neurons.length];
            int n = 0;
            for (Neuron neuron : neurons) {
                ArrayList<Connection> connections = neuron.getAllConnections();
                for (Connection connection : connections) {
                    double pervY = connection.leftNeuron.getOutput();
                    double y = neuron.getOutput();
                    double sumOutputs = 0;
                    if (j == hiddenLayers.size() - 1) {
                        int k = 0;
                        for (Neuron outputN : outputLayer) {
                            double wjk = outputN.getConnection(neuron.id).getWeight();
                            double dy = expectedOutput[k];
                            double yk = outputN.getOutput();
                            sumOutputs += (dy - yk) * yk * (1 - yk) * wjk;
                            k++;
                        }
                    } else {
                        int k = 0;
                        for (Neuron hiddenN : hiddenLayers.get(j + 1)) {
                            double wjk = hiddenN.getConnection(neuron.id).getWeight();
                            sumOutputs += pervPartialDerivatives[k] * wjk;
                            k++;
                        }
                    }
                    // TODO: Move pD calc to outside
                    double partialDerivative = y * (1 - y) * sumOutputs;
                    nowPartialDerivatives[n] = partialDerivative;
                    double deltaWeight = learningRate * partialDerivative * pervY;
                    double newWeight = connection.getWeight() + deltaWeight;
                    connection.setDeltaWeight(deltaWeight);
                    connection.setWeight(newWeight + momentum * connection.getPrevDeltaWeight());
                }
                n++;
            }
            pervPartialDerivatives = nowPartialDerivatives;
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
        double error = 1;
        int correct = 0;
        for (int i = 0; i < maxSteps && error > minError; i++) {
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
        return (double) correct / inputs.size() * 100 + "%";
    }

    public int[] getOutputKind(ArrayList<Double[]> inputs, int maxSteps, double minError) {
        int i, y[] = new int[inputs.size()];
        double error = 1;
        for (i = 0; i < maxSteps && error > minError; i++) {
            error = 0;
            int yi = 0;
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
                y[yi] = idx;
                yi++;
            }
        }
        return y;
    }

    private void printAllWeights() {
        //hiddenLayers.forEach(this::printWeights);
        outputLayer.forEach(this::printWeights);
        System.out.println();
    }

    private void printWeights(Neuron n) {
        ArrayList<Connection> connections = n.getAllConnections();
        for (Connection con : connections) {
            double w = con.getWeight();
            System.out.println("NeuronID = " + n.id + ", ConnectionID = " + con.id + ",Weight = " + w);
        }
    }
}
