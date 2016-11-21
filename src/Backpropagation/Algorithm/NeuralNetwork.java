package Backpropagation.Algorithm;

import java.text.*;
import java.util.*;

public class NeuralNetwork {

    final boolean isTrained = false;
    final DecimalFormat df;
    final Random rand = new Random();
    final ArrayList<Neuron> inputLayer = new ArrayList<>();
    final ArrayList<Neuron> hiddenLayer = new ArrayList<>();
    final ArrayList<Neuron> outputLayer = new ArrayList<>();
    final Neuron threshold = new Neuron();
    final int[] layers;
    final int randomWeightMultiplier = 1;

    final double epsilon = 0.00000000001;

    final double learningRate = 0.9f;
    final double momentum = 0.7f;

    // Inputs for xor problem
    final double inputs[][] = {{0, 0.4, 1}, {1, 0, 1}, {0, 0, 1}, {1, 1, 1}, {0, 1, 1}};

    // Corresponding outputs, xor training data
    final double expectedOutputs[][] = {{0.4}, {1}, {0}, {1}, {1}};
    double resultOutputs[][] = {{-1}, {-1}, {-1}, {-1}, {-1}}; // dummy init
    double output[];

    // for weight update all
    final HashMap<String, Double> weightUpdate = new HashMap<>();

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(3, 4, 1);
        int maxRuns = 50000;
        double minErrorCondition = 0.001;
        nn.run(maxRuns, minErrorCondition);
    }

    public NeuralNetwork(int input, int hidden, int output) {
        this.layers = new int[]{input, hidden, output};
        df = new DecimalFormat("#.0#");
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
                    neuron.addThresholdConnection(threshold);
                    hiddenLayer.add(neuron);
                }
            } else if (i == 2) { // output layer
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    neuron.addConnections(hiddenLayer);
                    neuron.addThresholdConnection(threshold);
                    outputLayer.add(neuron);
                }
            } else {
                System.out.println("!Error NeuralNetwork init");
            }
        }

        // initialize random weights
        for (Neuron neuron : hiddenLayer) {
            ArrayList<Connection> connections = neuron.getAllConnections();
            for (Connection conn : connections) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            }
        }
        for (Neuron neuron : outputLayer) {
            ArrayList<Connection> connections = neuron.getAllConnections();
            for (Connection conn : connections) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            }
        }

        // reset id counters
        Neuron.counter = 0;
        Connection.counter = 0;

        if (isTrained) {
            updateAllWeights();
        }
    }

    // random
    double getRandom() {
        return randomWeightMultiplier * (rand.nextDouble() * 2 - 1); // [-1;1[
    }

    /**
     * @param inputs There is equally many neurons in the input layer as there are
     *               in input variables
     */
    public void setInput(double inputs[]) {
        for (int i = 0; i < inputLayer.size(); i++) {
            inputLayer.get(i).setOutput(inputs[i]);
        }
    }

    public double[] getOutput() {
        double[] outputs = new double[outputLayer.size()];
        for (int i = 0; i < outputLayer.size(); i++)
            outputs[i] = outputLayer.get(i).getOutput();
        return outputs;
    }

    /**
     * Calculate the output of the neural network based on the input The forward
     * operation
     */
    public void activate() {
        for (Neuron n : hiddenLayer)
            n.calculateOutput();
        for (Neuron n : outputLayer)
            n.calculateOutput();
    }

    /**
     * all output propagate back
     *
     * @param expectedOutput first calculate the partial derivative of the error with
     *                       respect to each of the weight leading into the output neurons
     *                       threshold is also updated here
     */
    public void applyBackpropagation(double expectedOutput[]) {

        // error check, normalize value ]0;1[
        for (int i = 0; i < expectedOutput.length; i++) {
            double d = expectedOutput[i];
            if (d < 0 || d > 1) {
                if (d < 0)
                    expectedOutput[i] = 0 + epsilon;
                else
                    expectedOutput[i] = 1 - epsilon;
            }
        }

        int i = 0;
        for (Neuron n : outputLayer) {
            ArrayList<Connection> connections = n.getAllConnections();
            for (Connection connection : connections) {
                double pervY = connection.leftNeuron.getOutput();
                double y = n.getOutput();
                double dy = expectedOutput[i];
                double partialDerivative = (dy - y) * -y * (1 - y) * pervY;
                double deltaWeight = -learningRate * partialDerivative;
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
                    sumOutputs += (-(dy - yk) * yk * (1 - yk) * wjk);
                    j++;
                }
                double partialDerivative = y * (1 - y) * sumOutputs * pervY;
                double deltaWeight = -learningRate * partialDerivative;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
            }
        }
    }

    void run(int maxSteps, double minError) {
        int i;
        // Train neural network until minError reached or maxSteps exceeded
        double error = 1;
        for (i = 0; i < maxSteps && error > minError; i++) {
            error = 0;
            for (int p = 0; p < inputs.length; p++) {
                setInput(inputs[p]);
                activate();
                output = getOutput();
                resultOutputs[p] = output;
                for (int j = 0; j < expectedOutputs[p].length; j++) {
                    double err = Math.pow(output[j] - expectedOutputs[p][j], 2);
                    error += err;
                }
                applyBackpropagation(expectedOutputs[p]);
            }
        }

        printResult();

        System.out.println("Sum of squared errors = " + error);
        if (i == maxSteps) {
            System.out.println("!Error training try again");
        } else {
            printAllWeights();
            printWeightUpdate();
        }
    }

    void printResult() {
        System.out.println("NN example with xor training");
        for (int p = 0; p < inputs.length; p++) {
            System.out.print("INPUTS: ");
            for (int x = 0; x < layers[0]; x++) {
                System.out.print(inputs[p][x] + " ");
            }

            System.out.print("EXPECTED: ");
            for (int x = 0; x < layers[2]; x++) {
                System.out.print(expectedOutputs[p][x] + " ");
            }

            System.out.print("ACTUAL: ");
            for (int x = 0; x < layers[2]; x++) {
                System.out.print(resultOutputs[p][x] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }

    String weightKey(int neuronId, int conId) {
        return "N" + neuronId + "_C" + conId;
    }

    /**
     * Take from hash table and put into all weights
     */
    public void updateAllWeights() {
        // update weights for the output layer
        for (Neuron n : outputLayer) {
            ArrayList<Connection> connections = n.getAllConnections();
            for (Connection con : connections) {
                String key = weightKey(n.id, con.id);
                double newWeight = weightUpdate.get(key);
                con.setWeight(newWeight);
            }
        }
        // update weights for the hidden layer
        for (Neuron n : hiddenLayer) {
            ArrayList<Connection> connections = n.getAllConnections();
            for (Connection con : connections) {
                String key = weightKey(n.id, con.id);
                double newWeight = weightUpdate.get(key);
                con.setWeight(newWeight);
            }
        }
    }


    public void printWeightUpdate() {
        System.out.println("printWeightUpdate, put this i trainedWeights() and set isTrained to true");
        // weights for the hidden layer
        for (Neuron n : hiddenLayer) {
            ArrayList<Connection> connections = n.getAllConnections();
            for (Connection con : connections) {
                String w = df.format(con.getWeight());
                System.out.println("weightUpdate.put(weightKey(" + n.id + ", "
                        + con.id + "), " + w + ");");
            }
        }
        // weights for the output layer
        for (Neuron n : outputLayer) {
            ArrayList<Connection> connections = n.getAllConnections();
            for (Connection con : connections) {
                String w = df.format(con.getWeight());
                System.out.println("weightUpdate.put(weightKey(" + n.id + ", "
                        + con.id + "), " + w + ");");
            }
        }
        System.out.println();
    }

    public void printAllWeights() {
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
