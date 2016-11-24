package Backpropagation.Algorithm;

import java.util.*;

class Neuron {
    static int counter = 0;
    final int id;  // auto increment, starts at 0
    private double output;

    private ArrayList<Connection> connections = new ArrayList<>();
    private HashMap<Integer, Connection> connectionLookup = new HashMap<>();

    Neuron() {
        id = counter;
        counter++;
    }

    void calculateOutput() {
        double v = 0;
        for (Connection con : connections) {
            Neuron leftNeuron = con.getLeftNeuron();
            double weight = con.getWeight();
            double y = leftNeuron.getOutput();
            v += weight * y;
        }
        output = sigmoid(v);
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + (Math.exp(-x)));
    }

    void addConnections(ArrayList<Neuron> neurons) {
        for (Neuron n : neurons) {
            Connection con = new Connection(n);
            connections.add(con);
            connectionLookup.put(n.id, con);
        }
    }

    void addConnections(Neuron[] neurons) {
        for (Neuron n : neurons) {
            Connection con = new Connection(n);
            connections.add(con);
            connectionLookup.put(n.id, con);
        }
    }

    Connection getConnection(int neuronIndex) {
        return connectionLookup.get(neuronIndex);
    }

    ArrayList<Connection> getAllConnections() {
        return connections;
    }

    double getOutput() {
        return output;
    }

    void setOutput(double o) {
        output = o;
    }
}