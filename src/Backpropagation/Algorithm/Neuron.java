package Backpropagation.Algorithm;

import java.util.*;

public class Neuron {
    static int counter = 0;
    final public int id;  // auto increment, starts at 0
    double output;

    ArrayList<Connection> connections = new ArrayList<>();
    HashMap<Integer, Connection> connectionLookup = new HashMap<>();

    public Neuron() {
        id = counter;
        counter++;
    }

    public void calculateOutput() {
        double v = 0;
        for (Connection con : connections) {
            Neuron leftNeuron = con.getLeftNeuron();
            double weight = con.getWeight();
            double y = leftNeuron.getOutput();
            v += weight * y;
        }
        output = sigmoid(v);
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + (Math.exp(-x)));
    }

    public void addConnections(ArrayList<Neuron> neurons) {
        for (Neuron n : neurons) {
            Connection con = new Connection(n, this);
            connections.add(con);
            connectionLookup.put(n.id, con);
        }
    }

    public Connection getConnection(int neuronIndex) {
        return connectionLookup.get(neuronIndex);
    }

    public void addConnection(Connection con) {
        connections.add(con);
    }

    public ArrayList<Connection> getAllConnections() {
        return connections;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double o) {
        output = o;
    }
}