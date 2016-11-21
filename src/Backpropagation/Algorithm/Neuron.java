package Backpropagation.Algorithm;

import java.util.*;

public class Neuron {
    static int counter = 0;
    final public int id;  // auto increment, starts at 0
    Connection thresholdConnection;
    final double threshold = -1;
    double output;

    ArrayList<Connection> connections = new ArrayList<>();
    HashMap<Integer, Connection> connectionLookup = new HashMap<>();

    public Neuron() {
        id = counter;
        counter++;
    }

    /**
     * Compute Sj = Wij*Aij + w0j*threshold
     */
    public void calculateOutput() {
        double s = 0;
        for (Connection con : connections) {
            Neuron leftNeuron = con.getLeftNeuron();
            double weight = con.getWeight();
            double a = leftNeuron.getOutput(); //output from previous layer

            s = s + (weight * a);
        }
        s = s + (thresholdConnection.getWeight() * threshold);

        output = g(s);
    }


    double g(double x) {
        return sigmoid(x);
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + (Math.exp(-x)));
    }

    public void addConnections(ArrayList<Neuron> inNeurons) {
        for (Neuron n : inNeurons) {
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

    public void addThresholdConnection(Neuron n) {
        Connection con = new Connection(n, this);
        thresholdConnection = con;
        connections.add(con);
    }

    public ArrayList<Connection> getAllConnections() {
        return connections;
    }

    public double getThreshold() {
        return threshold;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double o) {
        output = o;
    }
}