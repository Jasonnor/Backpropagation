package Backpropagation;

import Backpropagation.Algorithm.NeuralNetwork;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.Line2D;
import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import static javax.swing.border.TitledBorder.CENTER;
import static javax.swing.border.TitledBorder.DEFAULT_POSITION;

public class MainFrame {
    private static JMenuItem loadMenuItem;
    private static JMenuItem generateMenuItem;
    private static JFrame frame;
    private JPanel layoutPanel;
    private JPanel coordinatePanel;
    private JButton loadButton;
    private JLabel loadValue;
    private JButton generateButton;
    private JTextField learningTextField;
    private JTextField thresholdTextField;
    private JLabel trainingValue;
    private JLabel testingValue;
    private JLabel weightsValue;
    private JSlider zoomerSlider;
    private JLabel timesValue;
    private JLabel fThresholdValue;
    private JTextField maxTimesValue;
    private JTextField wRangeMinValue;
    private JTextField wRangeMaxValue;
    private JTable trainTable;
    private JTable testTable;
    private DefaultTableModel trainTableModel = new DefaultTableModel();
    private DefaultTableModel testTableModel = new DefaultTableModel();
    private DecimalFormat df = new DecimalFormat("####0.00");
    private Color[] colorArray = {Color.GREEN, Color.BLUE, Color.RED, Color.YELLOW, Color.CYAN, Color.PINK};
    private NeuralNetwork network;
    private ArrayList<Double[]> inputs = new ArrayList<>();
    private ArrayList<Double[]> trainData = new ArrayList<>();
    private ArrayList<Double[]> testData = new ArrayList<>();
    private ArrayList<Double[]> weights = new ArrayList<>();
    private ArrayList<Double> outputKinds = new ArrayList<>();
    private Point mouse;
    private int maxTimes = 1000;
    private int magnification = 50;
    private double rate = 0.1;
    private double threshold = 0;
    private double minRange = -0.5;
    private double maxRange = 0.5;

    private MainFrame() {
        loadButton.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            FileNameExtensionFilter filter = new FileNameExtensionFilter("Text Files(*.txt)", "txt", "text");
            fileChooser.setFileFilter(filter);
            if (fileChooser.showOpenDialog(layoutPanel) == JFileChooser.APPROVE_OPTION) {
                loadFile(fileChooser);
            }
        });
        loadMenuItem.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            FileNameExtensionFilter filter = new FileNameExtensionFilter("Text Files(*.txt)", "txt", "text");
            fileChooser.setFileFilter(filter);
            if (fileChooser.showOpenDialog(layoutPanel) == JFileChooser.APPROVE_OPTION) {
                loadFile(fileChooser);
            }
        });
        generateButton.addActionListener(e -> startTrain());
        generateMenuItem.addActionListener(e -> startTrain());
        zoomerSlider.addChangeListener(e -> {
            zoomerSlider.setBorder(
                    BorderFactory.createTitledBorder(null,
                            Integer.toString(zoomerSlider.getValue()), CENTER, DEFAULT_POSITION));
            magnification = zoomerSlider.getValue();
            coordinatePanel.repaint();
        });
        coordinatePanel.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                super.mouseMoved(e);
                mouse = e.getPoint();
                coordinatePanel.repaint();
            }
        });
        learningTextField.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                changeRate();
            }

            public void removeUpdate(DocumentEvent e) {
                changeRate();
            }

            public void insertUpdate(DocumentEvent e) {
                changeRate();
            }

            void changeRate() {
                try {
                    alertBackground(learningTextField, false);
                    rate = Double.valueOf(learningTextField.getText());
                    startTrain();
                } catch (NumberFormatException e) {
                    alertBackground(learningTextField, true);
                    rate = 0.5f;
                }
            }
        });
        thresholdTextField.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                changeThreshold();
            }

            public void removeUpdate(DocumentEvent e) {
                changeThreshold();
            }

            public void insertUpdate(DocumentEvent e) {
                changeThreshold();
            }

            void changeThreshold() {
                try {
                    alertBackground(thresholdTextField, false);
                    threshold = Double.valueOf(thresholdTextField.getText());
                    startTrain();
                } catch (NumberFormatException e) {
                    alertBackground(thresholdTextField, true);
                    threshold = 0;
                }
            }
        });
        maxTimesValue.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                changeMaxTimes();
            }

            public void removeUpdate(DocumentEvent e) {
                changeMaxTimes();
            }

            public void insertUpdate(DocumentEvent e) {
                changeMaxTimes();
            }

            void changeMaxTimes() {
                try {
                    alertBackground(maxTimesValue, false);
                    maxTimes = Integer.valueOf(maxTimesValue.getText());
                    startTrain();
                } catch (NumberFormatException e) {
                    alertBackground(maxTimesValue, true);
                    maxTimes = 1000;
                }
            }
        });
        wRangeMinValue.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                changeMinRange();
            }

            public void removeUpdate(DocumentEvent e) {
                changeMinRange();
            }

            public void insertUpdate(DocumentEvent e) {
                changeMinRange();
            }

            void changeMinRange() {
                try {
                    if (Double.valueOf(wRangeMinValue.getText()) > maxRange)
                        alertBackground(wRangeMinValue, true);
                    else {
                        alertBackground(wRangeMinValue, false);
                        minRange = Double.valueOf(wRangeMinValue.getText());
                        startTrain();
                    }
                } catch (NumberFormatException e) {
                    alertBackground(wRangeMinValue, true);
                    minRange = -0.5f;
                }
            }
        });
        wRangeMaxValue.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                changeMaxRange();
            }

            public void removeUpdate(DocumentEvent e) {
                changeMaxRange();
            }

            public void insertUpdate(DocumentEvent e) {
                changeMaxRange();
            }

            void changeMaxRange() {
                try {
                    if (Double.valueOf(wRangeMaxValue.getText()) < minRange)
                        alertBackground(wRangeMaxValue, true);
                    else {
                        alertBackground(wRangeMaxValue, false);
                        maxRange = Double.valueOf(wRangeMaxValue.getText());
                        startTrain();
                    }
                } catch (NumberFormatException e) {
                    alertBackground(wRangeMaxValue, true);
                    maxRange = 0.5f;
                }
            }
        });
    }

    private void loadFile(JFileChooser fileChooser) {
        File loadedFile = fileChooser.getSelectedFile();
        loadValue.setText(loadedFile.getPath());
        resetFrame();
        resetData();
        try (BufferedReader br = new BufferedReader(new FileReader(loadedFile))) {
            String line = br.readLine();
            while (line != null) {
                // Split by space or tab
                String[] lineSplit = line.split("\\s+");
                // Remove empty elements
                lineSplit = Arrays.stream(lineSplit).
                        filter(s -> (s != null && s.length() > 0)).
                        toArray(String[]::new);
                Double[] numbers = new Double[lineSplit.length + 1];
                numbers[0] = -1.0;
                for (int i = 1; i <= lineSplit.length; i++)
                    numbers[i] = Double.parseDouble(lineSplit[i - 1]);
                inputs.add(numbers);
                Double output = numbers[numbers.length - 1];
                if (!outputKinds.contains(output))
                    outputKinds.add(output);
                line = br.readLine();
            }
            initialData();
            ArrayList<String> header = new ArrayList<>();
            header.add("w");
            for (int i = 1; i < trainData.get(0).length - 1; i++)
                header.add("x" + i);
            header.add("yd");
            trainTableModel.setColumnIdentifiers(header.toArray());
            testTableModel.setColumnIdentifiers(header.toArray());
            for (Double[] x : trainData)
                trainTableModel.addRow(x);
            for (Double[] x : testData)
                testTableModel.addRow(x);
            trainTable.setModel(trainTableModel);
            testTable.setModel(testTableModel);
            // TODO - show y result at data table
            generateButton.setEnabled(true);
            startTrain();
        } catch (IOException e1) {
            e1.printStackTrace();
        }
    }

    private void resetData() {
        inputs.clear();
        trainData.clear();
        testData.clear();
        outputKinds.clear();
        trainTableModel.setColumnCount(0);
        trainTableModel.setRowCount(0);
        testTableModel.setColumnCount(0);
        testTableModel.setRowCount(0);
    }

    private void initialData() {
        int[] trainKindTimes = new int[outputKinds.size()];
        int[] testKindTimes = new int[outputKinds.size()];
        for (Double[] x : inputs) {
            Double output = x[x.length - 1];
            int i;
            for (i = 0; i < outputKinds.size(); i++)
                if (output.equals(outputKinds.get(i)))
                    break;
            if (trainKindTimes[i] == 0 || testKindTimes[i] > trainKindTimes[i] / 2) {
                ++trainKindTimes[i];
                trainData.add(x);
            } else {
                ++testKindTimes[i];
                testData.add(x);
            }
        }
    }

    private void startTrain() {
        network = new NeuralNetwork(inputs, 4);
        int maxRuns = 1000;
        double minErrorCondition = 0.01;
        network.run(maxRuns, minErrorCondition);

        /*if (outputKinds.size() > 2)
            outputKinds.forEach(this::trainBackpropagation);
        else
            trainBackpropagation(outputKinds.get(0));*/
    }

    private void trainBackpropagation(Double dy) {
        if (trainData.size() == 0) return;
        Double[] w = new Double[trainData.get(0).length - 1];
        w[0] = threshold;
        for (int i = 1; i < trainData.get(0).length - 1; i++)
            w[i] = getRandomNumber();
        int wi = outputKinds.indexOf(dy);
        if (wi == 0) weights.clear();
        weights.add(w);
        int times = 0, correct = 0;
        while (times < maxTimes) {
            correct = 0;
            for (Double[] x : trainData) {
                Double sum = 0.0;
                for (int i = 0; i < weights.get(wi).length; i++) {
                    sum += weights.get(wi)[i] * x[i];
                }
                Double fx = Math.signum(sum);
                Double y = (x[x.length - 1].equals(dy)) ? 1.0 : -1.0;
                Double e = y - fx;
                if (e == 0) ++correct;
                for (int i = 0; i < weights.get(wi).length; i++) {
                    weights.get(wi)[i] = weights.get(wi)[i] + rate * e * x[i];
                }
            }
            if (correct == trainData.size()) break;
            ++times;
        }
        StringBuilder weightOutput = new StringBuilder("(");
        weightOutput.append(df.format(weights.get(wi)[1]));
        for (int i = 2; i < weights.get(wi).length; i++) {
            weightOutput.append(", ").append(df.format(weights.get(wi)[i]));
        }
        weightOutput.append(")");
        timesValue.setText(String.valueOf(times));
        weightsValue.setText(weightOutput.toString());
        fThresholdValue.setText(weights.get(wi)[0].toString());
        trainingValue.setText((double) correct / trainData.size() * 100 + "%");
        testBackpropagation(dy);
    }

    private void testBackpropagation(Double dy) {
        if (testData.size() == 0) return;
        int wi = outputKinds.indexOf(dy);
        int correct = 0;
        for (Double[] x : testData) {
            Double sum = 0.0;
            for (int i = 0; i < weights.get(wi).length; i++) {
                sum += weights.get(wi)[i] * x[i];
            }
            Double fx = Math.signum(sum);
            Double y = (x[x.length - 1].equals(dy)) ? 1.0 : -1.0;
            Double e = y - fx;
            if (e == 0) ++correct;
        }
        testingValue.setText((double) correct / testData.size() * 100 + "%");
        coordinatePanel.repaint();
    }

    private void alertBackground(JTextField textField, boolean alert) {
        if (alert)
            textField.setBackground(Color.PINK);
        else
            textField.setBackground(Color.WHITE);
    }

    private Double getRandomNumber() {
        Random r = new Random();
        return minRange + (maxRange - minRange) * r.nextDouble();
    }

    private Double[] convertCoordinate(Double[] oldPoint) {
        Double[] newPoint = new Double[2];
        newPoint[0] = (oldPoint[0] * magnification) + 250;
        newPoint[1] = 250 - (oldPoint[1] * magnification);
        return newPoint;
    }

    private static void resetFrame() {
        SwingUtilities.updateComponentTreeUI(frame);
        frame.pack();
        frame.setLocationRelativeTo(null);
    }

    private static void changeLAF(String name) {
        try {
            if (name.equals("Nimbus")) {
                for (UIManager.LookAndFeelInfo info : UIManager.getInstalledLookAndFeels())
                    if ("Nimbus".equals(info.getName())) {
                        UIManager.setLookAndFeel(info.getClassName());
                        break;
                    }
            } else {
                UIManager.setLookAndFeel(name);
            }
            resetFrame();
        } catch (ClassNotFoundException | InstantiationException | IllegalAccessException | UnsupportedLookAndFeelException e1) {
            System.out.println("Failed to load the skin!");
        }
    }

    private void createUIComponents() {
        coordinatePanel = new GPanel();
        zoomerSlider = new JSlider();
        zoomerSlider.setBorder(
                BorderFactory.createTitledBorder(null,
                        Integer.toString(zoomerSlider.getValue()), CENTER, DEFAULT_POSITION));
    }

    public static void main(String[] args) {
        JMenuBar menuBar = new JMenuBar();
        JMenu filesMenu = new JMenu("Files");
        JMenu skinsMenu = new JMenu("Skins");
        // Files menu
        loadMenuItem = new JMenuItem("Load", KeyEvent.VK_L);
        generateMenuItem = new JMenuItem("Generate", KeyEvent.VK_G);
        filesMenu.setMnemonic(KeyEvent.VK_F);
        filesMenu.add(loadMenuItem);
        filesMenu.add(generateMenuItem);
        menuBar.add(filesMenu);
        // Skins menu
        skinsMenu.setMnemonic(KeyEvent.VK_S);
        ButtonGroup group = new ButtonGroup();
        JRadioButtonMenuItem skinsMetalMenuItem = new JRadioButtonMenuItem("Metal");
        skinsMetalMenuItem.setMnemonic(KeyEvent.VK_M);
        skinsMenu.add(skinsMetalMenuItem);
        group.add(skinsMetalMenuItem);
        skinsMetalMenuItem.addActionListener(e -> changeLAF(UIManager.getCrossPlatformLookAndFeelClassName()));
        JRadioButtonMenuItem skinsDefaultMenuItem = new JRadioButtonMenuItem("Default");
        skinsDefaultMenuItem.setMnemonic(KeyEvent.VK_D);
        skinsMenu.add(skinsDefaultMenuItem);
        group.add(skinsDefaultMenuItem);
        skinsDefaultMenuItem.addActionListener(e -> changeLAF(UIManager.getSystemLookAndFeelClassName()));
        JRadioButtonMenuItem skinsMotifMenuItem = new JRadioButtonMenuItem("Motif");
        skinsMotifMenuItem.setMnemonic(KeyEvent.VK_M);
        skinsMenu.add(skinsMotifMenuItem);
        group.add(skinsMotifMenuItem);
        skinsMotifMenuItem.addActionListener(e -> changeLAF("com.sun.java.swing.plaf.motif.MotifLookAndFeel"));
        JRadioButtonMenuItem skinsGTKMenuItem = new JRadioButtonMenuItem("GTK");
        skinsGTKMenuItem.setMnemonic(KeyEvent.VK_G);
        skinsMenu.add(skinsGTKMenuItem);
        group.add(skinsGTKMenuItem);
        skinsGTKMenuItem.addActionListener(e -> changeLAF("com.sun.java.swing.plaf.gtk.GTKLookAndFeel"));
        JRadioButtonMenuItem skinsWindowsMenuItem = new JRadioButtonMenuItem("Windows");
        skinsWindowsMenuItem.setMnemonic(KeyEvent.VK_G);
        skinsMenu.add(skinsWindowsMenuItem);
        group.add(skinsWindowsMenuItem);
        skinsWindowsMenuItem.addActionListener(e -> changeLAF("com.sun.java.swing.plaf.windows.WindowsLookAndFeel"));
        JRadioButtonMenuItem skinsNimbusMenuItem = new JRadioButtonMenuItem("Nimbus");
        skinsNimbusMenuItem.setMnemonic(KeyEvent.VK_N);
        skinsNimbusMenuItem.setSelected(true);
        skinsMenu.add(skinsNimbusMenuItem);
        group.add(skinsNimbusMenuItem);
        skinsNimbusMenuItem.addActionListener(e -> changeLAF("Nimbus"));
        menuBar.add(skinsMenu);
        // Main frame
        frame = new JFrame("MainFrame");
        frame.setContentPane(new MainFrame().layoutPanel);
        frame.setIconImage(Toolkit.getDefaultToolkit().getImage("icon.png"));
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setJMenuBar(menuBar);
        changeLAF("Nimbus");
        frame.setVisible(true);
        frame.setLocationRelativeTo(null);
    }

    private class GPanel extends JPanel {
        @Override
        public void paintComponent(Graphics g) {
            super.paintComponent(g);
            g.drawLine(250, 0, 250, 500);
            g.drawLine(0, 250, 500, 250);
            Graphics2D g2 = (Graphics2D) g;
            // Draw scale
            for (Double i = 250.0; i >= 0; i -= 5.0 * magnification / 10) {
                drawScale(g2, i);
            }
            for (Double i = 250.0; i <= 500; i += 5.0 * magnification / 10) {
                drawScale(g2, i);
            }
            g2.setStroke(new BasicStroke(3));
            // Draw mouse position
            if (mouse != null) {
                Double mouse_x = (mouse.getX() - 250) / magnification;
                Double mouse_y = (250 - mouse.getY()) / magnification;
                g2.drawString("(" + df.format(mouse_x) + ", " + df.format(mouse_y) + ")", 420, 20);
            }
            // Draw point of file
            if (inputs.size() > 0 && inputs.get(0).length == 4) {
                for (Double[] x : inputs) {
                    Double[] point = convertCoordinate(new Double[]{x[1], x[2]});
                    g2.setColor(colorArray[(int) Math.round(x[x.length - 1])]);
                    g2.draw(new Line2D.Double(point[0], point[1], point[0], point[1]));
                }
            }
            g2.setStroke(new BasicStroke(2));
            // Draw line of MainFrame
            if (weights.size() != 0 && inputs.get(0).length == 4) {
                g2.setColor(Color.MAGENTA);
                for (Double[] weight : weights) {
                    Double[] lineStart, lineEnd;
                    if (weight[2] != 0) {
                        lineStart = convertCoordinate(
                                new Double[]{-250.0 / magnification,
                                        (weight[0] + 250.0 / magnification * weight[1]) / weight[2]});
                        lineEnd = convertCoordinate(
                                new Double[]{250.0 / magnification,
                                        (weight[0] - 250.0 / magnification * weight[1]) / weight[2]});
                    } else {
                        lineStart = convertCoordinate(
                                new Double[]{weight[0] / weight[1], 250.0 / magnification});
                        lineEnd = convertCoordinate(
                                new Double[]{weight[0] / weight[1], -250.0 / magnification});
                    }
                    g2.draw(new Line2D.Double(lineStart[0], lineStart[1], lineEnd[0], lineEnd[1]));
                }
            }
        }

        private void drawScale(Graphics2D g2, Double i) {
            Double[] top, btn;
            Double scaleLength = (i % (5.0 * magnification / 5) == 0) ? 2.0 * magnification / 20 : 1.0 * magnification / 20;
            top = convertCoordinate(new Double[]{(i - 250) / magnification, scaleLength / magnification});
            btn = convertCoordinate(new Double[]{(i - 250) / magnification, -scaleLength / magnification});
            g2.draw(new Line2D.Double(top[0], top[1], btn[0], btn[1]));
            top = convertCoordinate(new Double[]{-scaleLength / magnification, (250 - i) / magnification});
            btn = convertCoordinate(new Double[]{scaleLength / magnification, (250 - i) / magnification});
            g2.draw(new Line2D.Double(top[0], top[1], btn[0], btn[1]));
        }
    }
}
