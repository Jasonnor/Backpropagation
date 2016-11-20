package Backpropagation.Util;

import javax.swing.*;
import java.awt.*;
import java.util.Random;

public class MainUtil {

    public static Double getRandomNumber(double minRange, double maxRange) {
        Random r = new Random();
        return minRange + (maxRange - minRange) * r.nextDouble();
    }

    public static Double[] convertCoordinate(Double[] oldPoint, int magnification) {
        Double[] newPoint = new Double[2];
        newPoint[0] = (oldPoint[0] * magnification) + 250;
        newPoint[1] = 250 - (oldPoint[1] * magnification);
        return newPoint;
    }

    public static void alertBackground(JTextField textField, boolean alert) {
        if (alert)
            textField.setBackground(Color.PINK);
        else
            textField.setBackground(Color.WHITE);
    }
}
