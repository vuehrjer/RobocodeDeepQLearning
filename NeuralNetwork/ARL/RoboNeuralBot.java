package ARL;

import dsekercioglu.roboneural.net.*;
import robocode.AdvancedRobot;
import robocode.ScannedRobotEvent;
import robocode.util.Utils;

import java.awt.*;
import java.awt.geom.Point2D;
import java.util.ArrayList;


public class RoboNeuralBot extends AdvancedRobot {

    public static class GunUtils {

        public static double absoluteBearing(Point2D.Double l1, Point2D.Double l2) {
            return Math.atan2(l2.x - l1.x, l2.y - l1.y);
        }

        public static int limit(int min, int val, int max) {
            return Math.max(min, Math.min(val, max));
        }
    }

    public class Wave {

        double power;
        double velocity;
        double absoluteBearing;
        double distanceTraveled;
        int lateralDirection;
        double mea;
        double binWidth;
        Point2D.Double source;

        double[] input;

        public Wave(Point2D.Double source, double power, double absoluteBearing, int lateralDirection) {
            this.source = (Point2D.Double) source.clone();
            this.power = power;
            this.velocity = 20 - 3 * power;
            this.absoluteBearing = absoluteBearing;
            mea = 8 / velocity;
            binWidth = mea / BINS * 2;
            this.lateralDirection = lateralDirection;
        }

        public int update() {
            distanceTraveled += velocity;
            if (distanceTraveled > source.distance(enemyLocation)) {
                gfWavesToRemove.add(this);//These will be deleted.
                int bin = (int) Math.round(((Utils.normalRelativeAngle(GunUtils.absoluteBearing(source, enemyLocation) - absoluteBearing))
                        / (lateralDirection * binWidth)) + MIDDLE_BIN);
                return GunUtils.limit(0, bin, BINS - 1);//Hit guess factor
            }
            return -1;//If it didn't hit we will know that because of -1.
        }

        public double getFiringAngle(int firingBin) {
            return absoluteBearing + (lateralDirection * binWidth) * (firingBin - MIDDLE_BIN);//Producing a firing angle from a bin
        }
    }


    static final int BINS = 51;
    static final int MIDDLE_BIN = 25;
    static final double FIRE_POWER = 1.95;
    int[] splitNum = new int[]{9, 9};
    public static MultiLayerPerceptron mlp = new MultiLayerPerceptron(new int[]{18, BINS},
            new ActivationFunction[]{new Sigmoid()}, 0.2, 1);
    /*
        Creating a MultiLayerPerceptron is easy.
        First is how many cells for the nth layer.
        Second is the activation functions for hidden and output cells.
        Third is learning rate
        Fourth is batch size
     */

    Point2D.Double myLocation = new Point2D.Double();
    Point2D.Double enemyLocation = new Point2D.Double();

    ArrayList<Wave> gfWaves = new ArrayList<>();
    ArrayList<Wave> gfWavesToRemove = new ArrayList<>();

    public static ArrayList<double[]> neuralNetworkInput = new ArrayList<>();
    public static ArrayList<double[]> neuralNetworkOutput = new ArrayList<>();

    /*
    For saving past information to train our MLP.
    */

    public void run() { //Radar Stuff and Colors.
        setBodyColor(Color.ORANGE);
        setGunColor(Color.BLUE);
        setRadarColor(Color.CYAN);
        setScanColor(Color.CYAN);
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        for (;;) {
            turnRadarRightRadians(Double.POSITIVE_INFINITY);
        }
    }

    public void onScannedRobot(ScannedRobotEvent e) {
        myLocation.setLocation(getX(), getY());//Setting our location
        double absBearing = e.getBearingRadians() + getHeadingRadians();//For radar and targeting
        double distance = e.getDistance(); //For learning enemy location
        enemyLocation.setLocation(myLocation.x + Math.sin(absBearing) * distance,
                myLocation.y + Math.cos(absBearing) * distance);

        setTurnRadarRightRadians(Utils.normalRelativeAngle(absBearing - getRadarHeadingRadians()) * 2);//Locking the radar
        double enemyVelocity = e.getVelocity();
        double enemyLateralVelocity = (enemyVelocity * Math.sin(e.getHeadingRadians() - absBearing));//Some data
        double enemyAdvancingVelocity = (enemyVelocity * -Math.cos(e.getHeadingRadians() - absBearing));//Some data
        int enemyLateralDirection = enemyLateralVelocity >= 0 ? 1 : -1;

        //Don't forget to normalise the data between 0-1. FeatureSplitter works like that.
        double[] data = new double[]{Math.abs(enemyLateralVelocity), enemyAdvancingVelocity / 16 + 0.5};//Normal Data
        double[] preprocessedData = dsekercioglu.roboneural.format.FeatureSplitter.split(data, splitNum);
        //Preprocessing the data to make it more useful for small networks.

        Wave w = new Wave(myLocation, FIRE_POWER, absBearing, enemyLateralDirection);//Creating a wave
        w.input = preprocessedData;
        gfWaves.add(w);

        int firingBin = dsekercioglu.roboneural.format.Utils.getBin(mlp.getOutput(preprocessedData));//Getting the bestBin
        double firingAngle = w.getFiringAngle(firingBin);
        setTurnGunRightRadians(Utils.normalRelativeAngle(firingAngle - getGunHeadingRadians()));
        setFire(FIRE_POWER);
        train(); //For training the network
        updateWaves(); //For updating the waves
    }

    public void updateWaves() {
        for (Wave w : gfWaves) {
            int result = w.update();
            if (result != -1) { //Here we use the -1 to understand if the wave hit or didn't.
                neuralNetworkInput.add(0, w.input); //We add to the beginning to make training easier.
                double[] bins = new double[BINS];
                bins[result] = 1;//We set the correct gf. Others are zero initially.
                neuralNetworkOutput.add(0, bins); //Adding it.
            }
            if(neuralNetworkOutput.size() > 2000){
                neuralNetworkOutput.remove(neuralNetworkOutput.size()-1);
                neuralNetworkInput.remove(neuralNetworkInput.size()-1);
            }
        }
        gfWaves.removeAll(gfWavesToRemove);//Removing the hit waves.
        gfWavesToRemove.clear();//Clearing the list.
    }

    public void train() {
        if (!neuralNetworkInput.isEmpty()) {
            for (int i = 0; i < 25; i++) {
                int index = (int) (Math.random() * Math.min(200, neuralNetworkInput.size())); //Training the last 200 waves 25 times.
                mlp.backPropogate(neuralNetworkInput.get(index), neuralNetworkOutput.get(index));
            }
        }
    }
}
