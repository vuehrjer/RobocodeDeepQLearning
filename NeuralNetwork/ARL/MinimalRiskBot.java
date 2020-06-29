package ARL;

import dsekercioglu.roboneural.net.*;

import java.awt.*;
import java.awt.geom.Point2D;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import robocode.*;
import robocode.util.Utils;

/**
 * # MLP ARRAY [3...]
 * # Activation Function enum(int) ARRAY [0 ... 6] (Length: MLP ARRAY - 1)
 * # Learning Rate () double01
 * # Batch Size int [1 ... 10]
 * # nonHitReward double
 * # hitReward double
 * # ramReward double
 */

public class MinimalRiskBot extends AdvancedRobot {
    public static MultiLayerPerceptron mlp;
    //Using Hyperbolic Tangent to have negative results.
    //This one does deep learning =).
    Point2D.Double myLocation = new Point2D.Double();
    Point2D.Double enemyLocation = new Point2D.Double();

    public static double nonHitReward;
    public static double hitReward;
    public static double ramReward;

    public static ArrayList<double[]> notHit = new ArrayList<>();//Correct moves
    public static ArrayList<double[]> hit = new ArrayList<>();//Wrong moves
    public static ArrayList<double[]> rammed = new ArrayList<>();//Very bad moves
    //We have a special list for this so our bot will always be trained against rammers.

    double[] bestResultInput = null;//The input of our current move
    boolean hitWhenGoing = true;//Did we hit a bullet while going to that point?
    /*
    If the robot got hit when trying to reach it's goal this will let us to know
    on ScannedRobot event.
     */
    double enemyEnergy = 100;

    int pointNum = 100;//Number of positions we will produce

    double myLateralVelocity;
    double myAdvancingVelocity;

    /***
     *
     */
    static int[] LayerDefinition;
    static int[] ActivationFunctionDefinitionEnum;
    static ActivationFunction[] ActivationFunctionDefinition;
    static double learningRate;
    static int batchSize;

    boolean onlyRunFittestRobot = false;
    static int currentRobotId = -1;

    public void run() { //Radar Stuff and Colors.



        if (currentRobotId == -1){
            if (onlyRunFittestRobot){
                currentRobotId = 1;
            } else{
                //load config
                currentRobotId = selectNextRobotID("config.txt");
            }
        }


        if(mlp == null){

            loadHyperparameters(currentRobotId + "hyperparams.txt");
            //new int[]{6, 5, 5, 1}
            //new ActivationFunction[]{new Tanh(), new Tanh(), new Tanh()}
            ActivationFunctionDefinition = new ActivationFunction[ActivationFunctionDefinitionEnum.length];
            for(int i = 0; i < ActivationFunctionDefinitionEnum.length; ++i){
                switch (ActivationFunctionDefinitionEnum[i]){
                    case 0:
                        ActivationFunctionDefinition[i] = new Gaussian();
                        break;
                    case 1:
                        ActivationFunctionDefinition[i] = new Linear();
                        break;
                    case 2:
                        ActivationFunctionDefinition[i] = new ReLU();
                        break;
                    case 3:
                        ActivationFunctionDefinition[i] = new Sigmoid();
                        break;
                    case 4:
                        ActivationFunctionDefinition[i] = new Sine();
                        break;
                    case 5:
                        ActivationFunctionDefinition[i] = new SmoothMax();
                        break;
                    case 6:
                        ActivationFunctionDefinition[i] = new Tanh();
                        break;
                    default:
                        throw new IllegalStateException("Unexpected value: " + ActivationFunctionDefinitionEnum[i]);
                }
            }
            mlp = new MultiLayerPerceptron(LayerDefinition, ActivationFunctionDefinition, learningRate, batchSize);
        }

        setBodyColor(Color.BLACK);
        setGunColor(Color.CYAN);
        setRadarColor(Color.BLACK);
        setScanColor(Color.BLACK);
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        for (;;) {
            turnRadarRightRadians(Double.POSITIVE_INFINITY);
        }
    }

    /**
     * MLP ARRAY
     * Activation Function enum(int) ARRAY [0 ... 6] (Length: MLP ARRAY - 1)
     * Learning Rate () double01
     * Batch Size int [1 ... 10]
     * nonHitReward double
     * hitReward double
     * ramReward double
     */
    private void loadHyperparameters(String fileName) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(getDataFile(fileName)));

            try{
                LayerDefinition = getIntArrayFromString(reader.readLine());
                ActivationFunctionDefinitionEnum = getIntArrayFromString(reader.readLine());
                learningRate = Double.parseDouble(reader.readLine());
                batchSize = Integer.parseInt(reader.readLine());
                nonHitReward = Double.parseDouble(reader.readLine());
                hitReward = Double.parseDouble(reader.readLine());
                ramReward = Double.parseDouble(reader.readLine());

            } catch (IOException e) {
                e.printStackTrace();
            }finally {
                reader.close();
            }
        } catch (IOException e){
            e.printStackTrace();
        }

    }

    @Override
    public void onWin(WinEvent event) {
        super.onWin(event);
        saveWin(1);
    }

    @Override
    public void onDeath(DeathEvent event) {
        super.onDeath(event);
        saveWin(0);

    }
    void saveWin(int win) {
        if (currentRobotId == 1) {
            try {
                File file = getDataFile("wins.txt");
                RobocodeFileWriter writer = new RobocodeFileWriter(file.getAbsolutePath(),true);
                writer.write( win + "\r\n");
                writer.close();
            } catch (IOException e){
                e.printStackTrace();
            }
        }
    }

    private int[] getIntArrayFromString( String line) {
        //strip array from arraymarkers and split into array
        String[] str = line.substring(1,line.length()-1).split(",");
        int[] array = new int[str.length];
        for (int i = 0; i < str.length; ++i ) {
            array[i] = Integer.parseInt(str[i].replaceAll("\\s+",""));
        }
        return array;
    }

    public int selectNextRobotID(String robotAndRoundFile) {
        int id = -1;
        File file = getDataFile(robotAndRoundFile);
        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            id = Integer.parseInt(reader.readLine());
            reader.close();
            RobocodeFileWriter writer = new RobocodeFileWriter(file.getAbsolutePath(), false);
            writer.write((id + 1) + "\n");
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return id;
    }

    public double[] produceData(Point2D.Double target) {//We will use this as an input to our MLP.
        double[] data = new double[6];
        data[0] = target.distance(enemyLocation) / 1200;//For teaching to stay at a controlled distance.
        data[1] = target.distance(myLocation) / 199;//If we go too far there is a lot more chance to get hit by a bullet.
        data[2] = Math.min(enemyEnergy, 100) / 100;//Setting the balance of distance
        data[3] = Math.min(getEnergy(), 100) / 100;//Setting the balance of distance
        data[4] = Math.abs(myLateralVelocity) / 8;//Learning how enemy fires?
        data[5] = myAdvancingVelocity / 16 + 0.5;//Learning how enemy fires?
        return data;
    }



    //From the wiki, GoTo page.
    /**
     * This method is very verbose to explain how things work. Do not
     * obfuscate/optimize this sample.
     */
    private void goTo(double x, double y) {
        /* Transform our coordinates into a vector */
        x -= getX();
        y -= getY();

        /* Calculate the angle to the target position */
        double angleToTarget = Math.atan2(x, y);

        /* Calculate the turn required get there */
        double targetAngle = Utils.normalRelativeAngle(angleToTarget - getHeadingRadians());

        /*
         * The Java Hypot method is a quick way of getting the length
         * of a vector. Which in this case is also the distance between
         * our robot and the target location.
         */
        double distance = Math.hypot(x, y);

        /* This is a simple method of performing set front as back */
        double turnAngle = Math.atan(Math.tan(targetAngle));
        setTurnRightRadians(turnAngle);
        if (targetAngle == turnAngle) {
            setAhead(distance);
        } else {
            setBack(distance);
        }
    }

    public void onScannedRobot(ScannedRobotEvent e) {
        enemyEnergy = e.getEnergy();//Input to NN
        myLocation.setLocation(getX(), getY());//Setting our location
        double absBearing = e.getBearingRadians() + getHeadingRadians();//For radar and targeting
        double distance = e.getDistance(); //For learning enemy location
        enemyLocation.setLocation(myLocation.x + Math.sin(absBearing) * distance,
                myLocation.y + Math.cos(absBearing) * distance);
        setTurnRadarRightRadians(Utils.normalRelativeAngle(absBearing - getRadarHeadingRadians()) * 2);//Locking the radar

        myLateralVelocity = getVelocity() * Math.sin(e.getBearingRadians());
        myAdvancingVelocity = getVelocity() * -Math.cos(e.getBearingRadians());


        ArrayList<Point2D.Double> points = new ArrayList<>();//Points will be kept here
        if (getDistanceRemaining() < 18/*If touching the target we will create another set of points*/) {
            if (!hitWhenGoing) { //This will be true if we hit a bullet when going to our destination
                notHit.add(bestResultInput);//Because we didn't get hit this is a correct move.
                if(notHit.size() > 100){
                    notHit.remove(0);
                }
            }
            hitWhenGoing = false;//Because we are changing points this won't be true for the next destination.Possibly
            for (int i = 0; i < pointNum; i++) {//C
                points.add(MoveUtils.project(myLocation,
                        Math.random() * Math.PI * 2,
                        (int) (Math.random() * 100 + 100)));//To not to sit where we are.
                        /*
                        If we sit still we will arrive to our destination in 1 tick.
                        We won't be hit possibly and it will be added to notHit list and corrupt the data.
                        */
            }
            double lowestDanger = Double.POSITIVE_INFINITY;
            Point2D.Double best = new Point2D.Double(getBattleFieldWidth()/2, getBattleFieldHeight()/2);
            for (int i = 0; i < points.size(); i++) { //Finding the bestPoint
                Point2D.Double p = points.get(i);
                if (MoveUtils.distanceToWall(p, getBattleFieldWidth(),getBattleFieldHeight()) > 22) {//We don't want to leave this to MLP.
                    double[] data = produceData(p);
                    double currentDanger = mlp.getOutput(data)[0];
                    if (currentDanger < lowestDanger) {
                        lowestDanger = currentDanger;
                        best = (Point2D.Double) p.clone();
                        bestResultInput = data.clone(); //We will use this later to train the network
                    }
                }
            }
            goTo(best.x, best.y);//Going to the position
        }
        train();//Train!

        double absBearingDeg = getHeading() + e.getBearing();
        if (absBearingDeg < 0) absBearingDeg += 360;
        double deltaX = -distance * Math.sin(Math.toRadians(absBearingDeg));
        double deltaY = -distance * Math.cos(Math.toRadians(absBearingDeg));
        double enemyHeading = e.getHeadingRadians();
        int power = 2;
        double bulletSpeed = 20 - power * 3;
        long time = (long) (distance / bulletSpeed);
        double futureX = getX() - deltaX + Math.sin(enemyHeading) * e.getVelocity() * time;
        double futureY = getY() - deltaY + Math.cos(enemyHeading) * e.getVelocity() * time;
        double absDeg = absoluteBearing(getX(), getY(), futureX, futureY);
        setTurnGunRight(normalizeBearing(absDeg - getGunHeading()));
        if (Math.abs(getGunTurnRemaining()) <= 5) {
            setTurnGunRight(0);
            setFire(power);
        }
    }
    public double normalizeBearing(double angle) {
        while (angle >  180) angle -= 360;
        while (angle < -180) angle += 360;
        return angle;

    }
    //absolute bearing
    double absoluteBearing(double x1, double y1, double x2, double y2) {
        double xo = x2 - x1;
        double yo = y2 - y1;
        double hyp = Point2D.distance(x1, y1, x2, y2);
        double arcSin = Math.toDegrees(Math.asin(xo / hyp));
        double bearing = 0;
        if (xo > 0 && yo > 0) { // both pos:lower-Left
            bearing = arcSin;
        } else if (xo < 0 && yo > 0) { // x neg, y pos: lower-right
            bearing = 360 + arcSin; // arcsin is negative here, actuall 360 -ang
        } else if (xo > 0 && yo < 0) { // x pos, y neg:upper-left
            bearing = 180 - arcSin;
        } else if (xo < 0 && yo < 0) { // both neg: upper-right
            bearing = 180 - arcSin; // arcsin is negative here, actually 180 + ang
        }
        return bearing;
    }



    public void onHitByBullet(HitByBulletEvent e) {//Hit learning
        hit.add(bestResultInput);
        if(hit.size() > 100){
            hit.remove(0);
        }
        hitWhenGoing = true;
    }

    public void onHitRobot(HitRobotEvent e) {
        if (e.isMyFault()) {//If the enemy rammed us it's not our fault.
            rammed.add(bestResultInput);
            if(rammed.size() > 100){
                rammed.remove(0);
            }
            hitWhenGoing = true;
        }
    }

    public void train() {
        for (int i = notHit.size() - 1; i > Math.max(0, notHit.size() - 15); i--) {
            //System.out.println(Arrays.toString(notHit.get(i)));
            mlp.backPropogate(notHit.get(i), new double[]{nonHitReward});
        }
        for (int i = hit.size() - 1; i > Math.max(0, hit.size() - 15); i--) {
            /*
            We want this training process to have more effect on the result
            because the robots learn.
            */
            mlp.backPropogate(hit.get(i), new double[]{hitReward});
        }
        for (int i = rammed.size() - 1; i > Math.max(0, rammed.size() - 5); i--) {
            //This training process will have even more effect on the result.
            mlp.backPropogate(rammed.get(i), new double[]{ramReward});
        }

    }

    public static class MoveUtils {

        public static Point2D.Double project(Point2D.Double source, double angle, double distance) {
            return new Point2D.Double(source.x + Math.sin(angle) * distance, source.y + Math.cos(angle) * distance);
        }

        public static double distanceToWall(Point2D.Double location, double battleFieldWidth, double ballteFieldHeight) {
            return Math.min(Math.min(location.x, battleFieldWidth - location.x), Math.min(location.y, ballteFieldHeight - location.y));

        }
    }
}
