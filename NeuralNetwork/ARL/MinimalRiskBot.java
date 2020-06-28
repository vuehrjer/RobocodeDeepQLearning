package ARL;

import dsekercioglu.roboneural.net.*;

import java.awt.*;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Arrays;

import robocode.*;
import robocode.util.Utils;

public class MinimalRiskBot extends AdvancedRobot {
    public static MultiLayerPerceptron mlp = new MultiLayerPerceptron(new int[]{6, 5, 5, 1}, new ActivationFunction[]{new Tanh(), new Tanh(), new Tanh()}, 0.01, 1);
    //Using Hyperbolic Tangent to have negative results.
    //This one does deep learning =).
    Point2D.Double myLocation = new Point2D.Double();
    Point2D.Double enemyLocation = new Point2D.Double();

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

    public void run() { //Radar Stuff and Colors.
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
                if(notHit.size() > 1000){
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
            Point2D.Double best = null;
            for (int i = 0; i < points.size(); i++) { //Finding the bestPoint
                Point2D.Double p = points.get(i);
                if (MoveUtils.distanceToWall(p) > 22) {//We don't want to leave this to MLP.
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
    }

    public void onHitByBullet(HitByBulletEvent e) {//Hit learning
        hit.add(bestResultInput);
        if(hit.size() > 1000){
            hit.remove(0);
        }
        hitWhenGoing = true;
    }

    public void onHitRobot(HitRobotEvent e) {
        if (e.isMyFault()) {//If the enemy rammed us it's not our fault.
            rammed.add(bestResultInput);
            if(rammed.size() > 1000){
                rammed.remove(0);
            }
            hitWhenGoing = true;
        }
    }

    public void train() {
        for (int i = notHit.size() - 1; i > Math.max(0, notHit.size() - 15); i--) {
            System.out.println(Arrays.toString(notHit.get(i)));
            mlp.backPropogate(notHit.get(i), new double[]{-1});
        }
        for (int i = hit.size() - 1; i > Math.max(0, hit.size() - 15); i--) {
            /*
            We want this training process to have more effect on the result
            because the robots learn.
            */
            mlp.backPropogate(hit.get(i), new double[]{1});
        }
        for (int i = rammed.size() - 1; i > Math.max(0, hit.size() - 5); i--) {
            //This training process will have even more effect on the result.
            mlp.backPropogate(rammed.get(i), new double[]{1});
        }

    }

    public static class MoveUtils {

        public static Point2D.Double project(Point2D.Double source, double angle, double distance) {
            return new Point2D.Double(source.x + Math.sin(angle) * distance, source.y + Math.cos(angle) * distance);
        }

        public static double distanceToWall(Point2D.Double location) {
            return Math.min(Math.min(location.x, 800 - location.x), Math.min(location.y, 600 - location.y));
            //This robot is just for 1on1 but it uses MinimumRisk=)
        }
    }
}
