package ARL; //change the package name as required

import java.awt.Color;
import java.awt.geom.Point2D;
import java.io.*;

import robocode.*;
import robocode.util.Utils;

import java.util.Random;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;


public class Rl_nn extends AdvancedRobot {

    double distance=0;
    double mutationChance = 0.1;
    int mutationNumber = 12;
    //declaring actions
    int[] action=new int[4];
    int[] total_actions=new int[6];
    //quantized parameters
    double qrl_x=0;
    double qrl_y=0;
    double qenemy_x=0;
    double qenemy_y=0;
    double qdistancetoenemy=0;
    //-------------Explore or greedy----------------------//
    boolean explore=true; //set this true while training
    boolean greedy=false;
    //----------------------------------------------------//
    double absbearing=0;
    double q_absbearing=0;
    //initialize reward
    double reward=0;
    String state_action_combi=null;
    String state_action_combi_greedy=null;
    double robot_energy=0;



    double[] q_present_double;
    double current_q_value;
    int random_action=0;
    double next_q_value;
	double[] q_next_double;

	int Qmax_action=0;
	double[] q_possible=new double[total_actions.length];
	double enemy_energy=0;
	double gunTurnAmt;
	double bearing;
	int rlaction;
	int store_action;
	private double getHeadingRadians;
	private double getVelocity;
	private double absBearing;
	private double getBearing;
	private double getTime;
	private double normalizeBearing;
	int populationSize = 24;
	int win;
	//nn
	static int iter=0;
	double dummy=0;


	static double alpha; //0.7
	static double gamma; //0.9
	static double rho; //0.001
	static double epsilon;
	static double hitBulletReward;
	static double hitByBulletReward;
	static double hitwallReward;
	static double onDeathReward;
	static double onWinReward;
	static double hitRobotReward;


	static int inputNeurons = 8;
	static int hiddenLayerNeurons; //10
	static int outputNeurons = 1;

	static double[] hyperparams = {
			0, //alpha,
			0, //gamma,
			0, //rho,
			0, //epsilon
			0, //hitBulletReward,
			0, //hitByBulletReward,
			0, //hitRobotReward,
			0, //hitwallReward,
			0, //onDeathReward,
			0, //onWinReward,
			0, //hiddenLayerNeurons,
	};


    double[] inputValuesNew = new double[inputNeurons + 1];
	double[] inputValues = new double[inputNeurons + 1];
	double[] inputValues_next = new double[inputNeurons];
	double[] targetValues = new double[outputNeurons];

	static double[][] w_hx;
	String[][] w_hxs;

	static double[][] w_yh;
	String[][] w_yhs;

	float topParentPercent = 0.9f; //0-1 : indicates how many percent of the parents will be selected for the next generation
	float randomWeightStandardDeviation = 5;
	int actionIndex;
	int roundsPerRobot = 50;

	boolean onlyRunFittestRobot = false;
	boolean diverseSearch = true;
	boolean medianFitness = true;
	//
	static int currentRobotId = -1;

	static int winSaveAmount = 200;
	static int winArrayIndex;
	static int[] winArray;

	NNRobot currentRobot;
	private double enemyHeading = 0;
	private double velocity = 0;
	private long fireTime;
	private double deltaX = 0;
	private double deltaY = 0;
	private double enemyEnergy = 100;
	private double X = 0;
	private double Y = 0;
	private int lastAction = -1;



	public void run(){


		setColors(null, Color.PINK, Color.PINK, new Color(255,165,0,100), new Color(150, 0, 150));
		setBodyColor(Color.PINK);

		if (currentRobotId == -1){
			if (onlyRunFittestRobot){
				currentRobotId = 1;
			} else{
				//load config
				currentRobotId = selectNextRobotID("config.txt");
			}
		}

		if(hyperparams[10] == 0){

			loadHyperparameters(currentRobotId + "hyperparams.txt");
			alpha = hyperparams[0];
			gamma = hyperparams[1];
			rho = hyperparams[2];
			epsilon = hyperparams[3];
			hitBulletReward = hyperparams[4];
			hitByBulletReward = hyperparams[5];
			hitwallReward = hyperparams[6];
			onDeathReward = hyperparams[7];
			onWinReward = hyperparams[8];
			hitRobotReward = hyperparams[9];
			hiddenLayerNeurons = (int)hyperparams[10];
		}

		w_hx = new double[hiddenLayerNeurons][inputNeurons + 1];
		w_hxs = new String[hiddenLayerNeurons][inputNeurons + 1];

		//hidden layer output: amount of neurons + 1 bias
		w_yh = new double[outputNeurons][hiddenLayerNeurons + 1];
		w_yhs = new String[outputNeurons][hiddenLayerNeurons + 1];


		reward=0;

        NN NN_obj=new NN(w_hx, w_yh, rho);


		currentRobot = new NNRobot(currentRobotId, NN_obj,this);
		currentRobot.loadRobotWeights();
		setTurnRadarRightRadians(Double.POSITIVE_INFINITY);//turnRadarRight
		while(true){

			q_present_double = new double[total_actions.length];
			q_next_double = new double[total_actions.length];

			qrl_x=quantize_positionX(getX()); //your x position -state number 1
			qrl_y=quantize_positionY(getY()); //your y position -state number 2
			double q_deltaX = deltaX/getBattleFieldWidth();
			double q_deltaY = deltaY/getBattleFieldHeight();
			double myEnergy = getEnergy()/100;
			enemyEnergy = enemyEnergy/100;

			for(int i = 0; i < total_actions.length; i++)
			{
				inputValues[0] = qrl_x;
				inputValues[1] = qrl_y;
				inputValues[2] = q_deltaX;
				inputValues[3] = q_deltaY;
				inputValues[4] = myEnergy;
				inputValues[5] = enemyEnergy;
				inputValues[6] = i;
				inputValues[7] = lastAction;
				inputValues[8] = 1;

				q_present_double[i] = currentRobot.get_NN().NNfeedforward(inputValues);
			}
			reward = 0;

			actionIndex = getMax(q_present_double);
			lastAction = actionIndex;
			current_q_value = q_present_double[actionIndex];

			Random rand = new Random();
            dummy = rand.nextFloat();

			random_action = ThreadLocalRandom.current().nextInt(total_actions.length);
			if(dummy < epsilon) {
				rl_action(actionIndex);
			}
			else{
				rl_action(random_action);
			}

			execute();

			if(dummy < epsilon) {
				rl_action(actionIndex);
			}
			else{
				rl_action(random_action);
			}
			execute();
			
            qrl_x=quantize_positionX(getX()); //your x position -state number 1
            qrl_y=quantize_positionY(getY()); //your y position -state number 2


			for(int i = 0; i < total_actions.length; i++)
			{
				inputValuesNew[0] = qrl_x;
				inputValuesNew[1] = qrl_y;
				inputValuesNew[2] = q_deltaX;
				inputValuesNew[3] = q_deltaY;
				inputValuesNew[4] = myEnergy;
				inputValuesNew[5] = enemyEnergy;
				inputValuesNew[6] = i;
				inputValues[7] = lastAction;
				inputValues[8] = 1;

				q_next_double[i] = currentRobot.get_NN().NNfeedforward(inputValuesNew);
			}
            next_q_value = q_next_double[getMax(q_next_double)];
			current_q_value = current_q_value + alpha * (reward + gamma * next_q_value - current_q_value);

			if(!(epsilon == 1)) {
				currentRobot.get_NN().NNtrain(inputValues, current_q_value);  //don't train in evaluation stage
			}
		}
	}

	public void saveReward(){
		//currentRobot.set_fitness((float)reward);
		//currentRobot.saveRobotFitness();
		if (currentRobotId == 1 && epsilon != 1.0) {
			if (winArray == null){
				winArray = new int[winSaveAmount];
				winArrayIndex = 0;

			}

			winArray[winArrayIndex] = win;

			if (winArrayIndex == winSaveAmount - 1){
				//Calculate
				float winAverage = IntStream.of(winArray).sum()/(float)winSaveAmount;

				currentRobot.set_win(winAverage);
				currentRobot.saveRobotWins();
				winArrayIndex = 0;
			} else {
				++winArrayIndex;
			}
		}
		currentRobot.updateWeights();
	}

	private void loadHyperparameters(String fileName) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(getDataFile(fileName)));

			try{
				for(int i = 0; i < hyperparams.length; ++i){
					String line = reader.readLine();
					hyperparams[i] = Double.parseDouble(line);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}finally {
				reader.close();
			}
		} catch (IOException e){
			e.printStackTrace();
		}

	}

	public void doGun()
	{
		double power = Math.min(400/distance, 3);
		trackAndShoot(power);
	}
	public void doGunMAX()
	{
		double power = 3;
		trackAndShoot(power);
	}

	private void trackAndShoot(double power) {
		//setTurnRight(0);
		//setAhead(0);

		double bulletSpeed = 20 - power * 3;
		long time = (long) (distance / bulletSpeed);
		double futureX = getX() - deltaX + Math.sin(enemyHeading) * velocity * time;
		double futureY = getY() - deltaY + Math.cos(enemyHeading) * velocity * time;
		double absDeg = absoluteBearing(getX(), getY(), futureX, futureY);
		setTurnGunRight(normalizeBearing(absDeg - getGunHeading()));
		if (Math.abs(getGunTurnRemaining()) <= 2) {
			setTurnGunRight(0);
			setFire(power);
		}


	}

	public void onScannedRobot(ScannedRobotEvent e) {
		double angleToEnemy = getHeadingRadians() + e.getBearingRadians();
		double radarTurn = Utils.normalRelativeAngle(angleToEnemy - getRadarHeadingRadians());
		double extraTurn = Math.min(Math.atan(36.0 / e.getDistance()), Rules.RADAR_TURN_RATE_RADIANS);
		if (radarTurn < 0) {
			radarTurn -= extraTurn;
		} else {
			radarTurn += extraTurn;
		}
		X = getX();
		Y = getY();
		distance = e.getDistance();
		velocity = e.getVelocity();
		enemyHeading = e.getHeadingRadians();
		double absBearingDeg = getHeading() + e.getBearing();
		if (absBearingDeg < 0) absBearingDeg += 360;
		deltaX = -distance * Math.sin(Math.toRadians(absBearingDeg));
		deltaY = -distance * Math.cos(Math.toRadians(absBearingDeg));
		enemyEnergy = e.getEnergy();
		setTurnRadarRight(-getRadarTurnRemaining());

	}

	public double normalizeBearing(double angle) {
		while (angle >  180) angle -= 360;
		while (angle < -180) angle += 360;
		return angle;

	}

	public void saveFitness(String fileName, float fitness) throws IOException{
		File file = getDataFile(fileName);
		RobocodeFileWriter writer = new RobocodeFileWriter(file.getAbsolutePath(),true);
		writer.write(fitness + "\n");
		writer.close();
	}
	public void saveWinrate(String fileName, int[] winrate) throws IOException{
		File file = getDataFile(fileName);
		RobocodeFileWriter writer = new RobocodeFileWriter(file.getAbsolutePath(),true);
		for(int i = 0; i < winrate.length; i++) {
			writer.write(winrate[i] + "\n");
		}
		writer.close();
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

	public void resetConfig(String robotAndRoundFile){
		File file = getDataFile(robotAndRoundFile);
		try {
			RobocodeFileWriter writer = new RobocodeFileWriter(file.getAbsolutePath(), false);
			writer.write("0\n0\n");
			writer.close();
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void onHitRobot(HitRobotEvent event){reward+=hitRobotReward;} //our robot hit by enemy robot
	public void onBulletHit(BulletHitEvent event){		;
		reward+=hitBulletReward;
	} //one of our bullet hits enemy robot
	public void onHitByBullet(HitByBulletEvent event){reward+=hitByBulletReward;} //when our robot is hit by a bullet

	@Override
	public void onWin(WinEvent event) {
		super.onWin(event);
		reward += onWinReward;
		win = 1;
		saveReward();
	}

	@Override
	public void onDeath(DeathEvent event) {
		super.onDeath(event);
		reward += onDeathReward;

		win = 0;
		saveReward();

	}

	private double quantize_angle(double absbearing2) {

		if((absbearing2 > 0) && (absbearing2<=90)){
			q_absbearing=1;
			}
		else if((absbearing2 > 90) && (absbearing2<=180)){
			q_absbearing=2;
			}
		else if((absbearing2 > 180) && (absbearing2<=270)){
			q_absbearing=3;
			}
		else if((absbearing2 > 270) && (absbearing2<=360)){
			q_absbearing=4;
			}
		return absbearing2/90;
	}

	private double quantize_distance(double distance2) {

		if((distance2 > 0) && (distance2<=250)){
			qdistancetoenemy=1;
			}
		else if((distance2 > 250) && (distance2<=500)){
			qdistancetoenemy=2;
			}
		else if((distance2 > 500) && (distance2<=750)){
			qdistancetoenemy=3;
			}
		else if((distance2 > 750) && (distance2<=1000)){
			qdistancetoenemy=4;
			}
		qdistancetoenemy=distance2/100;
		return qdistancetoenemy;
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

	private double quantize_positionX(double rl_x2) {
			// TODO Auto-generated method stub

		/*if((rl_x2 > 0) && (rl_x2<=100)){
			qrl_x=1;
			}
		else if((rl_x2 > 100) && (rl_x2<=200)){
			qrl_x=2;
			}
		else if((rl_x2 > 200) && (rl_x2<=300)){
			qrl_x=3;
			}
		else if((rl_x2 > 300) && (rl_x2<=400)){
			qrl_x=4;
			}
		else if((rl_x2 > 400) && (rl_x2<=500)){
			qrl_x=5;
			}
		else if((rl_x2 > 500) && (rl_x2<=600)){
			qrl_x=6;
			}
		else if((rl_x2 > 600) && (rl_x2<=700)){
			qrl_x=7;
			}
		else if((rl_x2 > 700) && (rl_x2<=800)){
			qrl_x=8;
			}*/
		double width = getBattleFieldWidth();
		return rl_x2/width;

		}
	private double quantize_positionY(double rl_y2) {
		// TODO Auto-generated method stub

		/*if((rl_x2 > 0) && (rl_x2<=100)){
			qrl_x=1;
			}
		else if((rl_x2 > 100) && (rl_x2<=200)){
			qrl_x=2;
			}
		else if((rl_x2 > 200) && (rl_x2<=300)){
			qrl_x=3;
			}
		else if((rl_x2 > 300) && (rl_x2<=400)){
			qrl_x=4;
			}
		else if((rl_x2 > 400) && (rl_x2<=500)){
			qrl_x=5;
			}
		else if((rl_x2 > 500) && (rl_x2<=600)){
			qrl_x=6;
			}
		else if((rl_x2 > 600) && (rl_x2<=700)){
			qrl_x=7;
			}
		else if((rl_x2 > 700) && (rl_x2<=800)){
			qrl_x=8;
			}*/
		double height = getBattleFieldHeight();
		return rl_y2/height;

	}

	public void rl_action(int x){
		switch(x){
			case 0:
				doGun();
				break;
			case 1:
				int moveDirection=+1;  //moves in anticlockwise direction
				if (getVelocity == 0)
					moveDirection *= 1;

				// circle our enemy
				setTurnRight(getBearing + 90);
				setAhead(150 * moveDirection);
				break;
			case 2:
				int moveDirection1=-1;  //moves in clockwise direction
				if (getVelocity == 0)
					moveDirection1 *= 1;

				// circle our enemy
				setTurnRight(getBearing + 90);
				setAhead(150 * moveDirection1);
				break;
			case 3:
				//setTurnGunRight(gunTurnAmt); // Try changing these to setTurnGunRight,
				setTurnRight(getBearing-90); // and see how much Tracker improves...
				// (you'll have to make Tracker an AdvancedRobot)
				setAhead(150);
				break;
			case 4:
				//setTurnGunRight(gunTurnAmt); // Try changing these to setTurnGunRight,
				setTurnRight(getBearing-90); // and see how much Tracker improves...
				// (you'll have to make Tracker an AdvancedRobot)
				setBack(150);
				break;
			case 5:
				doGunMAX();
				break;
		}
	}//rl_action()

	public static int randInt(int min, int max) {

		// Usually this can be a field rather than a method variable
		Random rand = new Random();

		// nextInt is normally exclusive of the top value,
		// so add 1 to make it inclusive
		int randomNum = rand.nextInt((max - min) + 1) + min;

		return randomNum;
	}

	public static int getMax(double[] array){

		double largest = array[0];
		int index = 0;
		for (int i = 1; i < array.length; i++) {
		  if ( array[i] >= largest ) {
			  largest = array[i];
			  index = i;
		   }
		}
		return index;
	  }//end of getMax

	//wall smoothing (To make sure RL robot does not get stuck in the wall)
	public void onHitWall(HitWallEvent e){
		reward+=hitwallReward;
		double xPos=this.getX();
		double yPos=this.getY();
		double width=this.getBattleFieldWidth();
		double height=this.getBattleFieldHeight();
		if(yPos<80)
		{

			turnLeft(getHeading() % 90);

			if(getHeading()==0){turnLeft(0);}
			if(getHeading()==90){turnLeft(90);}
			if(getHeading()==180){turnLeft(180);}
			if(getHeading()==270){turnRight(90);}
			ahead(150);

			if ((this.getHeading()<180)&&(this.getHeading()>90))
			{
				this.setTurnLeft(90);
			}
			else if((this.getHeading()<270)&&(this.getHeading()>180))
			{
				this.setTurnRight(90);
			}
		}

		else if(yPos>height-80){ //to close to the top

			if((this.getHeading()<90)&&(this.getHeading()>0)){this.setTurnRight(90);}
			else if((this.getHeading()<360)&&(this.getHeading()>270)){this.setTurnLeft(90);}
			turnLeft(getHeading() % 90);
			if(getHeading()==0){turnRight(180);}
			if(getHeading()==90){turnRight(90);}
			if(getHeading()==180){turnLeft(0);}
			if(getHeading()==270){turnLeft(90);}
			ahead(150);

		}
		else if(xPos<80){
			turnLeft(getHeading() % 90);
			if(getHeading()==0){turnRight(90);}
			if(getHeading()==90){turnLeft(0);}
			if(getHeading()==180){turnLeft(90);}
			if(getHeading()==270){turnRight(180);}
			ahead(150);
		}
		else if(xPos>width-80) {
			turnLeft(getHeading() % 90);
			if (getHeading() == 0) {
				turnLeft(90);
			}
			if (getHeading() == 90) {
				turnLeft(180);
			}
			if (getHeading() == 180) {
				turnRight(90);
			}
			if (getHeading() == 270) {
				turnRight(0);
			}
			ahead(150);
		}

	}

	public NNRobot[] initializeRobots(int numberRobots){

		NNRobot[] robotArray = new NNRobot[numberRobots];

		for (int i = 0; i < numberRobots; i++){

			//Generate random weights_hidden array with Normal distribution
			double[][] weights_hidden = new double[hiddenLayerNeurons][inputNeurons + 1]; //Create
			for (int j = 0; j < weights_hidden.length; j++){
				for (int k = 0; k < weights_hidden[0].length; k++){
					Random r = new Random();
					double randomValue = r.nextGaussian()*randomWeightStandardDeviation;
					weights_hidden[j][k] = randomValue;
				}
			}

			//Generate random weights_output array with Normal distribution
			double[][] weights_output = new double[outputNeurons][hiddenLayerNeurons+1];
			for (int j = 0; j < weights_output.length; j++){
				for (int k = 0; k < weights_output[0].length; k++){
					Random r = new Random();
					double randomValue = r.nextGaussian()*randomWeightStandardDeviation;
					weights_output[j][k] = randomValue;
				}
			}
			NN newNN = new NN(weights_hidden, weights_output, rho);
			NNRobot newRobot = new NNRobot(i, newNN, this);
			newRobot.set_fitness(0);
			robotArray[i] = newRobot;
		}
		return robotArray;

	}

	public NNRobot[] selectParents(NNRobot[] robots){ //Returns the best 'topParentPercent' % of the input NNrobots sorted by their fitness values

		int out_amount = (int)((float)robots.length * topParentPercent);
		NNRobot[] best_parents = new NNRobot[out_amount]; //Create an array of NN_robots with 'topParentPercent' % of the input robots

		Arrays.sort(robots, new Comparator<NNRobot>() { //Sort robots by their fitness values
			@Override
			public int compare(NNRobot r1, NNRobot r2) {
				return Float.compare(r2.get_fitness(), r1.get_fitness());
			}
		});

		for (int i = 0; i < best_parents.length; i++){ //Fill best_parents array with the best robots
			best_parents[i] = robots[i];
		}


		//Get robot with biggest diversity to best robot
		int best_diversity_index = 1;
		double best_diversity = 0;

		for (int i = 1; i < best_parents.length; i++){
			double total_diversity = 0;

			for (int j = 0; j < best_parents[i].get_NN().w_hx.length; j++){
				for (int k = 0; k < best_parents[i].get_NN().w_hx[0].length; k++){
					double diversity = Math.pow(best_parents[0].get_NN().w_hx[j][k] - best_parents[i].get_NN().w_hx[j][k], 2);

					total_diversity += diversity;
				}
			}
			for (int j = 0; j < best_parents[i].get_NN().w_yh.length; j++){
				for (int k = 0; k < best_parents[i].get_NN().w_yh[0].length; k++){
					double diversity = Math.pow(best_parents[0].get_NN().w_yh[j][k] - best_parents[i].get_NN().w_yh[j][k], 2);

					total_diversity += diversity;
				}
			}

			if (total_diversity >=  best_diversity){
				best_diversity_index = i;
				best_diversity = total_diversity;
			}
		}

		//Put robot with biggest diversity from best robot to the first position of the output array
		NNRobot[] output_array = new NNRobot[out_amount];
		output_array[0] = best_parents[best_diversity_index];

		int best_parents_index = 0;
		for (int i = 1; i < output_array.length; i++){
			if (i == best_diversity_index){
				best_parents_index++;
			}
			output_array[i] = best_parents[best_parents_index];
			best_parents_index++;
		}
		return output_array;
	}


	// Evolution stuff
	// --------------------------------------------------------------

	// Crosses the provided parents' attributes and returns an equal amount of children
	// made up of randomized permutations. No resulting child will look like one of the parents.
	public NNRobot[] crossover(NNRobot father, NNRobot mother) {
		// Get the parents weights from the neural networks


		double[][] father_weights_hidden = mother.get_NN().w_hx;
		double[][] mother_weights_hidden = mother.get_NN().w_hx;
		double[][] father_weights_output = father.get_NN().w_yh;
		double[][] mother_weights_output = mother.get_NN().w_yh;

		// Check for compatibility of parents
		if (father_weights_hidden.length != mother_weights_hidden.length) {
			return null;
		}

		if (father_weights_output.length != mother_weights_output.length) {
			return null;
		}

		NNRobot son = new NNRobot(father);
		NNRobot daughter = new NNRobot(mother);

		// Iterate over each set of hidden weights
		for (int i = 0; i < father_weights_hidden.length; i++) {
			// Get randomized split index for crossover
			int split = ThreadLocalRandom.current().nextInt(0, father_weights_hidden[i].length);

			// Swap the parent weights from the split index onwards
			while (split < father_weights_hidden[i].length) {
				son.get_NN().w_hx[i][split] = mother_weights_hidden[i][split];
				daughter.get_NN().w_hx[i][split] = father_weights_hidden[i][split];
				split++;
			}
		}

		// Iterate over each set of output weights
		for (int i = 0; i < father_weights_output.length; i++) {
			// Get randomized split index for crossover
			int split = ThreadLocalRandom.current().nextInt(0, father_weights_output[i].length);

			// Swap the parent weights from the split index onwards
			while (split < father_weights_output[i].length) {
				son.get_NN().w_yh[i][split] = mother_weights_output[i][split];
				daughter.get_NN().w_yh[i][split] = father_weights_output[i][split];
				split++;
			}
		}

		return new NNRobot[]{son, daughter};
	}


    public NNRobot mutateParents(NNRobot Parent, int amplification)
    {

            NN ParentNN = Parent.get_NN();
            int ParentID = Parent.get_ID();
            NNRobot Child = new NNRobot(ParentID, ParentNN, this);
            for(int j = 0; j < ParentNN.w_hx.length; j++) {
                for (int k = 0; k < ParentNN.w_hx[0].length; k++) {
                    Random rand = new Random();
                    float randomFactor = rand.nextFloat();
                    if (randomFactor < mutationChance + (Math.log10(amplification))/(populationSize/2)) { //mutation chance is higher for bad robots
                        Child.get_NN().w_hx[j][k] = rand.nextGaussian() * (Math.log10(100*amplification) + 0.1) + Child.get_NN().w_hx[j][k]; // mutation is stronger for bad robots
                    }
                }
            }
            for(int j = 0; j < ParentNN.w_yh.length; j++) {
                for (int k = 0; k < ParentNN.w_yh[0].length; k++) {
                    Random rand = new Random();
                    float randomFactor = rand.nextFloat();
                    if (randomFactor < mutationChance + (Math.log10(amplification))/(populationSize/2)) {
                        Child.get_NN().w_yh[j][k] = rand.nextGaussian() * (Math.log10(100*amplification) + 0.1) + Child.get_NN().w_yh[j][k];
                    }
                }
            }
            return Child;

    }

	public NNRobot[] makeEvolution(NNRobot[] robots)
    {
        NNRobot[] nextGeneration = new NNRobot[populationSize];
        NNRobot[] parents;
        parents = selectParents(robots);

		try {
			saveFitness("generationInfo.txt", parents[1].get_fitness());
		}     catch (IOException e) {
			e.printStackTrace();
		}
		try {
			saveWinrate("winrate.txt", parents[1].get_winrate());
			//saveFitness("wins.txt", -101010101);
		}     catch (IOException e) {
			e.printStackTrace();
		}
		if (diverseSearch){
			nextGeneration[0]= parents[0];
		}else{
			nextGeneration[0]= parents[1];
		}
        nextGeneration[1] = parents[1];
        int crossoverNumber = populationSize - parents.length + 2;

        if(crossoverNumber % 2 == 0)
		{
			mutationNumber = parents.length;
		}
        else{
        	mutationNumber = parents.length - 1;
		}

		int i=2;
		while(i < mutationNumber)
		{
				nextGeneration[i] = mutateParents(parents[i], i);
				i++;

		}
        while(i < nextGeneration.length - 1)
        {

                NNRobot[] children;
                int random1 = ThreadLocalRandom.current().nextInt(0, parents.length);
                int random2 = ThreadLocalRandom.current().nextInt(0, parents.length);
                NNRobot parent1 = parents[random1];
                NNRobot parent2 = parents[random2];
                children = crossover(parent1, parent2);
                nextGeneration[i] = children[0];
                nextGeneration[i+1] = children[1];
                i+=2;
        }
        return nextGeneration;
    }




}//Rl_nn class
