package ARL; //change the package name as required

import static robocode.util.Utils.getRandom;
import static robocode.util.Utils.normalRelativeAngleDegrees;
import java.awt.Color;
import java.io.*;

import robocode.*;
import java.util.Random;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;


public class Rl_nn extends AdvancedRobot {
	final double alpha = 0.1;
    final double gamma = 0.9;
    double distance=0;
    double mutationChance = 0.1;
    int mutationNumber = 12;
    //declaring actions
    int[] action=new int[4];
    int[] total_actions=new int[4];
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
    int random_action=0;

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

	static int inputNeurons = 5;
	static int hiddenLayerNeurons = 4;
	static int outputNeurons = 4;

	double[] inputValues = new double[inputNeurons];
	double[] inputValues_next = new double[inputNeurons];
	double[] targetValues = new double[outputNeurons];

	static double[][] w_hx = new double[hiddenLayerNeurons][inputNeurons];
	String[][] w_hxs = new String[hiddenLayerNeurons][inputNeurons];

	//hidden layer output: amount of neurons + 1 bias
	static double[][] w_yh = new double[outputNeurons][hiddenLayerNeurons + 1];
	String[][] w_yhs = new String[outputNeurons][hiddenLayerNeurons + 1];

	float topParentPercent = 0.9f; //0-1 : indicates how many percent of the parents will be selected for the next generation
	float randomWeightStandardDeviation = 5;

	int roundsPerRobot = 20;
	boolean initialized = false;
	boolean generateWeightFiles = false;
	//
	int currentRobotId = 0;

	NNRobot currentRobot;
	public void run(){

		setColors(null, Color.PINK, Color.PINK, new Color(255,165,0,100), new Color(150, 0, 150));
		setBodyColor(Color.PINK);

		if(generateWeightFiles){
			NNRobot[] initR = initializeRobots(populationSize);
			for (NNRobot r: initR){
				r.saveRobot();
			}
			resetConfig("config.txt");
			return;
		}

		reward=0;

        NN NN_obj=new NN(w_hx, w_yh);
        if (!initialized){
            //load config
            currentRobotId = selectNextRobotID("config.txt");

            //if done with generation -> create new generation
            if (currentRobotId >= populationSize){
                NNRobot[] parents = new NNRobot[populationSize];
                for (int i = 0; i < populationSize; i++) {
                    parents[i] = new NNRobot(i, NN_obj, this);
                    parents[i].loadRobot();
                }

                NNRobot[] children = makeEvolution(parents);
                for (int i = 0; i < children.length; i++) {
                    children[i].set_ID(i);
                    children[i].saveRobot();

                }

                resetConfig("config.txt");
                currentRobotId = selectNextRobotID("config.txt");
            }
            currentRobot = new NNRobot(currentRobotId, NN_obj,this);
            currentRobot.loadRobotWeights();
            initialized = true;
        }

		while(true){

			q_present_double = new double[outputNeurons];
			//q_next_double = new double[1];
			turnGunRight(360);
			//random_action=randInt(1,total_actions.length);
			//state_action_combi=qrl_x+""+qrl_y+""+qdistancetoenemy+""+q_absbearing+""+random_action;
			inputValues[0]=qrl_x;
			inputValues[1]=qrl_y;
			inputValues[2]=qdistancetoenemy;
			inputValues[3]=q_absbearing;
			//inputValues[4]=random_action;
			inputValues[4]=1;

			q_present_double=currentRobot.get_NN().NNfeedforward(inputValues);

            int actionIndex = getMax(q_present_double);

			rl_action(actionIndex);

			turnGunRight(360);

			/*inputValues_next[0]=qrl_x;
			inputValues_next[1]=qrl_y;
			inputValues_next[2]=qdistancetoenemy;
			inputValues_next[3]=q_absbearing;
			inputValues_next[4]=random_action;
			inputValues_next[5]=q_next_double[0];
			q_next_double= currentRobot.get_NN().NNfeedforward(inputValues);


			//performing update
			q_present_double[0]=q_present_double[0]+alpha*(reward+gamma*q_next_double[0]-q_present_double[0]);
			targetValues[0]=q_present_double[0];
            */
		}//while loop ends
	}//run function ends

	public void saveReward(){
		currentRobot.set_fitness((float)reward);
		currentRobot.saveRobotFitness();
		currentRobot.set_win(win);
		currentRobot.saveRobotWins();
	}

	/*public void saveWin(){
		File file = getDataFile("winrate.txt");
		try {
			RobocodeFileWriter writer = new RobocodeFileWriter(file.getAbsolutePath(), true);
			writer.write(win + "\n");
			writer.close();
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}*/

	//function definitions:
	public void onScannedRobot(ScannedRobotEvent e)
		{
		double absBearing=e.getBearingRadians()+getHeadingRadians();

		this.absBearing=absBearing;
		double getVelocity=e.getVelocity();
		double getHeadingRadians=e.getHeadingRadians();
		this.getHeadingRadians=getHeadingRadians;
		this.getVelocity=getVelocity;

		double getBearing=e.getBearing();
		this.getBearing=getBearing;
		double getTime=getTime();
		this.getTime=getTime;
		gunTurnAmt = normalRelativeAngleDegrees(e.getBearing() + (getHeading() - getRadarHeading()));
		this.gunTurnAmt=gunTurnAmt;

		double normalizeBearing=normalizeBearing(getBearing + 90 - (15 * 1));
		this.normalizeBearing=normalizeBearing;
		robot_energy=getEnergy();
		enemy_energy=e.getEnergy();
		distance = e.getDistance(); //distance to the enemy
		qdistancetoenemy=quantize_distance(distance); //distance to enemy state number 3

		//fire
		/*if(qdistancetoenemy<=2.50){fire(3); }
		if(qdistancetoenemy>2.50&&qdistancetoenemy<5.00){fire(2);}
		if(qdistancetoenemy>5.00){fire(1);}*/
		//fire

		//your robot

		qrl_x=quantize_positionX(getX()); //your x position -state number 1
		qrl_y=quantize_positionY(getY()); //your y position -state number 2
		//Calculating Enemy X & Y:
		double angleToEnemy = e.getBearing();
		// Calculate the angle to the scanned robot
		double angle = Math.toRadians((getHeading() + angleToEnemy % 360));
		// Calculate the coordinates of the robot
		double enemyX = (getX() + Math.sin(angle) * e.getDistance());
		double enemyY = (getY() + Math.cos(angle) * e.getDistance());
		qenemy_x=quantize_positionX(enemyX);
		qenemy_y=quantize_positionY(enemyY);
		//distance to enemy
		//absolute angle to enemy
		absbearing=absoluteBearing((float) getX(),(float) getY(),(float) enemyX,(float) enemyY);
		q_absbearing=quantize_angle(absbearing); //state number 4
		double bearing = getHeadingRadians() + e.getBearingRadians();
		setTurnGunRight(robocode.util.Utils.normalRelativeAngle(bearing - getGunHeadingRadians()));
		if(qdistancetoenemy<=2.50){fire(3);}
		if(qdistancetoenemy>2.50&&qdistancetoenemy<5.00){fire(2);}
		else{fire(1);}
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
		int roundNum;
		File file = getDataFile(robotAndRoundFile);
		try {
			BufferedReader reader = new BufferedReader(new FileReader(file));
			id = Integer.parseInt(reader.readLine());
			roundNum = Integer.parseInt(reader.readLine());
			reader.close();


			if (roundNum != 0 && roundNum % roundsPerRobot == 0) {
				id++;
				roundNum = 1;
			}else{
				++roundNum;
			}

			RobocodeFileWriter writer = new RobocodeFileWriter(file.getAbsolutePath(), false);
			writer.write(id + "\n");
			writer.write(roundNum + "\n");
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

	public void onHitRobot(HitRobotEvent event){reward-=2;} //our robot hit by enemy robot
	public void onBulletHit(BulletHitEvent event){reward+=3;} //one of our bullet hits enemy robot
	public void onHitByBullet(HitByBulletEvent event){reward-=3;} //when our robot is hit by a bullet

	@Override
	public void onWin(WinEvent event) {
		super.onWin(event);
		reward += 10;

		win = 1;
		saveReward();

	}

	@Override
	public void onDeath(DeathEvent event) {
		super.onDeath(event);
		reward -= 10;

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
	double absoluteBearing(float x1, float y1, float x2, float y2) {
		double xo = x2-x1;
		double yo = y2-y1;
		double hyp = Math.sqrt(Math.pow(xo,2) + Math.pow(yo,2));
		double arcSin = Math.toDegrees(Math.asin(xo / hyp));
		double bearing = 0;

		if (xo > 0 && yo > 0) { // both pos: lower-Left
			bearing = arcSin;
		} else if (xo < 0 && yo > 0) { // x neg, y pos: lower-right
			bearing = 360 + arcSin; // arcsin is negative here, actuall 360 - ang
		} else if (xo > 0 && yo < 0) { // x pos, y neg: upper-left
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
				turnGunRight(gunTurnAmt); // Try changing these to setTurnGunRight,
				turnRight(getBearing-25); // and see how much Tracker improves...
				// (you'll have to make Tracker an AdvancedRobot)
				ahead(150);
				break;
			case 4:
				turnGunRight(gunTurnAmt); // Try changing these to setTurnGunRight,
				turnRight(getBearing-25); // and see how much Tracker improves...
				// (you'll have to make Tracker an AdvancedRobot)
				back(150);
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
		reward-=3.5;
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
			double[][] weights_hidden = new double[hiddenLayerNeurons][inputNeurons]; //Create
			for (int j = 0; j < weights_hidden.length; j++){
				for (int k = 0; k < weights_hidden[0].length; k++){
					Random r = new Random();
					double randomValue = r.nextGaussian()*randomWeightStandardDeviation;
					weights_hidden[j][k] = randomValue;
				}
			}

			//Generate random weights_output array with Normal distribution
			double[][] weights_output = new double[outputNeurons][hiddenLayerNeurons+1];
			for (int j = 0; j < weights_output[0].length; j++){
				Random r = new Random();
				double randomValue = r.nextGaussian()*randomWeightStandardDeviation;
				weights_output[0][j] = randomValue;
			}
			NN newNN = new NN(weights_hidden, weights_output);
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


    public NNRobot mutateParents(NNRobot Parent)
    {

            NN ParentNN = Parent.get_NN();
            int ParentID = Parent.get_ID();
            NNRobot Child = new NNRobot(ParentID, ParentNN, this);
            for(int j = 0; j < ParentNN.w_hx.length; j++) {
                for (int k = 0; k < ParentNN.w_hx[0].length; k++) {
                    Random rand = new Random();
                    float randomFactor = rand.nextFloat();
                    if (randomFactor < mutationChance) {
                        Child.get_NN().w_hx[j][k] = rand.nextGaussian() * 2 + Child.get_NN().w_hx[j][k];
                    }
                }
            }
            for(int j = 0; j < ParentNN.w_yh.length; j++) {
                for (int k = 0; k < ParentNN.w_yh[0].length; k++) {
                    Random rand = new Random();
                    float randomFactor = rand.nextFloat();
                    if (randomFactor < mutationChance) {
                        Child.get_NN().w_yh[j][k] = rand.nextGaussian() * 2 + Child.get_NN().w_yh[j][k];
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
			for (int i = 0; i < parents.length; i++) {
				saveFitness("generationInfo.txt", parents[i].get_fitness());
			}
			saveFitness("generationInfo.txt", -101010101);
		}     catch (IOException e) {
			e.printStackTrace();
		}
		try {
			saveWinrate("winrate.txt", parents[1].get_winrate());
			//saveFitness("wins.txt", -101010101);
		}     catch (IOException e) {
			e.printStackTrace();
		}

        nextGeneration[0]= parents[0];
        nextGeneration[1] = parents[1];
        int crossoverNumber = populationSize - mutationNumber - 2;
        int i=2;
        while(i < nextGeneration.length)
        {
            if(i < crossoverNumber - 2)
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
            else
            {
                NNRobot[] dummyParents = new NNRobot[mutationNumber];
                for(int j = 0; j < mutationNumber; j++)
                {
                	if(i < nextGeneration.length) {
						int randomDummy = ThreadLocalRandom.current().nextInt(0, parents.length);
						dummyParents[j] = mutateParents(parents[randomDummy]);
						nextGeneration[i] = dummyParents[j];
						i++;
					}
                }
            }
        }
        return nextGeneration;
    }




}//Rl_nn class
