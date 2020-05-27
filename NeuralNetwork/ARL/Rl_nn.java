package ARL; //change the package name as required

import static robocode.util.Utils.normalRelativeAngleDegrees;
import java.awt.Color;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import robocode.*;
import java.util.Random;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;


public class Rl_nn extends AdvancedRobot {
	final double alpha = 0.1;
    final double gamma = 0.9;
    double distance=0;
    double mutationChance = 0.1;
    int mutationNumber = 60;
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
	int populationSize = 0;

	//nn
	static int iter=0;
	double dummy=0;

	static int inputNeurons = 6;
	static int hiddenLayerNeurons = 19;
	static int outputNeurons = 1;
	double[] inputValues = new double[inputNeurons];
	double[] inputValues_next = new double[inputNeurons];
	double[] targetValues = new double[outputNeurons];

	static double[][] w_hx = new double[hiddenLayerNeurons][inputNeurons];
	String[][] w_hxs = new String[hiddenLayerNeurons][inputNeurons];

	//hidden layer output: amount of neurons + 1 bias
	static double[][] w_yh = new double[outputNeurons][hiddenLayerNeurons + 1];
	String[][] w_yhs = new String[outputNeurons][hiddenLayerNeurons + 1];

	float topParentPercent = 0.9f; //0-1 : indicateds how many percent of the parents will be selected for the next generation
	float randomWeightStandardDeviation = 5;

	//
	public void run(){
		setColors(null, Color.PINK, Color.PINK, new Color(255,165,0,100), new Color(150, 0, 150));
		setBodyColor(Color.PINK);
		while(true){
			if(explore){ //Explore event--------------------------------------------------//

				if(iter==0){

					//load command
					try {
						loadHiddenWeights();
					}
					catch (IOException e) {
						e.printStackTrace();
					}
					try {
						loadOutputWeights();
					}
					catch (IOException e) {
						e.printStackTrace();
					}
					//the loaded variable is in string converting it into double
					for(int i=0;i<hiddenLayerNeurons;i++){
						for(int j=0;j<inputNeurons;j++){
							w_hx[i][j]= Double.valueOf(w_hxs[i][j]).doubleValue();
						}
					}
					for(int i=0;i<outputNeurons;i++){
						for(int j=0;j<hiddenLayerNeurons;j++){
							w_yh[i][j]= Double.valueOf(w_yhs[i][j]).doubleValue();
						}
					}

					iter=iter+1;
				}

				NN NN_obj=new NN(w_hx, w_yh); //Neural Network Function

                // Testing Mutation
                NNRobot Robo = new NNRobot(1, NN_obj);
                NNRobot[] LonelyRobo = new NNRobot[] {Robo};
                mutateParents(LonelyRobo);
                NN newNN = LonelyRobo[0].get_NN();

				q_present_double = new double[1];
				q_next_double = new double[1];
				turnGunRight(360);
				random_action=randInt(1,total_actions.length);
				state_action_combi=qrl_x+""+qrl_y+""+qdistancetoenemy+""+q_absbearing+""+random_action;
				inputValues[0]=qrl_x;
				inputValues[1]=qrl_y;
				inputValues[2]=qdistancetoenemy;
				inputValues[3]=q_absbearing;
				inputValues[4]=random_action;
				inputValues[5]=1;

				q_present_double=NN_obj.NNfeedforward(inputValues);
				//NN.NNtrain(Xtrain, Ytrain, w_hx, w_yh,true);

					System.out.println(w_hx[0][0]);


				reward=0;
				//performing next state and scanning

				rl_action(random_action);

				turnGunRight(360);

				inputValues_next[0]=qrl_x;
				inputValues_next[1]=qrl_y;
				inputValues_next[2]=qdistancetoenemy;
				inputValues_next[3]=q_absbearing;
				inputValues_next[4]=random_action;
				inputValues_next[5]=q_next_double[0];
				q_next_double= NN_obj.NNfeedforward(inputValues_next);


				//performing update
				q_present_double[0]=q_present_double[0]+alpha*(reward+gamma*q_next_double[0]-q_present_double[0]);
				targetValues[0]=q_present_double[0];
				NN_obj.NNtrain(inputValues, targetValues);
				saveHiddenWeights();
				saveOutputWeights();

			}//explore loop ends

			//Greedy Moves//

			if(greedy){


				if(iter==0){


					try {
						loadHiddenWeights();
					}
					catch (IOException e) {
						e.printStackTrace();
					}
					//load command
					try {
						loadOutputWeights();
					}
					catch (IOException e) {
						e.printStackTrace();
					}
					//the loaded variable is in string converting it into double
					for(int i=0;i<hiddenLayerNeurons;i++){
						for(int j=0;j<inputNeurons;j++){
							w_hx[i][j]= Double.valueOf(w_hxs[i][j]).doubleValue();
						}
					}
					for(int i=0;i<outputNeurons;i++){
						for(int j=0;j<hiddenLayerNeurons+1;j++){
							w_yh[i][j]= Double.valueOf(w_yhs[i][j]).doubleValue();
						}
					}

					iter=iter+1;

				}

				NN NN_obj=new NN(w_hx, w_yh); //Neural Network Function


				//predict current state:
				turnGunRight(360);

				// finding action that produces maximum Q value


				for(int j=1;j<=total_actions.length;j++)
				{
					inputValues[0]=qrl_x;
					inputValues[1]=qrl_y;
					inputValues[2]=qdistancetoenemy;
					inputValues[3]=q_absbearing;
					inputValues[4]=j;
					inputValues[5]=1;
					q_possible[j-1]=NN_obj.NNfeedforward(inputValues)[0];
					//System.out.println(Xtrain[0][0]);
					//System.out.println(Xtrain[0][1]);
					//System.out.println(Xtrain[0][2]);
					//System.out.println(Xtrain[0][3]);
				}

				//converting table to double

				for(int i=0;i<4;i++){
					System.out.println(q_possible[i]+ "hi");
				}

				Qmax_action=getMax(q_possible)+1;
				int jj=0;

				inputValues[0]=qrl_x;
				inputValues[1]=qrl_y;
				inputValues[2]=qdistancetoenemy;
				inputValues[3]=q_absbearing;
				inputValues[4]=Qmax_action;
				inputValues[5]=1;
				System.out.println(qrl_x);
				q_present_double=NN_obj.NNfeedforward(inputValues);
				reward=0;
				//performing next state and scanning

				rl_action(Qmax_action);

				qrl_x=quantize_position(getX());
				qrl_y=quantize_position(getY());

				turnGunRight(360);
				System.out.println(qrl_x);
				for(int j=1;j<=total_actions.length;j++)
				{
					inputValues_next[0]=qrl_x;
					inputValues_next[1]=qrl_y;
					inputValues_next[2]=qdistancetoenemy;
					inputValues_next[3]=q_absbearing;
					inputValues_next[4]=j;
					inputValues_next[5]=1;
					q_possible[j-1]=NN_obj.NNfeedforward(inputValues)[0];

				}


				Qmax_action=getMax(q_possible)+1;

				inputValues_next[0]=qrl_x;
				inputValues_next[1]=qrl_y;
				inputValues_next[2]=qdistancetoenemy;
				inputValues_next[3]=q_absbearing;
				inputValues_next[4]=random_action;
				inputValues_next[5]=Qmax_action;;
				q_next_double=NN_obj.NNfeedforward(inputValues_next);

				System.out.println("h");
				//performing update
				q_present_double[0]=q_present_double[0]+alpha*(reward+gamma*q_next_double[0]-q_present_double[0]);
				targetValues=q_present_double;
				NN_obj.NNtrain(inputValues, targetValues);
			}//greedy loop ends
		}//while loop ends
	}//run function ends


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
		if(qdistancetoenemy<=2.50){fire(3);

		}
		if(qdistancetoenemy>2.50&&qdistancetoenemy<5.00){fire(3);}
		if(qdistancetoenemy>5.00&&qdistancetoenemy<7.50){fire(1);}
		//fire

		//your robot

		qrl_x=quantize_position(getX()); //your x position -state number 1
		qrl_y=quantize_position(getY()); //your y position -state number 2
		//Calculating Enemy X & Y:
		double angleToEnemy = e.getBearing();
		// Calculate the angle to the scanned robot
		double angle = Math.toRadians((getHeading() + angleToEnemy % 360));
		// Calculate the coordinates of the robot
		double enemyX = (getX() + Math.sin(angle) * e.getDistance());
		double enemyY = (getY() + Math.cos(angle) * e.getDistance());
		qenemy_x=quantize_position(enemyX);
		qenemy_y=quantize_position(enemyY);
		//distance to enemy
		//absolute angle to enemy
		absbearing=absoluteBearing((float) getX(),(float) getY(),(float) enemyX,(float) enemyY);
		q_absbearing=quantize_angle(absbearing); //state number 4


		}

	public double normalizeBearing(double angle) {
		while (angle >  180) angle -= 360;
		while (angle < -180) angle += 360;
		return angle;

	}


	public void onHitRobot(HitRobotEvent event){reward-=2;} //our robot hit by enemy robot
	public void onBulletHit(BulletHitEvent event){reward+=3;} //one of our bullet hits enemy robot
	public void onHitByBullet(HitByBulletEvent event){reward-=3;} //when our robot is hit by a bullet

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

	private double quantize_position(double rl_x2) {
			// TODO Auto-generated method stub

		if((rl_x2 > 0) && (rl_x2<=100)){
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
			}
		return rl_x2/100;

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

	public void saveHiddenWeights() {

		PrintStream hiddenWeightsStream = null;
		try {
			hiddenWeightsStream = new PrintStream(new RobocodeFileOutputStream(getDataFile("weights_hidden.txt")));
			for (int i=0;i<w_hx.length;i++) {

				String outputLine = String.valueOf(w_hx[i][0]);
				for (int j = 1; j < w_hx[i].length; j++) {
					outputLine += "    " + String.valueOf(w_hx[i][j]);
				}
				hiddenWeightsStream.println(outputLine);

			}
		} catch (IOException e) {
			e.printStackTrace();
		}finally {
			hiddenWeightsStream.flush();
			hiddenWeightsStream.close();
		}

	}

	public void saveOutputWeights() {

		PrintStream outputWeightsStream = null;
		try {
			outputWeightsStream = new PrintStream(new RobocodeFileOutputStream(getDataFile("weights_output.txt")));
			for (int i=0;i<w_yh.length;i++) {

				String outputLine = String.valueOf(w_yh[i][0]);
				for (int j = 1; j < w_yh[i].length; j++) {
					outputLine += "    " + String.valueOf(w_yh[i][j]);
				}

				outputWeightsStream.println(outputLine);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}finally {
			outputWeightsStream.flush();
			outputWeightsStream.close();
		}

	}

	public void loadHiddenWeights() throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(getDataFile("weights_hidden.txt")));
		String line = reader.readLine();
		try {
			int hidN_i=0;
			while (line != null) {
				String splitLine[] = line.split("    ");
				for (int inN_i = 0; inN_i < w_hxs[hidN_i].length; inN_i++) {
					w_hxs[hidN_i][inN_i]=splitLine[inN_i];

				}
				hidN_i++;
				line= reader.readLine();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}finally {
			reader.close();
		}
	}//load

	public void loadOutputWeights() throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(getDataFile("weights_output.txt")));
		String line = reader.readLine();
		try {
			int outN_i=0;
			while (line != null) {
				String splitLine[] = line.split("    ");
				for (int hidN_i = 0; hidN_i < w_yhs[outN_i].length; hidN_i++) {
					w_yhs[outN_i][hidN_i]=splitLine[hidN_i];
				}
				outN_i++;
				line= reader.readLine();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}finally {
			reader.close();
		}
	}//load

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
		else if(xPos>width-80){
			turnLeft(getHeading() % 90);
			if(getHeading()==0){turnLeft(90);}
			if(getHeading()==90){turnLeft(180);}
			if(getHeading()==180){turnRight(90);}
			if(getHeading()==270){turnRight(0);}
			ahead(150);
		}

	}

	public NNRobot[] initializeRobots(int numberRobots){

		NNRobot[] robotArray = new NNRobot[numberRobots];

		for (int i = 0; i < numberRobots; i++){
			NNRobot newRobot = new NNRobot(i+1);
			newRobot.set_fitness(0);

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
			newRobot.set_NN(newNN);

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

		return best_parents;
	}

    public void mutateParents(NNRobot[] Parents)
    {
        int _parentSize = Parents.length;
        NNRobot[] Children = Parents;
        for(int i=0; i < _parentSize; i++)
        {
            NN ParentNN = Parents[i].get_NN();
            for(int j = 0; j < ParentNN.w_hx.length; j++) {
                for (int k = 0; k < ParentNN.w_hx[0].length; k++) {
                    Random rand = new Random();
                    float randomFactor = rand.nextFloat();
                    if (randomFactor < mutationChance) {
                        w_hx[j][k] = rand.nextGaussian() * 2 + w_hx[j][k];
                    }
                }
            }
            for(int j = 0; j < ParentNN.w_yh.length; j++) {
                for (int k = 0; k < ParentNN.w_yh[0].length; k++) {
                    Random rand = new Random();
                    float randomFactor = rand.nextFloat();
                    if (randomFactor < mutationChance) {
                        w_yh[j][k] = rand.nextGaussian() * 2 + w_yh[j][k];
                    }
                }
            }
        }
    }

	public NNRobot[] makeEvolution(NNRobot[] robots)
    {
        NNRobot[] nextGeneration = new NNRobot[populationSize];
        NNRobot[] parents;
        parents = selectParents(robots);
        //NNRobot diverseRobot = getMostDistinct();
        nextGeneration[0] = parents[0];
        //nextGeneration[1]= diverseRobot;
        int crossoverNumber = populationSize - mutationNumber - 2;
        int i=2;
        while(i < nextGeneration.length)
        {
            if(i < crossoverNumber -2)
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
                    int randomDummy = ThreadLocalRandom.current().nextInt(0, parents.length);
                    dummyParents[j] = parents[randomDummy];
                    nextGeneration[i] = dummyParents[j];
                    i++;
                }
            }
        }
        return nextGeneration;
    }




}//Rl_nn class
