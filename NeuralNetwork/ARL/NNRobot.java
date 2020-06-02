package ARL;

import robocode.RobocodeFileOutputStream;
import robocode.RobocodeFileWriter;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

public class NNRobot {
    private int _ID;
    private NN _NN;
    private float _fitness;
    private Rl_nn robotRef;
    private int win;
    private int[] winrate;


    public NNRobot(int ID){
        this._ID = ID;
    }

    public NNRobot(int ID, NN neuralNetwork, Rl_nn robotRef){
        this._ID = ID;
        this._NN = new NN(neuralNetwork.w_hx,neuralNetwork.w_yh);
        this.robotRef = robotRef;
    }

    public NNRobot(NNRobot robot) {
        this._ID = robot.get_ID();
        this._NN = new NN(robot.get_NN().w_hx,robot.get_NN().w_yh); // TODO: Check if this is pass by value? NO ITS NOT!!
        this.robotRef = robot.robotRef;
    }

    public void saveRobotFitness(){
        try {
            saveFitness(_ID + "fitness.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public void saveRobotWins(){
        try {
            saveWins(_ID + "wins.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public void saveRobot(){
        //hidden weights
        saveWeights(_NN.w_hx,_ID + "weights_hidden.txt");

        //output weights
        saveWeights(_NN.w_yh,_ID + "weights_output.txt");

        try{
            resetFitness(_ID + "fitness.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
        try{
            resetWins(_ID + "wins.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void deleteFitness(){
        try{
            resetFitness(_ID + "fitness.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private void resetFitness(String fileName) throws IOException {

        File file = robotRef.getDataFile(fileName);
        RobocodeFileWriter writer = new RobocodeFileWriter(file.getAbsolutePath(),false);
        writer.close();
    }
    private void resetWins(String fileName) throws IOException {

        File file = robotRef.getDataFile(fileName);
        RobocodeFileWriter writer = new RobocodeFileWriter(file.getAbsolutePath(),false);
        writer.close();
    }

    public void loadRobot(){
            loadRobotWeights();
            loadAndCalculateFitness();
            loadWins();
        }

    public void loadRobotWeights(){
        try{
            //hidden weights
            loadWeights(_NN.w_hx,_ID + "weights_hidden.txt");

            //output weights
            loadWeights(_NN.w_yh,_ID + "weights_output.txt");
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void loadWins() {
        try {
            int[] winrate = loadWinrates(_ID + "wins.txt");
        }
        catch(IOException e){
            e.printStackTrace();
        }
    }

    public int[] loadWinrates(String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(robotRef.getDataFile(fileName)));
        ArrayList<String> input = new ArrayList<>();
        try{
            String line = reader.readLine();
            while (line != null) {
                input.add(line);
                line= reader.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            reader.close();
        }
        int[] returnArray = new int[input.size()];
        for (int i = 0; i < returnArray.length; i++) {

            returnArray[i] = Integer.parseInt(input.get(i));
        }
        return returnArray;
    }
    public void loadAndCalculateFitness(){

        try{
            //output weights
            double[] fitnesses = loadFitness(_ID + "fitness.txt");
            calculateFitness(fitnesses);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    public NN get_NN() {
        return _NN;
    }

    public void set_NN(NN _NN) {
        this._NN = _NN;
    }

    public int get_ID() {return _ID;}

    public void set_ID(int _ID) { this._ID = _ID; }

    public float get_fitness() {
        return _fitness;
    }

    public void set_fitness(float _fitness) {
        this._fitness = _fitness;
    }
    public void set_win(int win){
        this.win = win;
    }

    private void saveWeights(double[][] weights, String fileName) {

        PrintStream weightsStream = null;
        try {
            weightsStream = new PrintStream(new RobocodeFileOutputStream(robotRef.getDataFile(fileName)));
            for (int i=0;i<weights.length;i++) {

                String outputLine = String.valueOf(weights[i][0]);
                for (int j = 1; j < weights[i].length; j++) {
                    outputLine += "    " + weights[i][j];
                }
                weightsStream.println(outputLine);

            }
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            weightsStream.flush();
            weightsStream.close();
        }

    }

    //size of weights has to be set correctely aready!
    private void loadWeights(double[][] weights, String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(robotRef.getDataFile(fileName)));
        String line = reader.readLine();
        String[][] weightStrings = new String[weights.length][weights[0].length];
        try {
            int i=0;
            while (line != null) {
                String[] splitLine = line.split("    ");
                for (int j = 0; j < weightStrings[i].length; j++) {
                    weightStrings[i][j]=splitLine[j];

                }
                i++;
                line= reader.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            reader.close();
        }

        //convert string to double
        for(int i=0;i<weights.length;i++){
            for(int j=0;j<weights[0].length;j++){
                weights[i][j]= Double.parseDouble(weightStrings[i][j]);
            }
        }


    }
    private void saveFitness(String fileName) throws IOException{

        File file = robotRef.getDataFile(fileName);
        RobocodeFileWriter writer = new RobocodeFileWriter(file.getAbsolutePath(),true);
        writer.write(_fitness + "\n");
        writer.close();
    }

    private void saveWins(String fileName) throws IOException{

        File file = robotRef.getDataFile(fileName);
        RobocodeFileWriter writer = new RobocodeFileWriter(file.getAbsolutePath(),true);
        writer.write(win + "\n");
        writer.close();
    }


    private double[] loadFitness(String fileName) throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader(robotRef.getDataFile(fileName)));
        ArrayList<String> input = new ArrayList<>();
        try{
            String line = reader.readLine();
            while (line != null) {
                input.add(line);
                line= reader.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            reader.close();
        }
        double[] returnArray = new double[input.size()];
        for (int i = 0; i < returnArray.length; i++) {

            returnArray[i] = Double.parseDouble(input.get(i));
        }
        return returnArray;
    }


    private void calculateFitness(double[] fitnesses) {
        double sum = 0;
        Arrays.sort(fitnesses);
        if (fitnesses.length == 0){
            _fitness = 0;
        } else {
            _fitness =  (float)fitnesses[ (fitnesses.length/2)];
        }
    }

    public void initializeWeightFiles(){

        PrintStream hiddenWeightsStream = null;
        try {
            hiddenWeightsStream = new PrintStream(new RobocodeFileOutputStream(robotRef.getDataFile(_ID +"weights_hidden.txt")));
            for (int i = 0; i< Rl_nn.hiddenLayerNeurons; i++) {

                String outputLine = "0.000000000000000";
                for (int j = 1; j < Rl_nn.inputNeurons; j++) {
                    outputLine += "    0.000000000000000";
                }
                hiddenWeightsStream.println(outputLine);

            }
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            hiddenWeightsStream.flush();
            hiddenWeightsStream.close();
        }


        PrintStream outputWeightsStream = null;
        try {
            outputWeightsStream = new PrintStream(new RobocodeFileOutputStream(robotRef.getDataFile(_ID +"weights_output.txt")));
            for (int i=0;i<Rl_nn.outputNeurons;i++) {

                String outputLine = "0.000000000000000";
                for (int j = 0; j < Rl_nn.hiddenLayerNeurons; j++) {
                    outputLine += "    0.000000000000000";
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
}
