package ARL;

import robocode.RobocodeFileOutputStream;

import java.io.*;
import java.util.ArrayList;

public class NNRobot {
    private final int _ID;
    private NN _NN;
    private float _fitness;
    private Rl_nn robotRef;

    public NNRobot(int ID){
        this._ID = ID;
    }

    public NNRobot(int ID, NN neuralNetwork, Rl_nn robotRef){
        this._ID = ID;
        this._NN = neuralNetwork;
        this.robotRef = robotRef;
    }

    public void saveRobotFitness(){
        saveFitness(_ID + "fitness.txt");
    }

    public void saveRobot(){
        //hidden weights
        saveWeights(_NN.w_hx,_ID + "weights_hidden.txt");

        //output weights
        saveWeights(_NN.w_yh,_ID + "weights_output.txt");

        resetFitness(_ID + "fitness.txt");
    }

    private void resetFitness(String fileName) {

        PrintStream fitnessStream = null;
        try {
            fitnessStream = new PrintStream(new RobocodeFileOutputStream(fileName, false));

        } catch (IOException e) {
            e.printStackTrace();

        } finally {
            fitnessStream.flush();
            fitnessStream.close();
        }

    }

    public void loadRobot(){
        loadRobotWeights();
        loadAndCalculateFitness();
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

    public int get_ID() {
        return _ID;
    }

    public float get_fitness() {
        return _fitness;
    }

    public void set_fitness(float _fitness) {
        this._fitness = _fitness;
    }


    private void saveWeights(double[][] weights, String fileName) {

        PrintStream weightsStream = null;
        try {
            weightsStream = new PrintStream(new RobocodeFileOutputStream(robotRef.getDataFile(fileName)));
            for (int i=0;i<weights.length;i++) {

                String outputLine = String.valueOf(weights[i][0]);
                for (int j = 1; j < weights[i].length; j++) {
                    outputLine += "    " + String.valueOf(weights[i][j]);
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

    private void saveFitness(String fileName){
        PrintStream fitnessStream = null;
        try {
            fitnessStream = new PrintStream(new RobocodeFileOutputStream(fileName, true));
            fitnessStream.println(_fitness);

        } catch (IOException e) {
            e.printStackTrace();

        }finally {
            fitnessStream.flush();
            fitnessStream.close();

        }
    }

    private double[] loadFitness(String fileName) throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader(robotRef.getDataFile(fileName)));
        ArrayList<String> input = new ArrayList<String>();
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
        for (double f: fitnesses) {
            sum += f;
        }
        _fitness = (float)(sum/fitnesses.length);
    }
}
