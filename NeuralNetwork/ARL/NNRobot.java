package ARL;

public class NNRobot {
    private final int _ID;
    private NN _NN;
    private double[][] w_hx;
    private double[][] w_yh;
    private float _fitness;

    public NNRobot(int ID){
        this._ID = ID;
    }

    public NNRobot(int ID, NN neuralNetwork){
        this._ID = ID;
        this._NN = neuralNetwork;
    }

    public NNRobot(NNRobot robot) {
        this._ID = robot.get_ID();
        this._NN = robot.get_NN(); // TODO: Check if this is pass by value?
    }

    public void saveRobotFitness(){

    }

    public void saveRobot(){

    }

    public void loadRobot(){

    }

    public void loadRobotWeights(){

    }

    public void loadAndCalculateFitness(){

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
}
