package ARL; //change the package name as required
import java.io.File;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class NN {


	double rho;
	double alpha=0.9;
	double[][] w_hx;
	double[][] w_yh;
	double[] h;
	int inputSize_;
	int hiddenLayerSize_;
	int outputSize_;
	double[][] prev_delw_y;
	double[][] prev_delw_h;
	double curr_delw_y;


	public NN()
	{ }

	public NN(double[][] w_hx, double[][] w_yh, double rho)
	{
        this.rho = rho;
		this.inputSize_ = w_hx[0].length - 1;
		this.hiddenLayerSize_ = w_hx.length;
		this.outputSize_ = w_yh.length;
        this.w_hx = w_hx.clone();
        this.w_yh = w_yh.clone();
        for (int i = 0; i < w_hx.length; i++) {
            this.w_hx[i] = w_hx[i].clone();
        }
        for (int i = 0; i < w_yh.length; i++) {
            this.w_yh[i] = w_yh[i].clone();
        }
		prev_delw_y=new double[outputSize_][hiddenLayerSize_+1];
		prev_delw_h=new double[hiddenLayerSize_][inputSize_ + 1];
		double[] tj=new double[hiddenLayerSize_+1];
		ArrayList<Double> error = new ArrayList<Double>();
		h=new double[hiddenLayerSize_+1];
		h[hiddenLayerSize_]=1; //hardcoding bias term

	}

	public double NNfeedforward(double[] Input)
	{


		double beta_2=0;
		curr_delw_y=0;

		for (int i=0;i < hiddenLayerSize_; i++){
			for (int j=0;j<inputSize_;j++){
				prev_delw_h[i][j]=0;
			}
		}

		for (int i=0;i < outputSize_; i++){
			for (int j=0;j<hiddenLayerSize_+1;j++){
				prev_delw_y[i][j]=0;
			}
		}

		for (int i=0;i < hiddenLayerSize_; i++){

			h[i]=0;

		}
		//hardcoding bias term to 1:

		double[] y_hat=new double[outputSize_];
		for (int i=0;i < outputSize_; i++){

		}

        for (int i=0;i<hiddenLayerSize_;i++)
        {
            double s=0;
            for(int j=0;j<inputSize_ + 1;j++){
                s=s+w_hx[i][j]*Input[j];
            }
            h[i]=sigmoid(s);
        }
        double Output = 0;
        for(int i=0;i<outputSize_;i++)
        {
            double s1=0;
            for(int j=0;j<hiddenLayerSize_ + 1;j++)
            {
                s1=s1+w_yh[i][j]*h[j];
            }
            Output = s1;

        }
		return Output;

	}

	public void NNbackpropagate(double Output, double[] Input, double Target)
	{

		double[] tj = new double[hiddenLayerSize_];
		double beta_2 = (Target - Output);

			for (int j = 0; j < hiddenLayerSize_ + 1; j++) {
				for(int i = 0; i < outputSize_; i++)
				{
					curr_delw_y = beta_2 * h[j];
					w_yh[i][j] = w_yh[i][j] + rho * curr_delw_y + alpha * prev_delw_y[i][j];
					prev_delw_y[i][j] = rho * curr_delw_y + alpha * prev_delw_y[i][j];
				}
			}
			for (int j = 0; j < hiddenLayerSize_; j++) {
				double s = 0;
				for (int k = 0; k < inputSize_; k++) {
					s = s + w_hx[j][k] * Input[k];
				}
				tj[j] = s;
			}


        double[][] delw_h=new double[hiddenLayerSize_][inputSize_ + 1];

        for (int i=0;i<hiddenLayerSize_;i++){
            for (int j=0;j<inputSize_ + 1;j++){
                delw_h[i][j]=beta_2*sigmoid(tj[i])*(1-sigmoid(tj[i]))*Input[j];
                w_hx[i][j]=w_hx[i][j]+rho*delw_h[i][j]+alpha*prev_delw_h[i][j];
                prev_delw_h[i][j]=rho*delw_h[i][j]+alpha*prev_delw_h[i][j];
            }
        }
		}




	public void NNtrain(double[] Input, double Target)
	{
		double Output;
		Output = NNfeedforward(Input);
		NNbackpropagate(Output, Input, Target);
	}

	/*public static double NNtrain(double[][] Xtrain, double[] Ytrain,double[][] w_hx,double[][] w_yh,boolean switch1) {
		 
		      //variable declaration:
		      int no_h=19; //no. of hidden units
		      int r_b=no_h; //w_hx.length;
			  int c_b=6;// no. of inputs to neural network(4states+1action) + 1 (bias)=6;
		      double rho=0.00001;
		      double alpha=0.9;
		      int n_x=1;//Xtrain.length;
		      int d_x=Xtrain[0].length;
		      int n_y=Ytrain.length;
		      int d_y=1;//Ytrain[0].length;
	    	  //max and min for random number
		      double max=0.5;
		      double min=-0.5;
		      
		      
		      
		      double[][] prev_delw_y=new double[d_y][no_h+1];
		      double[][] prev_delw_h=new double[no_h][d_x];
		      double beta_2=0;
		      double curr_delw_y=0;
		      double[] tj=new double[no_h+1];
		      ArrayList<Double> error = new ArrayList<Double>();
		      
		   
		      for (int i=0;i < no_h; i++){
		    	  for (int j=0;j<d_x;j++){
		    		  prev_delw_h[i][j]=0;
		    	  }
		      }
		      
		      for (int i=0;i < d_y; i++){
		    	  for (int j=0;j<no_h+1;j++){
		    		  prev_delw_y[i][j]=0;
		    	  }
		      }
		      
		      //forward propogation:
		      double[] h=new double[no_h+1];
		      for (int i=0;i < no_h+1; i++){
		    	  
		    		  h[i]=0;
		    	  
		      }
		      //hardcoding bias term to 1:
		      h[no_h]=1;
		      double[] y_hat=new double[n_y];
		      for (int i=0;i < n_y; i++){
		    	  
		    		  y_hat[i]=0;
		    	 
		      }
		      int z=1;
		      

		      MatrixMultiplication matrix=new MatrixMultiplication();

                
                double nn_qvalue=0;
                
                if(switch1==false){//return qvalue
                	 for (int i=0;i<no_h;i++)
		    		  {
		    			  
		    			  double s=0;
		    			  for(int j=0;j<d_x;j++){
		    				  s=s+w_hx[i][j]*Xtrain[0][j];
		    			  }
		    			  h[i]=matrix.sigmoid(s);
		    		  }
		    		  
		    		  for(int i=0;i<d_y;i++)
		    		  {
		    			double s1=0;
		    			for(int j=0;j<no_h+1;j++)
		    			{
		    				s1=s1+w_yh[i][j]*h[j];
		    			}
		    			nn_qvalue = s1;
		    			
		    		  }
                	
		    		 
                	
                }
                else{//if true train weights
                	
                	  for (int i=0;i<no_h;i++)
		    		  {
		    			  
		    			  double s=0;
		    			  for(int j=0;j<d_x;j++){
		    				  s=s+w_hx[i][j]*Xtrain[0][j];
		    			  }
		    			  h[i]=matrix.sig(s);
		    		  }
		    		  
		    		  for(int i=0;i<d_y;i++)
		    		  {
		    			double s1=0;
		    			for(int j=0;j<no_h+1;j++)
		    			{
		    				s1=s1+w_yh[i][j]*h[j];
		    			}
		    			y_hat[0]=s1;
		    			
		    		  }
		    		  
		    		  for(int i=0;i<d_y;i++)
		    		  {
		    			double s1=0;
		    			for(int j=0;j<no_h+1;j++){
		    				s1=s1+w_yh[i][j]+h[j];
		    			}
		    		  }
		    		  
		    		  beta_2=(Ytrain[0]-y_hat[0]);
		    		  for (int i=0;i<w_yh[0].length;i++){
		    			 curr_delw_y=beta_2*h[i];
		    			 w_yh[0][i]=w_yh[0][i]+rho*curr_delw_y+alpha*prev_delw_y[0][i];
		    			 prev_delw_y[0][i]=rho*curr_delw_y+alpha*prev_delw_y[0][i];
		    			 
		    		  }
		    		  for (int i=0;i<no_h;i++){
		    			  double s=0;
		    			  for (int j=0;j<d_x;j++){
		    				  s=s+w_hx[i][j]*Xtrain[0][j];
		    			  }
		    			  tj[i]=s;
		    		  }
		    		  
		    		  
		    		  
		    		  double[][] delw_h=new double[r_b][c_b];
		    		  
		    		  for (int i=0;i<r_b;i++){
		    			  for (int j=0;j<c_b;j++){
		    				  delw_h[i][j]=beta_2*matrix.sigmoid(tj[i])*(1-matrix.sigmoid(tj[i]))*Xtrain[0][j];
		    				  w_hx[i][j]=w_hx[i][j]+rho*delw_h[i][j]+alpha*prev_delw_h[i][j];
		    				  prev_delw_h[i][j]=rho*delw_h[i][j]+alpha*prev_delw_h[i][j];
		    			  }
		    		  }
				      
				    
                }
				return nn_qvalue;
           	
		      
	}*/

	public double sigmoid(double x1){
		double sig;

		sig=1/(1+Math.exp(-x1));

		return sig;


	}//public static void main
	
}//main class
