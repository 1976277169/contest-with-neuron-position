import java.io.*;
import java.io.FileWriter;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class DN {

	private int numSensor;
	private SensorLayer[] sensor;
	
	private int numMotor;
	private MotorLayer[] motor;
	
	private int[] mHiddentopk;
	private int numHidden;
	private int[] numHiddenNeurons;
	private HiddenLayer[] hidden;

	public FileWriter fw;
	
	public DN(int numInput, int[][] inputSize, int numMotor, int[][] motorSize,int[] topKMotor, int numHidden, int[] numHiddenNeurons, int[] topKHidden){
		
		// Initialize the layers

		// total sizes of the sensor 
		int totalSensor = totalSize(inputSize);
		int totalMotor = totalSize(motorSize);
		
		// Initialize the hidden layers
		this.mHiddentopk = topKHidden;
		this.numHidden = numHidden;
		this.numHiddenNeurons = numHiddenNeurons;
		hidden = new HiddenLayer[numHidden];
		
		for (int i = 0; i < numHidden; i++) {
			hidden[i] = new HiddenLayer(numHiddenNeurons[i], topKHidden[i], totalSensor, totalMotor);
		}
		
		int totalHidden = totalSize(hidden);
		
		// Initialize the sensor layers
		this.numSensor = numInput;
		sensor = new SensorLayer[numSensor];
		
		for (int i = 0; i < numSensor; i++) {
			sensor[i] = new SensorLayer(inputSize[i][0], inputSize[i][1]);
		}
		
		// Initialize the motor layers
		this.numMotor = numMotor;
		motor = new MotorLayer[numMotor];
		
		for (int i = 0; i < numMotor; i++) {
			motor[i] = new MotorLayer(motorSize[i][0], motorSize[i][1], topKMotor[i], totalHidden);
		}
		
	}
	
	public void setSensorInput(int index, float[][] input){
		sensor[index].setInput(input);
	}
	
	public void setMotorInput(int index, float[][] input){
		motor[index].setInput(input);
	}
	
	public void computeHiddenResponse(float[][][] sensorInput, float[][][] motorInput ){

		// set the input and motors
		float[][] allSensorInput = new float[numSensor][];
		int[] allSensorSize = new int[numSensor];
		for (int i = 0; i < numSensor; i++) {
			sensor[i].setInput(sensorInput[i]);
			
			allSensorInput[i] = sensor[i].getInput1D();
			allSensorSize[i] =  allSensorInput[i].length;
		}
		

		// update the motor input
		float[][] allMotorInput = new float[numMotor][];
		int[] allMotorSize = new int[numMotor];
		for (int i = 0; i < numMotor; i++) {
			motor[i].setInput(motorInput[i]);
			allMotorInput[i] = motor[i].getInput1D();
			allMotorSize[i] =  allMotorInput[i].length;
		}
		
		// computes the new response for Y
		for (int i = 0; i < numHidden; i++) {
			
			hidden[i].computeBottomUpResponse(allSensorInput, allSensorSize);
			hidden[i].computeTopDownResponse(allMotorInput, allMotorSize);
			hidden[i].computeResponse();
			
			System.out.println("HiddenResponse");
			displayResponse(hidden[i].getResponse1D());
			
			hidden[i].hebbianLearnHidden(inputToWeights(allSensorInput, allSensorSize), inputToWeights(allMotorInput, allMotorSize));
		}
		try{ 
			fw= new FileWriter("Speech_Data/Hidden_winner_Index.txt",true);
			}catch (IOException e) {
	            e.printStackTrace();
	        }

	}
	
	private void displayResponse(float[] r) {
			System.out.print(r[0]);
			for (int j = 1; j < r.length; j++) {
				System.out.print("," + r[j]);
			}
			System.out.println();
	}

	// compute motor response only, no update for the weights
	public float[][] computeMotorResponse(int motorIndex){
		
		float[][] response;
		
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numHidden][];
		int[] allHiddenSize = new int[numHidden];
		
		for (int i = 0; i < numHidden; i++) {
			allHiddenInput[i] = hidden[i].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;
			
		}
		
		// once the final responses are set, compute the motor responses
		
		motor[motorIndex].computeResponse(inputToWeights(allHiddenInput, allHiddenSize));

			
			//System.out.println(maxIndex(motor[i].getResponse1D()));
			
			//if(maxIndex(motor[i].getResponse1D()) != -1)
			//	motorLearn();
			
			//Get the motor responses
		response = motor[motorIndex].getNewMotorResponse2D();
		
		return response;
		
		
	}
	
	// compute motor response only, no update for the weights
	public float[][][] computeMotorResponse(){
		float[][][] response = new float[numMotor][][];
		
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numHidden][];
		int[] allHiddenSize = new int[numHidden];
		
		for (int i = 0; i < numHidden; i++) {
			allHiddenInput[i] = hidden[i].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;
			
		}
		
		// once the final responses are set, compute the motor responses
		for (int i = 0; i < numMotor; i++) {
			motor[i].computeResponse(inputToWeights(allHiddenInput, allHiddenSize));

			
			//System.out.println(maxIndex(motor[i].getResponse1D()));
			
			//if(maxIndex(motor[i].getResponse1D()) != -1)
			//	motorLearn();
			
			//Get the motor responses
			response[i] = motor[i].getNewMotorResponse2D();
		}
		
		
		return response;
		
		
	}
	
	// Set the new motor response then call motorLearn
	public void updateSupervisedMotorWeights(float[][][] supervisedMotor){
		
		for (int i = 0; i < supervisedMotor.length; i++) {
			
			motor[i].setSupervisedResponse(supervisedMotor[i]);
		}
		
		updateMotorWeights();
	}
	
	// Set the new motor response then call motorLearn
	public void updateSupervisedMotorWeights(int motorIndex, float[][] supervisedMotor){
		
		motor[motorIndex].setSupervisedResponse(supervisedMotor);
		
		
		updateMotorWeights(motorIndex);
	}
	
	public void replaceHiddenResponse(){
		for (int i = 0; i < hidden.length; i++) {
			hidden[i].replaceHiddenLayerResponse();
		}
	}
	
	public void replaceMotorResponse(){
		for (int i = 0; i < motor.length; i++) {
			motor[i].replaceMotorLayerResponse();
		}
	}
	
	// Does the hebbian learning based on the computed response.
	public void updateMotorWeights(){
		
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numHidden][];
		int[] allHiddenSize = new int[numHidden];
		
		for (int i = 0; i < numHidden; i++) {
			allHiddenInput[i] = hidden[i].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;	
		}
		
		for (int i = 0; i < numMotor; i++) {
			motor[i].hebbianLearnMotor(inputToWeights(allHiddenInput, allHiddenSize));
			
		}
	}
	

	// Does the hebbian learning based on the computed response.
	public void updateMotorWeights(int motorIndex){
		
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numHidden][];
		int[] allHiddenSize = new int[numHidden];
		
		for (int i = 0; i < numHidden; i++) {
			allHiddenInput[i] = hidden[i].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;	
		}
		
		
		motor[motorIndex].hebbianLearnMotor(inputToWeights(allHiddenInput, allHiddenSize));
			
		
	}
	
	private int maxIndex(float[] values){
		int index = (values[0] != 0.0f) ? 0:-1;
		
		float max = values[0];
		
		for (int i = 0; i < values.length; i++) {
			if(values[i] > max){
				max = values[i];
				index = i;
			}
		}
		
		return index;
	}
	
	private float[] inputToWeights(float[][] input, int[] size){
		
		int total = 0;
		// compute the total size
		for (int i = 0; i < size.length; i++) {
			total += size[i];
		}
		
		float[] weights = new float[total];
			
		int beginIndex = 0;
		for (int j = 0; j < input.length; j++) {
			System.arraycopy(input[j], 0, weights, beginIndex, size[j]);
			beginIndex += size[j];
		}
			
		return weights;
		
	}
	
	private int totalSize(HiddenLayer[] hidden2){
		int total = 0;
		
		for (int i = 0; i < hidden2.length; i++) {
			total += (hidden2[i].getNumNeurons());
		}
		
		return total;
	}

	private int totalSize(int[][] size){
		int total = 0;
		
		for (int i = 0; i < size.length; i++) {
			total += (size[i][0] * size[i][1]);
		}
		
		return total;
	}

	public int getNumSensor() {
		return numSensor;
	}


	public void setNumSensor(int numSensor) {
		this.numSensor = numSensor;
	}


	public int getNumMotor() {
		return numMotor;
	}


	public void setNumMotor(int numMotor) {
		this.numMotor = numMotor;
	}


	public int getNumHidden() {
		return numHidden;
	}


	public void setNumHidden(int numHidden) {
		this.numHidden = numHidden;
	}


	public SensorLayer[] getSensor() {
		return sensor;
	}


	public void setSensor(SensorLayer[] sensor) {
		this.sensor = sensor;
	}


	public MotorLayer[] getMotor() {
		return motor;
	}


	public void setMotor(MotorLayer[] motor) {
		this.motor = motor;
	}


	public HiddenLayer[] getHidden() {
		return hidden;
	}


	public void setHidden(HiddenLayer[] hidden) {
		this.hidden = hidden;
	}
	
	public void updateHiddenLocation(){
		for(int i = 0; i < numHidden; i++){
			hidden[i].pullneurons();
		}
	}
	
	public void outputHiddenLocation(){
		String filename = "Speech_Data/hidden_location.txt";
		try {
			   PrintWriter wr = new PrintWriter(new File(filename));
			   
			    wr.println("Glial cells Location");
			    
					for (int i = 0; i < numHidden; i++) {
						for (int j = 0; j < hidden[i].numGlial; j++) {
							int index = hidden[i].glialcells[j].getindex();
							float[] temp1 = hidden[i].glialcells[j].getlocation();
							//wr.println("Glialcell "+index+"'s location: ("+temp1[0]+", "+temp1[1]+", "+temp1[2]+")");
							wr.println(temp1[0]+", "+temp1[1]+", "+temp1[2]);
						}
						wr.println();
			    }
				wr.println("Hidden Neourons Location");
				for (int i = 0; i < numHidden; i++) {
					wr.println("Hidden area: " + i);
					for (int j = 0; j < numHiddenNeurons[i]; j++) {
						int index = hidden[i].hiddenNeurons[j].getindex();
						float[] temp2 = hidden[i].hiddenNeurons[j].getlocation();
						//wr.println("Neuron "+index+"'s location: ("+temp2[0]+", "+temp2[1]+", "+temp2[2]+")");
						wr.println(temp2[0]+" "+temp2[1]+" "+temp2[2]);
					}
					
					wr.println();
				}
				wr.close();
			}catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
	}
	
	public void trackHiddenwinners(int time) throws FileNotFoundException{
		PrintWriter wwr = new PrintWriter(fw);
		    
				for (int i = 0; i < numHidden; i++) {
					//wwr.println("For hidden layer"+i);
				    int[] index = new int[mHiddentopk[i]];
				    index = hidden[i].getwinnerIndexs();
				    String tempp = "";
				    for(int j = 0; j < mHiddentopk[i]; j++){
				    	tempp = tempp+" "+index[j];
				    }
						//wwr.println("Time"+time+", the top"+mHiddentopk[i]+" winner neurons' indexs are:"+tempp);
				     wwr.println(time+" " +tempp);
					//wwr.println();
		    }

			wwr.close();
	}
	
	public void trackHiddenweights(){
		String filename = "Speech_Data/bottom_up_weight.txt";
		try {
			   
			  PrintWriter wwr = new PrintWriter(new File(filename));
		    	
		    
				for (int i = 0; i < numHidden; i++) {
					//wwr.println("For hidden layer"+i);
					int num = hidden[i].getNumNeurons();
				    float[] buweight = new float[num];
				    for(int j = 0; j < num; j++){
				    	buweight = hidden[i].hiddenNeurons[j].getBottomUpWeights();
				    	String temp = "";
				    	for(int k = 0; k < buweight.length; k++){
				    		temp = temp+Float.toString(buweight[k])+" ";
				    	}
				    	wwr.println(temp);
				    }
						//wwr.println("Time"+time+", the top"+mHiddentopk[i]+" winner neurons' indexs are:"+tempp);
				     
					//wwr.println();
		    }

			wwr.close();
		}catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	public void trackHiddenweights2(){
		String filename = "Speech_Data/top_down_weight.txt";
		try {
			   
			  PrintWriter wwr = new PrintWriter(new File(filename));
		    	
		    
				for (int i = 0; i < numHidden; i++) {
					//wwr.println("For hidden layer"+i);
					int num = hidden[i].getNumNeurons();
				    float[] buweight = new float[num];
				    for(int j = 0; j < num; j++){
				    	buweight = hidden[i].hiddenNeurons[j].getTopDownWeights();
				    	String temp = "";
				    	for(int k = 0; k < buweight.length; k++){
				    		temp = temp+Float.toString(buweight[k])+" ";
				    	}
				    	wwr.println(temp);
				    }
						//wwr.println("Time"+time+", the top"+mHiddentopk[i]+" winner neurons' indexs are:"+tempp);
				     
					//wwr.println();
		    }

			wwr.close();
		}catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void trackHiddenage(int t){
		String filename;
		if(t==1){
		 filename = "Speech_Data/training_age.txt";}
		else{
			filename = "Speech_Data/test_age.txt";}
		try {
			   
			  PrintWriter wwr = new PrintWriter(new File(filename));
		    	
		    
				for (int i = 0; i < numHidden; i++) {
					//wwr.println("For hidden layer"+i);
					int num = hidden[i].getNumNeurons();
				    int age = 0;
				    for(int j = 0; j < num; j++){
				    	age = hidden[i].hiddenNeurons[j].getfiringage();
				    	
				    	wwr.println(age);
				    }
						//wwr.println("Time"+time+", the top"+mHiddentopk[i]+" winner neurons' indexs are:"+tempp);
				     
					//wwr.println();
		    }

			wwr.close();
		}catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}	
	
	public void trackMotorage(int t){
		String filename;
		if(t==1){
		 filename = "Speech_Data/motor_training_age.txt";}
		else{
			filename = "Speech_Data/motor_test_age.txt";}
		try {
			   
			  PrintWriter wwr = new PrintWriter(new File(filename));
		    	
		    
				for (int i = 0; i < numMotor; i++) {
					wwr.println("For motor layer"+i);
					int num = motor[i].getNumNeurons();
				    int age = 0;
				    for(int j = 0; j < num; j++){
				    	age = motor[i].motorNeurons[j].getfiringage();
				    	
				    	wwr.println(age);
				    }
						//wwr.println("Time"+time+", the top"+mHiddentopk[i]+" winner neurons' indexs are:"+tempp);
				     
					//wwr.println();
		    }

			wwr.close();
		}catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}	
}
