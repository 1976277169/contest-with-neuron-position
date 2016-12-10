import java.util.Arrays;
import java.util.Random;

public class MotorLayer {
	
	private int height;
	private int width;
	
	private float[][] input;
	
	// new variables
	private int topK;
	
	private int numNeurons;
	
	private int numBottomUpWeights;
	
	private final float MACHINE_FLOAT_ZERO = 0.00001f;
	private final float TOP_MOTOR_RESPONSE = 1.0f - MACHINE_FLOAT_ZERO;
	
	private Neuron[] motorNeurons;
	
	public MotorLayer(int height, int width, int topK, int hiddenSize){
		this.setWidth(width);
		this.setHeight(height);
		
		this.setTopK(topK);
		
		input = new float[height][width];
		
		numNeurons = height * width;
		numBottomUpWeights = hiddenSize;
		
		motorNeurons = new Neuron[numNeurons];
		for(int i=0; i<numNeurons; i++){
			motorNeurons[i] = new Neuron(numBottomUpWeights,0,0,i);
		}
	}
	
	public MotorLayer(int width, int height, float[][] input){
		this.setWidth(width);
		this.setHeight(height);
		
		this.setInput(input);
	}
	
	
	public void hebbianLearnMotor(float[] hiddenResponse){
		for (int i = 0; i < numNeurons; i++) {
			//System.out.println("New Response: " + newResponse[i] + " > " + TOP_MOTOR_RESPONSE);
	
			if(motorNeurons[i].getnewresponse() >= TOP_MOTOR_RESPONSE){
				
				motorNeurons[i].hebbianLearnHidden(hiddenResponse);		

			}	
		}
	}
	
	// convert into 1d Array
	// This is 
	public float[] getNewMotorResponse1D() {
		float[] inputArray = new float[numNeurons];
		
		for(int i = 0; i < numNeurons; i++){
			
			inputArray[i] = motorNeurons[i].getnewresponse();
		}
		
		return inputArray;
	}
	
	public float[][] getNewMotorResponse2D() {
		float[][] outputArray = new float[height][width];

		for (int i = 0; i < numNeurons; i++) {
			
			outputArray[i/width][i%width] = motorNeurons[i].getnewresponse();
		}
		
		return outputArray;
	}
	
		
	public void replaceMotorLayerResponse(){
		for (int i = 0; i < numNeurons; i++) {
			
			motorNeurons[i].replaceResponse();
		}
	}
	/*
	private float getLearningRate(int age){
		
		// simple version for learningRate
		return (1.0f / ((float) age));
	}*/
	
	public void computeResponse(float[] hiddenResponse){
		
		// do the dot product between the weights
	 
		for (int i = 0; i < numNeurons; i++) {
				motorNeurons[i].computeBottomUpResponse(hiddenResponse);
				motorNeurons[i].computeResponse();
		
	    }	  
		
		// do the topKcompetition
		topKCompetition();
	}
	
	// Sort the topK elements to the beginning of the sort array where the index of the top 
	// elements are still in the pair.
	private static void topKSort(Pair[] sortArray, int topK){
		
		for (int i = 0; i < topK; i++) {
			Pair maxPair = sortArray[i]; 
			int maxIndex = i;
					
			for (int j = i+1; j < sortArray.length; j++) {
				
				if(sortArray[j].value > maxPair.value){ // select temporary max
					maxPair = sortArray[j];
					maxIndex = j;
					
				}
			}
			
			if(maxPair.index != i){
				Pair temp = sortArray[i]; // store the value of pivot (top i) element
				sortArray[i] = maxPair; // replace with the maxPair object.
				sortArray[maxIndex] = temp; // replace maxPair index elements with the pivot. 
			}
		}
	}

	
	private void topKCompetition(){
		
		// Pair is an object that contains the (index,response_value) of each hidden neurons.
		Pair[] sortArray = new Pair[numNeurons]; 

		for (int i = 0; i < numNeurons; i++) {
			sortArray[i] = new Pair(i, motorNeurons[i].getnewresponse());
			//System.out.println("Motor responses before topK: " + newResponse[i]);
			motorNeurons[i].setnewresponse(0.0f);
			//motorNeurons[i].setwinnerflag(false);
		}
		
		// Sort the array of Pair objects by its response_value in non-increasing order.
		// The index is in the Pair, goes with the response ranked.
		topKSort(sortArray, topK);
		

		//System.out.println("Motor top1 value: " + sortArray[0].value);

		// Find the top1 element and set to one.
		
		// binary conditioning for the topK neurons.
		
		int winnerIndex = 0;
		
		while(winnerIndex < topK){
			
			// get the index of the top element.
			int topIndex = sortArray[winnerIndex].index;		
			motorNeurons[topIndex].setnewresponse(1.0f); 
			//motorNeurons[topIndex].setwinnerflag(true);
			if(topIndex==0){
				motorNeurons[topIndex+1].setnewresponse(0.5f);
			}
			else if(topIndex==(numNeurons-1)){
				motorNeurons[topIndex-1].setnewresponse(0.5f);
			}
			else{
				motorNeurons[topIndex-1].setnewresponse(0.5f);
				motorNeurons[topIndex+1].setnewresponse(0.5f);
			
			}
			
			winnerIndex++;
			
		}
		
		
	}

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	public int getWidth() {
		return width;
	}

	public void setWidth(int width) {
		this.width = width;
	}

	// convert into 1d Array
	public float[] getInput1D() {
		float[] inputArray = new float[height * width];
		
		for (int i = 0; i < height; i++) {
			System.arraycopy(input[i], 0, inputArray, i * width, width);
			
		}
		
		return inputArray;
	}
	
	public float[][] getInput() {
		return input;
	}

	public void setInput(float[][] input) {
		this.input = input;
	}

	public int getTopK() {
		return topK;
	}

	public void setTopK(int topK) {
		this.topK = topK;
	}

	public int getNumBottomUpWeights() {
		return numBottomUpWeights;
	}

	public void setNumBottomUpWeights(int numBottomUpWeights) {
		this.numBottomUpWeights = numBottomUpWeights;
	}

	public void setSupervisedResponse(float[][] supervisedResponse){
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int index = i*width + j;
			
				motorNeurons[index].setnewresponse(supervisedResponse[i][j]);
			}
		}
		
	}
	
	public class Pair implements Comparable<Pair> {
	    public final int index;
	    public final float value;

	    public Pair(int index, float value) {
	        this.index = index;
	        this.value = value;
	    }

		public int compareTo(Pair other) {
			return -1*Float.valueOf(this.value).compareTo(other.value);
		}
		
		public int get_index(){
			return index;
		}
	}
}
