
import java.io.*;
import java.util.*;


/*
 * AIML Contest Interface
 * 
 * For simplicity, we decided to use a Main class for evaluating the participant's contest submission.
 * The participant's DN submission must have the same initialization procedures discussed during the workshop.
 * The participant can modify the function calls used by their DN implementation, 
 * but the function must work similarly to our discussed functions during the workshops.
 * The input, motor, and performance readers' filenames cannot be modified. The compressed zip files will contain the filename for the corresponding modality.
 * 
 * Also, the participant is allowed to add display commands to debug their program.
 * 
 * The readers and settings objects are hidden from the participant. However, they are allowed to modify the number of hidden neurons. 
 * The participant can only use less neurons than the number provided in the setting files.
 * The max number of neurons is dependent on the problem.
 * 
 */
public class Contest_Interface {

	public static void main(String[] args) throws IOException {
		
		// Stores all the intialization information for the network
		// Indicate the filename to initialize the settings.
		
		Scanner sc = new Scanner(System.in);
		
		int modalityOption = 0;
		
		String modalityDirectory = "";
		String modalityName = "";
		
		do{
		
			try{
				System.out.println("Select which modality you would like to run.");
				System.out.println("1. Vision Modality");
				System.out.println("2. Language Modality");
				System.out.println("3. Audition Modality");
				System.out.print("Enter an integer between 1-3: ");
				modalityOption = sc.nextInt();
				System.out.println();
				
			}
			catch(Exception e){
				System.out.println("Option must be an integer number");
			}
			
		}while(modalityOption < 1 || modalityOption > 3);
		
		Settings st = null;
		
		switch(modalityOption){
			case 1:
				modalityDirectory = "Image_Data";
				modalityName = "vision";
				break;
				
			case 2:
				modalityDirectory = "Language_Data";
				modalityName = "language";
				break;
				
			case 3:
				modalityDirectory = "Speech_Data";
				modalityName = "audition";
				break;
		}

		st = new Settings(modalityDirectory + "/settings_" + modalityName + ".txt");
		
		/*
		 * Get the information from the settings file.
		 */
		int numSegments = st.getNumSegments();
		int startTestSegment = st.getTestSegmentStart(); // this index indicates when training ends and testing begins in terms of Segments.
		int[] lenSequence = st.getLenSequence(); // total elements within the sequence.
		
		
		int numInput = st.getNumInput(); // The number of sensor the network will use. For this contest, we will use one sensor. 
		int[][] inputSize = st.getInputSize(); // This array will have the (height, width) of each sensor.
				
		int numMotor = st.getNumMotor(); // The number of motors or effectors the DN will have. 
		int[][] motorSize = st.getMotorSize(); // (height, width) of each motor.
		int[] topKMotor = st.getTopKMotor(); // Topk winning neurons of each motor layer.
		
		int numHidden = st.getNumHidden(); // Number of hidden layers.
		int[] numHiddenNeurons = st.getHiddenSize(); // The number of neurons for each hidden layer. This value can be changed, only to a smaller value.
		int[] topKHidden = st.getTopKHidden(); // Topk winning neurons for each hidden layer.
		
		/*
		 * The modality of the input.
		 * V -> image data
		 * A -> audition data
		 * T -> language (text) data
		 */
		char modality = st.getModality();
		
		// The network is initialized the same way is was discussed during the workshop.
		DN network = new DN(numInput, inputSize, numMotor, motorSize, topKMotor, numHidden, numHiddenNeurons, topKHidden);
		
		// Initialize the input and motor streams.
		InputDataReader inputReader = null; 
		MotorDataReader motorReader = null;
		MotorDataReader performanceReader = null;
		
		
		/*
		 * Measures the network performance.
		 * 
		 * Computing the error rate is hidden from the participant.
		 * This object will display the computing error for debugging purposes.
		 */
		PerformanceMeasure performance = new PerformanceMeasure(numSegments, numMotor, lenSequence, startTestSegment, numHidden, numHiddenNeurons, modality);
				
		boolean[] supervisedMotors = new boolean[numMotor]; // determine whether the array needs supervision or not.
		
		float[][][] oldInputPattern = new float[numInput][][]; // The input pattern used to compute the hidden response.
		float[][][] oldMotorPattern = new float[numMotor][][]; // The motor pattern used to compute the hidden response.
		float[][][] currentMotorPattern = new float[numMotor][][]; // The motor pattern at the current timestep to determine if supervision is needed.
		float[][][] newMotorPattern = new float[numMotor][][]; // The motor pattern computed by the network or supervised by the data.
		float[][][] performanceMotorPattern = new float[numMotor][][]; // The expected motor pattern used to compare with the computed pattern.
		
		/* 
		 * Start the Segment iteration
		 * This case we run two Segments. One for training and other for testing.
		 */
		for (int k = 0; k < numSegments; k++) {
			
			// Keep track of the current input/motor sequence.
			int seqCount = 0;
			
			
			/*
			 * We will have three Segments
			 * k = 0 -> Training Segment
			 * k = 1 -> Resubstitution Test Segment. Only supervise the first two time steps. 
			 * k = 2 -> Disjoint Test Segment. This Segment will test whether the network is able to generalize using attention with a new dataset.
			 */
			inputReader = getInputDataReader(k, lenSequence[k], modality);
			motorReader = getMotorDataReader(k, lenSequence[k], modality);
			
			// In the future, maybe training including testing.
			performanceReader = getPerformanceMotorDataReader(k, lenSequence[k], modality);
		
			System.out.println("Segment " + k);
			
			// t = 1 (initialization of inputs and motors).
			//read the old input
			System.out.println("t = 1");
			
			oldInputPattern = inputReader.getStreamInput();
			oldMotorPattern = motorReader.getStreamInput();
			performanceReader.getStreamInput();
			
			
			seqCount++;
			
			// t = 2 (compute the first hidden response).
			System.out.println("t = 2");
			
			//compute the hidden response with the old input
			network.computeHiddenResponse(oldInputPattern, oldMotorPattern);
			
			//get top-k winner neurons' indexs
			if(k<2){
			    network.trackHiddenwinners(seqCount);
			    }
			
			network.replaceHiddenResponse();
			
			oldInputPattern = inputReader.getStreamInput();
			oldMotorPattern = motorReader.getStreamInput();
			performanceReader.getStreamInput();
			
			seqCount++;
			
			// start at t = 3
			do{
		
				System.out.println("t = " + (seqCount+1));
				
				//compute the hidden response with the old input
				// The order doesn't matter, we choose to compute the Y area first.
				
				network.computeHiddenResponse(oldInputPattern, oldMotorPattern);
				
				//get top-k neurons' indexs
				if(k < 2){
				      network.trackHiddenwinners(seqCount);
				}
				
				// Read the current motor pattern to determine if supervision is required.
				currentMotorPattern = motorReader.getStreamInput();
				performanceMotorPattern = performanceReader.getStreamInput();
				
				//using the old response from Y compute the new motor
				// all zeros will represent that the pattern is to be emergent.
				supervisedMotors = allZeros(currentMotorPattern);				

				// compute the response for each motor individually.
				for (int i = 0; i < numMotor; i++) {
					
					// If the current motor pattern has at least one non-zero element.
					// we compute the motor response and update its weights without supervision.
					if(supervisedMotors[i]){
						newMotorPattern[i] = network.computeMotorResponse(i);
						network.updateMotorWeights(i);
						
						//Indicate that this instance is to count on final errorComputation
						performance.updateInstanceCount(k,i);
						
						// compute the error per sequence.
						performance.updateErrorTimeRate(k, i, seqCount, newMotorPattern, performanceMotorPattern);
						
						System.out.println("Supervision is not required!");
					}
					
					else{ // new state is supervised to be the current
						
						newMotorPattern[i] = currentMotorPattern[i];
						network.updateSupervisedMotorWeights(i,newMotorPattern[i]);
						System.out.println("Supervision is required!");
					}
					

				}
				
				System.out.println("Old Motor Pattern");
				displayResponse(oldMotorPattern);
				
				System.out.println("Current Motor Pattern");
				displayResponse(currentMotorPattern);
				
				System.out.println("New Motor Pattern");
				displayResponse(newMotorPattern);
				
				// set the current input as the old input for next computation
				oldInputPattern = inputReader.getStreamInput();
				oldMotorPattern = newMotorPattern;
				
				
				
				// Increment the sequence index counter.
				if(motorReader.hasMotor())
					seqCount++;
				
				// replace old by new.
				// see DN book algorithm 6.1, Step 2(b)
				network.replaceHiddenResponse(); // All computation was using oldResponses for individual neurons, now new response replace the old. 
				network.replaceMotorResponse();  // All computation was using oldResponses for individual neurons, now new response replace the old.
			// get the neuron's location
				if((seqCount%20 == 0)&&(k ==0)){
					network.updateHiddenLocation();	
				}
				
			}while(inputReader.hasInput() && motorReader.hasMotor());
			
			//measure the error per concept
			performance.updateErrorMotorRate(k);
			
			// measure the error per Segment
			performance.updateErrorSegmentRate(k);
			
			if(k==1){
				network.trackHiddenweights();
				network.trackHiddenweights2();
				

			}
			

		}
		
		// We divide the total error per Segment by  the number of Segments.
		performance.computeErrorRate();
		network.outputHiddenLocation();		

		//Write results to file
		/*
		switch(modality){
			case 'V':
				performance.writePerformance(modalityDirectory + "/performance_vision.txt");
				break;
				
			case 'A':
				performance.writePerformance(modalityDirectory + "/performance_audition.txt");
				break;
				
			case 'T':
				performance.writePerformance(modalityDirectory + "/performance_language.txt");
				
			default:
				break;
		}
		*/
		
		performance.writePerformance(modalityDirectory + "/performance_" + modalityName + ".txt");
		
		
				
	}
	
	/*
	 * Initialize the InputDataReader with the corresponding dataset
	 * Segment + 1 = 1 -> Training dataset. 
	 * Segment + 1 = 2 -> Resubstitution Test dataset. Same sequence as training dataset.
	 * Segment + 1 = 3 -> Disjoint Test dataset. New sequence.
	 */
	private static InputDataReader getInputDataReader(int Segment, int lenSequence, char modality){
		
		InputDataReader inRead = null;
		
		switch(Segment){
			case 0:
				switch(modality){
				
				case 'V':
					inRead = new InputDataReader("Image_Data/Input/vision_training_input.mat", lenSequence);
					break;
					
				case 'A':
					inRead = new InputDataReader("Speech_Data/Input/audition_training_input.mat", lenSequence);
					break;
					
				case 'T':
					inRead = new InputDataReader("Language_Data/Input/language_training_input.mat", lenSequence);
					break;
					
				default:
					inRead = null;
					break;
			}
			break;
			
			case 1:
				switch(modality){
				
				case 'V':
					inRead = new InputDataReader("Image_Data/Input/vision_resubstitution_input.mat", lenSequence);
					break;
					
				case 'A':
					inRead = new InputDataReader("Speech_Data/Input/audition_resubstitution_input.mat", lenSequence);
					break;
					
				case 'T':
					inRead = new InputDataReader("Language_Data/Input/language_resubstitution_input.mat", lenSequence);
					break;
					
				default:
					inRead = null;
					break;
			}
			break;
			
			case 2:
				switch(modality){
				
				case 'V':
					inRead = new InputDataReader("Image_Data/Input/vision_disjoint_1_input.mat", lenSequence);
					break;
					
				case 'A':
					inRead = new InputDataReader("Speech_Data/Input/audition_disjoint_1_input.mat", lenSequence);
					break;
					
				case 'T':
					inRead = new InputDataReader("Language_Data/Input/language_disjoint_1_input.mat", lenSequence);
					break;
					
				default:
					inRead = null;
					break;
			}
			break;
			
			case 3:
				switch(modality){
				
				case 'V':
					inRead = new InputDataReader("Image_Data/Input/vision_disjoint_2_input.mat", lenSequence);
					break;
					
				case 'A':
					inRead = new InputDataReader("Speech_Data/Input/audition_disjoint_2_input.mat", lenSequence);
					break;
					
				case 'T':
					inRead = new InputDataReader("Language_Data/Input/language_disjoint_2_input.mat", lenSequence);
					break;
					
				default:
					inRead = null;
					break;
			}
			break;
			
			case 4:
				switch(modality){
				
				case 'V':
					inRead = new InputDataReader("Image_Data/Input/vision_disjoint_3_input.mat", lenSequence);
					break;
					
				case 'A':
					inRead = new InputDataReader("Speech_Data/Input/audition_disjoint_3_input.mat", lenSequence);
					break;
					
				case 'T':
					inRead = new InputDataReader("Language_Data/Input/language_disjoint_3_input.mat", lenSequence);
					break;
					
				default:
					inRead = null;
					break;
			}
			break;
		}
		
		return inRead;
	}
	

	/*
	 * Initialize the MotorDataReader with the corresponding dataset for the corresponding modality.
	 * Segment + 1 = 1 -> Training dataset. All motors are supervised so the network can learn all sequences.
	 * Segment + 1 = 2 -> Resubstitution Test dataset. Same sequence as the training dataset. The first two motors are supervised, the other motors are free. 
	 * Segment + 1 = 3 -> Disjoint Test dataset. New sequence, not previously trained. The first two motors are supervised, the other motors are free. 
	 */
	private static MotorDataReader getMotorDataReader(int Segment, int lenSequence, char modality){
		
		MotorDataReader moRead = null;
		
		switch(Segment){
			case 0:
				switch(modality){
				
				case 'V':
					moRead = new MotorDataReader("Image_Data/Input/vision_training_motor.mat", lenSequence);
					break;
					
				case 'A':
					moRead = new MotorDataReader("Speech_Data/Input/audition_training_motor.mat", lenSequence);
					break;
					
				case 'T':
					moRead = new MotorDataReader("Language_Data/Input/language_training_motor.mat", lenSequence);
					break;
					
				default:
					moRead = null;
					break;
			}
			break;
			
			case 1:
				switch(modality){
				
				case 'V':
					moRead = new MotorDataReader("Image_Data/Input/vision_resubstitution_motor.mat", lenSequence);
					break;
					
				case 'A':
					moRead = new MotorDataReader("Speech_Data/Input/audition_resubstitution_motor.mat", lenSequence);
					break;
					
				case 'T':
					moRead = new MotorDataReader("Language_Data/Input/language_resubstitution_motor.mat", lenSequence);
					break;
					
				default:
					moRead = null;
					break;
			}
			break;
			
			case 2:
				switch(modality){
				
				case 'V':
					moRead = new MotorDataReader("Image_Data/Input/vision_disjoint_1_motor.mat", lenSequence);
					break;
					
				case 'A':
					moRead = new MotorDataReader("Speech_Data/Input/audition_disjoint_1_motor.mat", lenSequence);
					break;
					
				case 'T':
					moRead = new MotorDataReader("Language_Data/Input/language_disjoint_1_motor.mat", lenSequence);
					break;
					
				default:
					moRead = null;
					break;
			}
			break;
			
			case 3:
				switch(modality){
				
				case 'V':
					moRead = new MotorDataReader("Image_Data/Input/vision_disjoint_2_motor.mat", lenSequence);
					break;
					
				case 'A':
					moRead = new MotorDataReader("Speech_Data/Input/audition_disjoint_2_motor.mat", lenSequence);
					break;
					
				case 'T':
					moRead = new MotorDataReader("Language_Data/Input/language_disjoint_2_motor.mat", lenSequence);
					break;
					
				default:
					moRead = null;
					break;
			}
			break;
			
			case 4:
				switch(modality){
				
				case 'V':
					moRead = new MotorDataReader("Image_Data/Input/vision_disjoint_3_motor.mat", lenSequence);
					break;
					
				case 'A':
					moRead = new MotorDataReader("Speech_Data/Input/audition_disjoint_3_motor.mat", lenSequence);
					break;
					
				case 'T':
					moRead = new MotorDataReader("Language_Data/Input/language_disjoint_3_motor.mat", lenSequence);
					break;
					
				default:
					moRead = null;
					break;
			}
			break;
		}
		
		return moRead;	
	}
	
	/*
	 * Initialize the MotorDataReader with the corresponding dataset for the corresponding modality.
	 * Segment + 1 = 1 -> Training dataset. All motors are supervised so the network can learn all sequences.
	 * Segment + 1 = 2 -> Resubstitution Test dataset. Same sequence as the training dataset. The first two motors are supervised, the other motors are free. 
	 * Segment + 1 = 3 -> Disjoint Test dataset. New sequence, not previously trained. The first two motors are supervised, the other motors are free. 
	 */
	private static MotorDataReader getPerformanceMotorDataReader(int Segment, int lenSequence, char modality){
		
		MotorDataReader moRead = null;
		
		switch(Segment){
			case 0:
				switch(modality){
				
				case 'V':
					moRead = new MotorDataReader("Image_Data/Input/vision_training_performance_motor.mat", lenSequence);
					break;
					
				case 'A':
					moRead = new MotorDataReader("Speech_Data/Input/audition_training_performance_motor.mat", lenSequence);
					break;
					
				case 'T':
					moRead = new MotorDataReader("Language_Data/Input/language_training_performance_motor.mat", lenSequence);
					break;
					
				default:
					moRead = null;
					break;
			}
			break;
			
			case 1:
				switch(modality){
				
				case 'V':
					moRead = new MotorDataReader("Image_Data/Input/vision_resubstitution_performance_motor.mat", lenSequence);
					break;
					
				case 'A':
					moRead = new MotorDataReader("Speech_Data/Input/audition_resubstitution_performance_motor.mat", lenSequence);
					break;
					
				case 'T':
					moRead = new MotorDataReader("Language_Data/Input/language_resubstitution_performance_motor.mat", lenSequence);
					break;
					
				default:
					moRead = null;
					break;
			}
			break;
			
			case 2:
				switch(modality){
				
				case 'V':
					moRead = new MotorDataReader("Image_Data/Input/vision_disjoint_1_performance_motor.mat", lenSequence);
					break;
					
				case 'A':
					moRead = new MotorDataReader("Speech_Data/Input/audition_disjoint_1_performance_motor.mat", lenSequence);
					break;
					
				case 'T':
					moRead = new MotorDataReader("Language_Data/Input/language_disjoint_1_performance_motor.mat", lenSequence);
					break;
					
				default:
					moRead = null;
					break;
			}
			break;
			
			case 3:
				switch(modality){
				
				case 'V':
					moRead = new MotorDataReader("Image_Data/Input/vision_disjoint_2_performance_motor.mat", lenSequence);
					break;
					
				case 'A':
					moRead = new MotorDataReader("Speech_Data/Input/audition_disjoint_2_performance_motor.mat", lenSequence);
					break;
					
				case 'T':
					moRead = new MotorDataReader("Language_Data/Input/language_disjoint_2_performance_motor.mat", lenSequence);
					break;
					
				default:
					moRead = null;
					break;
			}
			break;
			
			case 4:
				switch(modality){
				
				case 'V':
					moRead = new MotorDataReader("Image_Data/Input/vision_disjoint_3_performance_motor.mat", lenSequence);
					break;
					
				case 'A':
					moRead = new MotorDataReader("Speech_Data/Input/audition_disjoint_3_performance_motor.mat", lenSequence);
					break;
					
				case 'T':
					moRead = new MotorDataReader("Language_Data/Input/language_disjoint_3_performance_motor.mat", lenSequence);
					break;
					
				default:
					moRead = null;
					break;
			}
			break;
		}
		
		return moRead;	
	}
	
		
	/*
	 * This methods displays a 3d array of response on console.
	 * 
	 * You can use it to see the motor responses for debugging purposes.
	 */
	private static void displayResponse(float[][][] r) {
		
		for (int k = 0; k < r.length; k++) {
			for (int i = 0; i < r[k].length; i++) {
				System.out.print("Motor " + (k+1) + " " + r[k][i][0]);
				for (int j = 1; j < r[k][i].length; j++) {
					System.out.print("," + r[k][i][j]);
				}
				System.out.println();
			}
		}	
	}


	/*
	 *  This procedure checks if all the elements of a motor pattern are zero.
	 *  If all the elements are zero, then supervision is not required.
	 */
	private static boolean[] allZeros(float[][][] r) {
		
		 
		boolean[] sup = new boolean[r.length]; // indicates whether the motor needs to be supervised.
		
		for (int i = 0; i < r.length; i++) { // motor loop
			
			int count = 0;
			
			for (int j = 0; j < r[i].length; j++) { // height of each motor loop
				
				for (int k = 0; k < r[i][j].length; k++) { // width of each motor loop
					// if there is at least one motor neuron active, increase the count.
					if( ((int) r[i][j][k]) == 1){
						count++;
					}
				}
				
			}

			// If the number of active motor neurons is greater than zero, supervise the motor response.
			if(count == 0)
				sup[i] = true;
	
		}
		
		return sup;
	}

}
