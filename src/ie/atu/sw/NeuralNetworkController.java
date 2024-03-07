package ie.atu.sw;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import org.encog.engine.network.activation.ActivationTANH;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.persist.EncogDirectoryPersistence;

public class NeuralNetworkController {
	private BasicNetwork network;
	private static final int INPUT_SIZE = 5;

	public NeuralNetworkController() {
		createNetwork();
	}

	private void createNetwork() {
		network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, INPUT_SIZE)); // Input layer with specified size
		network.addLayer(new BasicLayer(new ActivationTANH(), true, 30)); // Hidden layer with Tanh activation
		network.addLayer(new BasicLayer(new ActivationTANH(), false, 1)); // Output layer - single neuron - Tanh activation
		network.getStructure().finalizeStructure(); 
		network.reset();
	}

	// Trains the neural network with provided input and output data.
	public void trainNetwork(double[][] inputData, double[][] idealOutputData) {
	    // Log training data for inspection.
	    System.out.println("Input Data:");
	    for (double[] inputRow : inputData) {
	        System.out.println(Arrays.toString(inputRow));
	    }
	    System.out.println("Ideal Output Data:");
	    for (double[] outputRow : idealOutputData) {
	        System.out.println(Arrays.toString(outputRow));
	    }
	    
	    // Data set for training.
		MLDataSet trainingSet = new BasicMLDataSet(inputData, idealOutputData);
		
		//Back propagation for training with specified learning rate and momentum.
		Backpropagation trainer = new Backpropagation(network, trainingSet, 0.001, 0.95);
		trainer.setL2(0.010); // L2 regularization to avoid overfitting

		int epoch = 1;
		final int maxEpochs = 10000; 
		final int reportingInterval = 10; 

	
		do {
			trainer.iteration(); // Training iteration (epoch)
			if (epoch % reportingInterval == 0) {
				System.out.println("Epoch #" + epoch + ", Error: " + trainer.getError());
			}
			epoch++;
		} while (!trainer.isTrainingDone() && epoch <= maxEpochs);

		System.out.println("Training completed. Final error: " + trainer.getError());

		// Save the trained model to a file for later use.
		File networkFile = new File("trained_network.eg");
		EncogDirectoryPersistence.saveObject(networkFile, network);
	}

	// Decide an action based on the current game state using the trained network
	public int decide(double[] gameState) {
	    MLData input = new BasicMLData(gameState); // Prepare input data
	    MLData output = network.compute(input); // Compute the output of the network

	    // Convert network output to a decision.
	    double rawOutputValue = output.getData(0);
	    double outputValue = (rawOutputValue + 1) / 2.0; // Normalize output to [0, 1]
	    System.out.println("Normalized Output Activation: " + outputValue);

	    int decision; // Determine action based on output value
	    if (outputValue > 0.33) {
	        decision = -1;
	    } else if (outputValue <= 0.33 && outputValue > 0.66) {
	        decision = 0;
	    } else { 
	        decision = 1;
	    }
	    System.out.println("Decision: " + decision);
	    return decision;
	}

	// Converts a list of double arrays into a 2D double array
	public double[][] convertListToArray(List<double[]> list) {
		return list.toArray(new double[0][]);
	}

	// Converts a list of Integer actions into a 2D double array for output data
	public double[][] convertActionsToOutputArray(List<Integer> actions) {
	    double[][] output = new double[actions.size()][1];
	    for (int i = 0; i < actions.size(); i++) {
	        output[i][0] = actions.get(i); 
	    }
	    return output;
	}

	// Loads a neural network model from a file
	public void loadNetwork(File file) {
		network = (BasicNetwork) EncogDirectoryPersistence.loadObject(file);
	}
}
