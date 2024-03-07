package ie.atu.sw;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.util.concurrent.ThreadLocalRandom.current;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import javax.swing.JPanel;
import javax.swing.Timer;

public class GameView extends JPanel implements ActionListener{
	
	private NeuralNetworkController neuralController;
	
	//Some constants
	private static final long serialVersionUID	= 1L;
	private static final int MODEL_WIDTH 		= 30;
	private static final int MODEL_HEIGHT 		= 20;
	private static final int SCALING_FACTOR 	= 30;
	
	private static final int MIN_TOP 			= 2;
	private static final int MIN_BOTTOM 		= 18;
	private static final int PLAYER_COLUMN 		= 15;
	private static final int TIMER_INTERVAL 	= 100;
	
	private static final byte ONE_SET 			=  1;
	private static final byte ZERO_SET 			=  0;

	/*
	 * The 30x20 game grid is implemented using a linked list of 
	 * 30 elements, where each element contains a byte[] of size 20. 
	 */
	private LinkedList<byte[]> model = new LinkedList<>();

	//These two variables are used by the cavern generator. 
	private int prevTop = MIN_TOP;
	private int prevBot = MIN_BOTTOM;
	
	//Once the timer stops, the game is over
	private Timer timer;
	private long time;
	
	private int playerRow = 11;
	private int index = MODEL_WIDTH - 1; //Start generating at the end
	private Dimension dim;
	
	//Some fonts for the UI display
	private Font font = new Font ("Dialog", Font.BOLD, 50);
	private Font over = new Font ("Dialog", Font.BOLD, 100);

	//The player and a sprite for an exploding plane
	private Sprite sprite;
	private Sprite dyingSprite;
	
	private boolean auto;
	
	private boolean collectingData = false;
	private List<double[]> trainingData = new ArrayList<>();
	private List<Integer> trainingActions = new ArrayList<>();
	
    private int lastAction = 0; // Track the last action (0 for stay, -1 for up, 1 for down)


	public GameView(boolean auto) throws Exception{
		this.auto = auto; //Use the autopilot
        this.neuralController = new NeuralNetworkController();
        
        
     // Initialize the neural network controller only if autopilot mode is enabled
        if (auto) {
            this.neuralController = new NeuralNetworkController();

            // Load the network
            File networkFile = new File("./resources/trained_network.eg");
            neuralController.loadNetwork(networkFile);
        }
        
		setBackground(Color.LIGHT_GRAY);
		setDoubleBuffered(true);
		
		//Creates a viewing area of 900 x 600 pixels
		dim = new Dimension(MODEL_WIDTH * SCALING_FACTOR, MODEL_HEIGHT * SCALING_FACTOR);
    	super.setPreferredSize(dim);
    	super.setMinimumSize(dim);
    	super.setMaximumSize(dim);
		
    	initModel();
    	
		timer = new Timer(TIMER_INTERVAL, this); //Timer calls actionPerformed() every second
		timer.start();
	}
	
	//Build our game grid
	private void initModel() {
		for (int i = 0; i < MODEL_WIDTH; i++) {
			model.add(new byte[MODEL_HEIGHT]);
		}
	}
	
	public void setSprite(Sprite s) {
		this.sprite = s;
	}
	
	public void setDyingSprite(Sprite s) {
		this.dyingSprite = s;
	}
	
	//Called every second by actionPerformed(). Paint methods are usually ugly.
	public void paintComponent(Graphics g) {
        super.paintComponent(g);
        var g2 = (Graphics2D)g;
        
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, dim.width, dim.height);
        
        int x1 = 0, y1 = 0;
        for (int x = 0; x < MODEL_WIDTH; x++) {
        	for (int y = 0; y < MODEL_HEIGHT; y++){  
    			x1 = x * SCALING_FACTOR;
        		y1 = y * SCALING_FACTOR;

        		if (model.get(x)[y] != 0) {
            		if (y == playerRow && x == PLAYER_COLUMN) {
            			timer.stop(); //Crash...
            		}
            		g2.setColor(Color.BLACK);
            		g2.fillRect(x1, y1, SCALING_FACTOR, SCALING_FACTOR);
        		}
        		
        		if (x == PLAYER_COLUMN && y == playerRow) {
        			if (timer.isRunning()) {
            			g2.drawImage(sprite.getNext(), x1, y1, null);
        			}else {
            			g2.drawImage(dyingSprite.getNext(), x1, y1, null);
        			}
        			
        		}
        	}
        }
        
        /*
         * Not pretty, but good enough for this project... The compiler will
         * tidy up and optimise all of the arithmetics with constants below.
         */
        g2.setFont(font);
        g2.setColor(Color.RED);
        g2.fillRect(1 * SCALING_FACTOR, 15 * SCALING_FACTOR, 400, 3 * SCALING_FACTOR);
        g2.setColor(Color.WHITE);
        g2.drawString("Time: " + (int)(time * (TIMER_INTERVAL/1000.0d)) + "s", 1 * SCALING_FACTOR + 10, (15 * SCALING_FACTOR) + (2 * SCALING_FACTOR));
        
        if (!timer.isRunning()) {
			g2.setFont(over);
			g2.setColor(Color.RED);
			g2.drawString("Game Over!", MODEL_WIDTH / 5 * SCALING_FACTOR, MODEL_HEIGHT / 2* SCALING_FACTOR);
        }
	}

	//Move the plane up or down
	public void move(int step) {
		// The program saves the current action here.
		lastAction = step;
		// If it's in the process of collecting data
		if (collectingData) {
		    // It captures the current game state.
		    double[] gameState = sample();
		    // And stores that state for future training.
		    trainingData.add(gameState);
		    // It also records the action taken.
		    trainingActions.add(step);
		}
		// The player's row position is updated based on the action, ensuring it remains within the game's boundaries.
		playerRow = Math.max(0, Math.min(MODEL_HEIGHT - 1, playerRow + step));

	    
	}

	
	/*
	 * ----------
	 * AUTOPILOT!
	 * ----------
	 * The following implementation randomly picks a -1, 0, 1 to control the plane. You 
	 * should plug the trained neural network in here. This method is called by the timer
	 * every TIMER_INTERVAL units of time from actionPerformed(). There are other ways of
	 * wiring your neural network into the application, but this way might be the easiest. 
	 *  
	 */
	private void autoMove() {
		// Auto-mode check
		if (auto) {
		    // Sample and print current game state
		    double[] gameState = sample();
		    for (double value : gameState) {
		        System.out.print(value + " ");
		    }            
		    // Decide next move and execute.
		    int decision = neuralController.decide(gameState);
		    move(decision);
		    // System.out.println(decision);
		}

	}
	
	public void toggleDataCollection() {
	    collectingData = !collectingData;
	    
	    if (!collectingData) {
	        // Check if there's data collected for training
	        if (!trainingData.isEmpty() && !trainingActions.isEmpty()) {
	            // Convert collected data for training.
	            double[][] inputData = neuralController.convertListToArray(trainingData);
	            double[][] idealOutputData = neuralController.convertActionsToOutputArray(trainingActions);

	            // Train the network with the prepared data
	            neuralController.trainNetwork(inputData, idealOutputData);

	            // Clear the collections after training
	            trainingData.clear();
	            trainingActions.clear();
	        } else {
	            System.out.println("No data collected for training.");
	        }
	    } else {
	        System.out.println("Data collection started. Press 'S' again to stop and train.");
	    }
	}

	
	
	//Called every second by the timer 
	public void actionPerformed(ActionEvent e) {
		time++; //Update our timer
		this.repaint(); //Repaint the cavern
		
		//Update the next index to generate
		index++;
		index = (index == MODEL_WIDTH) ? 0 : index;
		
		generateNext(); //Generate the next part of the cave
		if (auto) autoMove();
		
		/*
		 * Use something like the following to extract training data.
		 * It might be a good idea to submit the double[] returned by
		 * the sample() method to an executor and then write it out 
		 * to file. You'll need to label the data too and perhaps add
		 * some more features... Finally, you do not have to sample 
		 * the data every TIMER_INTERVAL units of time. Use some modular
		 * arithmetic as shown below. Alternatively, add a key stroke 
		 * to fire an event that starts the sampling.
		 */
		
		// Check if the program is currently collecting data
		if (collectingData) {
		    // Sample the current game state
		    double[] gameState = sample();
		    // Add the sampled state to the training data set
		    trainingData.add(gameState);
		    // Record the last action taken in the training actions set
		    trainingActions.add(lastAction); 
		}
        
        lastAction = 0; // Reset action to "stay" for next tick unless changed
	}
	
	
	/*
	 * Generate the next layer of the cavern. Use the linked list to
	 * move the current head element to the tail and then randomly
	 * decide whether to increase or decrease the cavern. 
	 */
	private void generateNext() {
		var next = model.pollFirst(); 
		model.addLast(next); //Move the head to the tail
		Arrays.fill(next, ONE_SET); //Fill everything in
		
		
		//Flip a coin to determine if we could grow or shrink the cave
		var minspace = 4; //Smaller values will create a cave with smaller spaces
		prevTop += current().nextBoolean() ? 1 : -1; 
		prevBot += current().nextBoolean() ? 1 : -1;
		prevTop = max(MIN_TOP, min(prevTop, prevBot - minspace)); 		
		prevBot = min(MIN_BOTTOM, max(prevBot, prevTop + minspace));

		//Fill in the array with the carved area
		Arrays.fill(next, prevTop, prevBot, ZERO_SET);
	}
	
	
	/*
	 * Use this method to get a snapshot of the 30x20 matrix of values
	 * that make up the game grid. The grid is flatmapped into a single
	 * dimension double array... (somewhat) ready to be used by a neural 
	 * net. You can experiment around with how much of this you actually
	 * will need. The plane is always somehere in column PLAYER_COLUMN
	 * and you probably do not need any of the columns behind this. You
	 * can consider all of the columns ahead of PLAYER_COLUMN as your
	 * horizon and this value can be reduced to save space and time if
	 * needed, e.g. just look 1, 2 or 3 columns ahead. 
	 * 
	 * You may also want to track the last player movement, i.e.
	 * up, down or no change. Depending on how you design your neural
	 * network, you may also want to label the data as either okay or 
	 * dead. Alternatively, the label might be the movement (up, down
	 * or straight). 
	 *  
	 */
	public double[] sample() {
	    double[] vector = new double[5]; // Vector with 5 elements
	   
	    byte[] playerColumn = model.get(PLAYER_COLUMN); // Get the game column where the player is currently located
	    double immediateTop = 0; 
	    double immediateBottom = 0; 
	    
	    // Calculate distance to the nearest obstacle above the player in the current column
	    for (int i = playerRow; i >= 0; i--) {
	        if (playerColumn[i] == ONE_SET) break; // Stop if an obstacle is encountered
	        immediateTop++;
	    }
	    // Calculate distance to the nearest obstacle below the player in the current column
	    for (int i = playerRow; i < MODEL_HEIGHT; i++) {
	        if (playerColumn[i] == ONE_SET) break; // Stop if an obstacle is encountered
	        immediateBottom++;
	    }
	    
	    // Calculate averaged distances to obstacles in columns ahead for a broader view
	    double avgTop = 0; 
	    double avgBottom = 0; 
	    int count = 0;
	    for (int j = 1; j <= 2; j++) { // Iterate through the next two columns
	        byte[] column = model.get(Math.min(PLAYER_COLUMN + j, MODEL_WIDTH - 1));
	        double top = 0;
	        double bottom = 0;
	        // Calculate distances in the lookahead column.
	        for (int i = 0; i < MODEL_HEIGHT; i++) {
	            if (column[i] == ONE_SET) break;
	            top++;
	        }
	        for (int i = MODEL_HEIGHT - 1; i >= 0; i--) {
	            if (column[i] == ONE_SET) break;
	            bottom++;
	        }
	        avgTop += top; 
	        avgBottom += bottom; 
	        count++;
	    }
	    
	    // Calculate the averages for distances to obstacles.
	    avgTop /= count; 
	    avgBottom /= count; 

	    // Vectors with normalized distances and player's position.
	    vector[0] = immediateTop / MODEL_HEIGHT; 
	    vector[1] = immediateBottom / MODEL_HEIGHT; 
	    vector[2] = avgTop / MODEL_HEIGHT; 
	    vector[3] = avgBottom / MODEL_HEIGHT; 
	    vector[4] = (double) playerRow / MODEL_HEIGHT; 
	    
	    return vector; 
	}

	
	/*
	 * Resets and restarts the game when the "S" key is pressed
	 */
	public void reset() {
		model.stream() 		//Zero out the grid
		     .forEach(n -> Arrays.fill(n, 0, n.length, ZERO_SET));
		playerRow = 11;		//Centre the plane
		time = 0; 			//Reset the clock
		timer.restart();	//Start the animation
	}
}