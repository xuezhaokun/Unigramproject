package unigramproject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.io.*;

/**
 * class to implement Unigram model 
 * @author Zhaokun Xue
 *
 */
public class Unigram {
	
	/**
	 * The function parses an input file to a list of strings
	 * @param filename input file name
	 * @return a list of strings 
	 * @throws IOException input/output exception
	 */
	public static List<String> readInputFile(String filename) throws IOException {
		List<String> words = new ArrayList<String>(); // initial an empty list
		
		// try to open and read the file
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
            String fileRead = br.readLine();

            // parse file content by space and add each token to the list
            while (fileRead != null) {
                String[] tokens = fileRead.split(" ");
                for(String token: tokens) {
                		if (token.length() > 0){
                    		words.add(token);	
                		}
                }
                fileRead = br.readLine();
            }

            br.close();
        } catch (FileNotFoundException fnfe) {
            System.err.println("file not found");
        }
		
		return words;
	}
	
	/**
	 * The function builds vocabulary based on the given files
	 * @param filenames a list of input files' names
	 * @return a set of strings which is the vocabulary
	 * @throws IOException
	 */
	public static Set<String> buildVocabulary(List<String> filenames) throws IOException {
		Set<String> vocabulary = new HashSet<String>(); // initialize a set for vocabulary
		List<String> allwords = new ArrayList<String>(); // initialize an empty list to read all files content
		// read each file and add each word in file to the allwords list
		for (String filename: filenames) {
			List<String> words = Unigram.readInputFile(filename);
			allwords.addAll(words);		
		}
		// build the vocabulary by adding all words we read from files to a set collection
		// set collection only contains unique element
		vocabulary.addAll(allwords);	
	
		return vocabulary;
	}
	
	/**
	 * The function calculates the frequency for each word in the vocabulary
	 * @param trainingData the training data
	 * @param vocabulary the vocabulary used for this training process
	 * @return a hash map has word as key and frequency as value
	 */
	public static HashMap<String, Double> calculateFrequency (List<String> trainingData, Set<String> vocabulary) {
		HashMap<String, Double> wordsFrequency = new HashMap<String, Double>(); // initialize a hashmap with word as key and frequency as value
		// initialized the hashmap by setting the frequency of each word to 0
		for (String word : vocabulary) {
			wordsFrequency.put(word, (double) 0);
		}
		// iterate the training data and update the corresponding frequency in our hashmap
		for (String data : trainingData) {
			if (wordsFrequency.containsKey(data)) {
				double currentCount = wordsFrequency.get(data);
				double newCount = currentCount + 1;
				wordsFrequency.put(data, newCount);
			}
		}
		return wordsFrequency;
	} 

	/**
	 * The function calculates the total number of words based on the word frequency hashmap we built
	 * @param wordFrequency the word frequency hash map we built for input file
	 * @return the total number of words in this frequency hashmap
	 */
	public static double totalNoWords (HashMap<String, Double> wordFrequency) {
	    double totalNoWords = 0;
	    // add up all the frequncies in the hashmap
		for (Map.Entry<String, Double> entry : wordFrequency.entrySet()) {
	        double mk = entry.getValue();
	        totalNoWords += mk; 
	    }
		return totalNoWords;
	}
	
	/**
	 * The function calculates the maximum likelihood for each word
	 * @param wordFrequency the word frequency hash map we built for input file
	 * @return a hashmap with word->key and  ml->value
	 */
	public static HashMap<String, Double> calculateMLEst (HashMap<String, Double> wordFrequency) {
		HashMap<String, Double> mlProb = new HashMap<String, Double>(); // initialize a hashmap for holding the ML probability for each word
		double n = totalNoWords(wordFrequency); // calculate the total number of words in this input data set
		
		// iterate each word in wordfrequency and calculate the ML for each of them. Then put (word, ml) to our ml hashmap
		for (Map.Entry<String, Double> entry : wordFrequency.entrySet()) {
			String word = entry.getKey();
	        double mk = entry.getValue();
	        double mlest = Equations.mlEstimate(mk, n);
	        mlProb.put(word, mlest);
	    }
		return mlProb;
	}
	
	/**
	 * The function calculates the MAP for each word
	 * @param wordFrequency wordFrequency the word frequency hash map we built for input file
	 * @param alphak a scalar for Dirichlet distribution
	 * @return a hashmap with word->key and map->value
	 */
	public static HashMap<String, Double> calculateMAPEst (HashMap<String, Double> wordFrequency,double alphak) {
		HashMap<String, Double> mapProb = new HashMap<String, Double>();// initialize a hashmap for holding the MAP probability for each word
		double n = totalNoWords(wordFrequency); // calculate the total number of words in this input data set
		double k = wordFrequency.size(); // calculate the number of different words in wordfrequency i.e. calculate the size of vocabulary
		double alpha0 = alphak * k; // calculate aplpha0
		
		// iterate each word in wordfrequency and calculate the MAP for each of them. Then put (word, map) to our map hashmap
		for (Map.Entry<String, Double> entry : wordFrequency.entrySet()) {
			String word = entry.getKey();
	        double mk = entry.getValue();
	        double mapest = Equations.mapEstimate(mk, alphak, alpha0, n, k);
	        mapProb.put(word, mapest);
	    }
		return mapProb;
	}

	/**
	 * The function calculates the Predictive Distribution for each word
	 * @param wordFrequency wordFrequency wordFrequency the word frequency hash map we built for input file
	 * @param alphak a scalar for Dirichlet distribution
	 * @return a hashmap with word->key and predictive distribution->value
	 */
	public static HashMap<String, Double> calculatePredictiveEst (HashMap<String, Double> wordFrequency, double alphak) {
		HashMap<String, Double> predictiveProb = new HashMap<String, Double>(); // initialize a hashmap for holding the pd probability for each word
		double n = totalNoWords(wordFrequency); // calculate the total number of words in this input data set
		double k = wordFrequency.size(); // calculate the number of different words in wordfrequency i.e. calculate the size of vocabulary
		double alpha0 = alphak * k; // calculate alpha0
		
		// iterate each word in wordfrequency and calculate the PD for each of them. Then put (word, pd) to our pd hashmap
		for (Map.Entry<String, Double> entry : wordFrequency.entrySet()) {
			String word = entry.getKey();
	        double mk = entry.getValue();
	        double predictiveEst = Equations.predictiveDis(mk, alphak, alpha0, n);
	        predictiveProb.put(word, predictiveEst);
	    }
		return predictiveProb;
	}
	
	/**
	 * The function calculates the prediction results for each word in testing data set
	 * @param probs a hashmap contains probabilities calculated from ML, MAP or Predictive Distribution
	 * @param testData given test data list
	 * @return a list of prediction results for test data
	 */
	public static List<Double> predictionResults (HashMap<String, Double> probs, List<String> testData) {
		List<Double> results = new ArrayList<Double>(); // initialize a list to hold all prediction results
		
		// iterate each word in test data and get its probability, then add to our output list
		for (String data : testData) {
			double prob = probs.get(data);
			results.add(prob);
		}
		return results;
	}
	
	/**
	 * The function calculates the log evidence for given training data
	 * @param vocabulary the vocabulary built from input files
	 * @param wordFrequency wordFrequency wordFrequency wordFrequency the word frequency hash map we built for input file
	 * @param alphak a scalar for Dirichlet distribution
	 * @return the log of the evidence
	 */
	public static double logEvidence(Set<String> vocabulary, HashMap<String, Double> wordFrequency, double alphak) {
		double evidence = 0; // initialize evidence to 0
		double k = vocabulary.size(); // calculates the vocabulary's size
		double log_gamma_alphak_mk_product = 0; // initialize the part for product of gamma(mk + alphak) to 0
		double log_gamma_alphak_product = 0; // initialize the part for product of gamma(mk + alphak) to 0
		double n = totalNoWords(wordFrequency); // calculate the total number of words in this input data set
		double alpha0 = alphak * k; //calculate alpha0
		double log_gamma_alpha0 = Equations.logGamma(alpha0); // calculate the log gamma result for alpha0
		double log_gamma_alpha0_N = Equations.logGamma(alpha0 + n); // calculate the log gamma result for (alpha0 + n)
		double log_gamma_alphak = Equations.logGamma(alphak); // calcualte the log gamma result for alphak
		
		// iterate each word in vocabulary and sum up the log_gamma_alphak_mk_product am log_gamma_alphak_product
		for (String word : vocabulary) {
			double mk = wordFrequency.get(word);
			log_gamma_alphak_mk_product += Equations.logGamma(alphak + mk);
			log_gamma_alphak_product += log_gamma_alphak;
		}
		double numerator = log_gamma_alpha0 + log_gamma_alphak_mk_product;
		double denominator = log_gamma_alpha0_N + log_gamma_alphak_product;
		evidence = numerator - denominator;
		return evidence;
	}
	
	/**
	 * main function for outputting results for each task
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		String dataPath = "data/"; // data path
		String trainingData = dataPath + "training_data.txt";
		String testData = dataPath + "test_data.txt";

		System.out.println("-------- Task 1 ---------");
		Task1.task1Test(trainingData, testData);

		System.out.println("-------- Task 2 ---------");
		Task2.Task2Test(trainingData, testData);
		
		String page84 = dataPath + "pg84.txt.clean";
		String page345 = dataPath + "pg345.txt.clean";
		String page1188 = dataPath + "pg1188.txt.clean";
		System.out.println("-------- Task 3 ---------");
		Task3.Task3Test(page345, page84, page1188);
	}

}
