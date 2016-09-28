package unigramproject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

/**
 * class to implement tests in task2
 * @author Zhaokun Xue
 *
 */
public class Task2 {
	/**
	 * The function implements tests for task 2
	 * @param trainingData training data file
	 * @param testData testing data file
	 * @throws IOException
	 */
	public static void Task2Test(String trainingData, String testData) throws IOException {
		List<String> trainingWords = Unigram.readInputFile(trainingData); // parse training data
		int trainingSize = trainingWords.size(); // calculate training data size
		
		// test based on N/128 training data size
		List<String> trainingWords_128 = trainingWords.subList(0, trainingSize/128);	
		List<String> testWords = Unigram.readInputFile(testData); // parse testing data 
		
		// build vocabulary
		List<String> filenames = new ArrayList<String> ();
		filenames.add(trainingData);
		filenames.add(testData);
		Set<String> vocabulary = Unigram.buildVocabulary(filenames);
		
		// calcualte word frequency in document 
		HashMap<String, Double> wordFrequency_128 = Unigram.calculateFrequency(trainingWords_128, vocabulary);
		
		// compute the log evidence at alpha' from 1 to 10 where alphak = alpha' * 1
		for (double alphak = 1.0; alphak < 11.0; alphak++) {
			double evidence = Unigram.logEvidence(vocabulary, wordFrequency_128, alphak);
			HashMap<String, Double> pdProb_128 = Unigram.calculatePredictiveEst(wordFrequency_128, alphak);
			List<Double> pdresults_128 = Unigram.predictionResults(pdProb_128, testWords);
			double pd_pp_128 = Equations.perplexity(pdresults_128);
			System.out.println("alpha' = " + alphak + " log evidence: " + evidence + " perplexity: " + pd_pp_128);
		}
	}
}
