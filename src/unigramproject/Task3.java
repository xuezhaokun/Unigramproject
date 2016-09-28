package unigramproject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

/**
 * class to implement tests in task 3
 * @author Zhaokun Xue
 *
 */
public class Task3 {
	/**
	 * The function implements tests in task 3
	 * @param trainingData training data file
	 * @param testData1 testing data file1
	 * @param testData2 testing data file2
	 * @throws IOException
	 */
	public static void Task3Test(String trainingData, String testData1, String testData2) throws IOException {
		double alphak = 2; // set alpha'= 2 and alphak = alpha' * 1
		List<String> trainingWords = Unigram.readInputFile(trainingData); // parse training data
		List<String> testWords1 = Unigram.readInputFile(testData1); // parse testing data1
		List<String> testWords2 = Unigram.readInputFile(testData2); // parse testing data2
		
		// build vocabulary
		List<String> filenames = new ArrayList<String> (); 
		filenames.add(trainingData);
		filenames.add(testData1);
		filenames.add(testData2);
		Set<String> vocabulary = Unigram.buildVocabulary(filenames);
		// calculate word frequency
		HashMap<String, Double> wordFrequency = Unigram.calculateFrequency(trainingWords, vocabulary);
		// calculate predictive distribution
		HashMap<String, Double> pdProb = Unigram.calculatePredictiveEst(wordFrequency, alphak);
		// calculate prediction results for each testing file
		List<Double> test1_pdresults = Unigram.predictionResults(pdProb, testWords1);
		List<Double> test2_pdresults = Unigram.predictionResults(pdProb, testWords2);
		// calcualte perplexity for each testing file
		double test1_pd_pp = Equations.perplexity(test1_pdresults);
		double test2_pd_pp = Equations.perplexity(test2_pdresults);
		System.out.println("perplexity for pg84: " + test1_pd_pp + " perplexity for pg1188: " + test2_pd_pp);
	}
}
