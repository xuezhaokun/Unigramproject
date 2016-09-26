package unigramproject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Task3 {
	public static void Task3Test(String trainingData, String testData1, String testData2) throws IOException {
		double alphak = 2;
		List<String> trainingWords = Unigram.readInputFile(trainingData);
		List<String> testWords1 = Unigram.readInputFile(testData1);
		List<String> testWords2 = Unigram.readInputFile(testData2);
		List<String> filenames = new ArrayList<String> ();
		
		filenames.add(trainingData);
		filenames.add(testData1);
		filenames.add(testData2);
		Set<String> vocabulary = Unigram.buildVocabulary(filenames);
		System.out.println("~~~task3 voc size: " + vocabulary.size());
		HashMap<String, Double> wordFrequency = Unigram.calculateFrequency(trainingWords, vocabulary);
		HashMap<String, Double> pdProb = Unigram.calculatePredictiveEst(wordFrequency, alphak);
		
		List<Double> test1_pdresults = Unigram.predictionResults(pdProb, testWords1);
		List<Double> test2_pdresults = Unigram.predictionResults(pdProb, testWords2);
		double test1_pd_pp = Equations.perplexity(test1_pdresults);
		double test2_pd_pp = Equations.perplexity(test2_pdresults);
		System.out.println("pp pg84: " + test1_pd_pp + " pp pg1188: " + test2_pd_pp);
	}
}
