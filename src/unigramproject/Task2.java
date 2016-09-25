package unigramproject;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

public class Task2 {
	public static void Task2Test(String trainingData, String testData) throws IOException {
		List<String> trainingWords = Unigram.readInputFile(trainingData);
		int trainingSize = trainingWords.size();
		
		List<String> trainingWords_128 = trainingWords.subList(0, trainingSize/128);	
		List<String> testWords = Unigram.readInputFile(testData);
		
		List<String> filenames = new ArrayList<String> ();
		filenames.add(trainingData);
		filenames.add(testData);
		Set<String> vocabulary = Unigram.buildVocabulary(filenames);
		
		HashMap<String, Double> wordFrequency_128 = Unigram.calculateFrequency(trainingWords_128, vocabulary);
		
		for (double alphak = 1.0; alphak < 11.0; alphak++) {
			double evidence = Unigram.evidence(vocabulary, wordFrequency_128, alphak);
			double logEvidence = Unigram.logEvidence(evidence);
			HashMap<String, Double> pdProb_128 = Unigram.calculatePredictiveEst(wordFrequency_128, alphak);
			List<Double> pdresults_128 = Unigram.predictionResults(pdProb_128, testWords);
			double pd_pp_128 = Equations.perplexity(pdresults_128);
			System.out.println("alpha' = " + alphak + " log evidence: " + evidence + " perplexity: " + pd_pp_128);
		}
	}
}
