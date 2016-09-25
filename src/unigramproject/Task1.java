package unigramproject;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

public class Task1 {
	public static void task1Tets(String trainingData, String testData) throws IOException {
		List<String> trainingWords = Unigram.readInputFile(trainingData);
		int trainingSize = trainingWords.size();
		//int[] trainingSizes = new int[]{trainingSize/128, trainingSize/64, trainingSize/16, trainingSize/4, trainingSize};
		List<String> trainingWords_128 = trainingWords.subList(0, trainingSize/128);
		List<String> trainingWords_64 = trainingWords.subList(0, trainingSize/64);
		List<String> trainingWords_16 = trainingWords.subList(0, trainingSize/16);
		List<String> trainingWords_4 = trainingWords.subList(0, trainingSize/4);
		
		List<String> testWords = Unigram.readInputFile(testData);
		System.out.println("128 training data length: " + trainingWords_128.size());

		System.out.println("64 training data length: " + trainingWords_64.size());

		System.out.println("16 training data length: " + trainingWords_16.size());

		System.out.println("4 training data length: " + trainingWords_4.size());

		System.out.println("training data length: " + trainingWords.size());
		System.out.println("test data length: " + testWords.size());
		
		Set<String> vocabulary = Unigram.buildVocabulary(trainingData, testData);
		
		HashMap<String, Double> wordFrequency_128 = Unigram.calculateFrequency(trainingWords_128, vocabulary);
		HashMap<String, Double> wordFrequency_64 = Unigram.calculateFrequency(trainingWords_64, vocabulary);
		HashMap<String, Double> wordFrequency_16 = Unigram.calculateFrequency(trainingWords_16, vocabulary);
		HashMap<String, Double> wordFrequency_4 = Unigram.calculateFrequency(trainingWords_4, vocabulary);
		HashMap<String, Double> wordFrequency = Unigram.calculateFrequency(trainingWords, vocabulary);
		
		HashMap<String, Double> mlProb_128 = Unigram.calculateMLEst(wordFrequency_128);
		HashMap<String, Double> mlProb_64 = Unigram.calculateMLEst(wordFrequency_64);
		HashMap<String, Double> mlProb_16 = Unigram.calculateMLEst(wordFrequency_16);
		HashMap<String, Double> mlProb_4 = Unigram.calculateMLEst(wordFrequency_4);
		HashMap<String, Double> mlProb = Unigram.calculateMLEst(wordFrequency);
		
		double alphak = 2;
		HashMap<String, Double> mapProb_128 = Unigram.calculateMAPEst(wordFrequency_128, alphak);
		HashMap<String, Double> mapProb_64 = Unigram.calculateMAPEst(wordFrequency_64, alphak);
		HashMap<String, Double> mapProb_16 = Unigram.calculateMAPEst(wordFrequency_16, alphak);
		HashMap<String, Double> mapProb_4 = Unigram.calculateMAPEst(wordFrequency_4, alphak);
		HashMap<String, Double> mapProb = Unigram.calculateMAPEst(wordFrequency, alphak);
		
		HashMap<String, Double> pdProb_128 = Unigram.calculatePredictiveEst(wordFrequency_128, alphak);
		HashMap<String, Double> pdProb_64 = Unigram.calculatePredictiveEst(wordFrequency_64, alphak);
		HashMap<String, Double> pdProb_16 = Unigram.calculatePredictiveEst(wordFrequency_16, alphak);
		HashMap<String, Double> pdProb_4 = Unigram.calculatePredictiveEst(wordFrequency_4, alphak);
		HashMap<String, Double> pdProb = Unigram.calculatePredictiveEst(wordFrequency, alphak);
		
		List<Double> mlresults_128 = Unigram.predictionResults(mlProb_128, testWords);
		List<Double> mlresults_64 = Unigram.predictionResults(mlProb_64, testWords);
		List<Double> mlresults_16 = Unigram.predictionResults(mlProb_16, testWords);
		List<Double> mlresults_4 = Unigram.predictionResults(mlProb_4, testWords);
		List<Double> mlresults = Unigram.predictionResults(mlProb, testWords);
		
		List<Double> mapresults_128 = Unigram.predictionResults(mapProb_128, testWords);
		List<Double> mapresults_64 = Unigram.predictionResults(mapProb_64, testWords);
		List<Double> mapresults_16 = Unigram.predictionResults(mapProb_16, testWords);
		List<Double> mapresults_4 = Unigram.predictionResults(mapProb_4, testWords);
		List<Double> mapresults = Unigram.predictionResults(mapProb, testWords);

		double ml_pp_128 = Equations.perplexity(mlresults_128);
		double ml_pp_64 = Equations.perplexity(mlresults_64);
		double ml_pp_16 = Equations.perplexity(mlresults_16);
		double ml_pp_4 = Equations.perplexity(mlresults_4);
		double ml_pp = Equations.perplexity(mlresults);
		System.out.println("******* ml results *******");
		System.out.println(ml_pp_128 + "  " + ml_pp_64 + "  " + ml_pp_16 + "  " + ml_pp_4 + "  " + ml_pp);
		
		double map_pp_128 = Equations.perplexity(mapresults_128);
		double map_pp_64 = Equations.perplexity(mapresults_64);
		double map_pp_16 = Equations.perplexity(mapresults_16);
		double map_pp_4 = Equations.perplexity(mapresults_4);
		double map_pp = Equations.perplexity(mapresults);
		System.out.println("******* map results *******");
		System.out.println(map_pp_128 + "  " + map_pp_64 + "  " + map_pp_16 + "  " + map_pp_4 + "  " + map_pp);
		
		List<Double> pdresults_128 = Unigram.predictionResults(pdProb_128, testWords);
		List<Double> pdresults_64 = Unigram.predictionResults(pdProb_64, testWords);
		List<Double> pdresults_16 = Unigram.predictionResults(pdProb_16, testWords);
		List<Double> pdresults_4 = Unigram.predictionResults(pdProb_4, testWords);
		List<Double> pdresults = Unigram.predictionResults(pdProb, testWords);
		
		double pd_pp_128 = Equations.perplexity(pdresults_128);
		double pd_pp_64 = Equations.perplexity(pdresults_64);
		double pd_pp_16 = Equations.perplexity(pdresults_16);
		double pd_pp_4 = Equations.perplexity(pdresults_4);
		double pd_pp = Equations.perplexity(pdresults);

		System.out.println("******* predictive distribution results *******");
		System.out.println(pd_pp_128 + "  " + pd_pp_64 + "  " + pd_pp_16 + "  " + pd_pp_4 + "  " + pd_pp);
	}
}
