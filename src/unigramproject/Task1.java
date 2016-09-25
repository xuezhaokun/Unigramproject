package unigramproject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

public class Task1 {
	public static void task1Test(String trainingData, String testData) throws IOException {
		List<String> trainingWords = Unigram.readInputFile(trainingData);
		int trainingSize = trainingWords.size();
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
		
		List<String> filenames = new ArrayList<String> ();
		filenames.add(trainingData);
		filenames.add(testData);
		Set<String> vocabulary = Unigram.buildVocabulary(filenames);
		
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
		
		System.out.println("******* ml results *******");
		List<Double> train_mlresults_128 = Unigram.predictionResults(mlProb_128, trainingWords_128);
		List<Double> train_mlresults_64 = Unigram.predictionResults(mlProb_64, trainingWords_64);
		List<Double> train_mlresults_16 = Unigram.predictionResults(mlProb_16, trainingWords_16);
		List<Double> train_mlresults_4 = Unigram.predictionResults(mlProb_4, trainingWords_4);
		List<Double> train_mlresults = Unigram.predictionResults(mlProb, trainingWords);
		
		double train_ml_pp_128 = Equations.perplexity(train_mlresults_128);
		double train_ml_pp_64 = Equations.perplexity(train_mlresults_64);
		double train_ml_pp_16 = Equations.perplexity(train_mlresults_16);
		double train_ml_pp_4 = Equations.perplexity(train_mlresults_4);
		double train_ml_pp = Equations.perplexity(train_mlresults);

		System.out.println(train_ml_pp_128 + "  " + train_ml_pp_64 + "  " + train_ml_pp_16 + "  " + train_ml_pp_4 + "  " + train_ml_pp);
		
		List<Double> test_mlresults_128 = Unigram.predictionResults(mlProb_128, testWords);
		List<Double> test_mlresults_64 = Unigram.predictionResults(mlProb_64, testWords);
		List<Double> test_mlresults_16 = Unigram.predictionResults(mlProb_16, testWords);
		List<Double> test_mlresults_4 = Unigram.predictionResults(mlProb_4, testWords);
		List<Double> test_mlresults = Unigram.predictionResults(mlProb, testWords);

		double test_ml_pp_128 = Equations.perplexity(test_mlresults_128);
		double test_ml_pp_64 = Equations.perplexity(test_mlresults_64);
		double test_ml_pp_16 = Equations.perplexity(test_mlresults_16);
		double test_ml_pp_4 = Equations.perplexity(test_mlresults_4);
		double test_ml_pp = Equations.perplexity(test_mlresults);
		
		System.out.println(test_ml_pp_128 + "  " + test_ml_pp_64 + "  " + test_ml_pp_16 + "  " + test_ml_pp_4 + "  " + test_ml_pp);
		
		System.out.println("******* map results *******");
		List<Double> train_mapresults_128 = Unigram.predictionResults(mapProb_128, trainingWords_128);
		List<Double> train_mapresults_64 = Unigram.predictionResults(mapProb_64, trainingWords_64);
		List<Double> train_mapresults_16 = Unigram.predictionResults(mapProb_16, trainingWords_16);
		List<Double> train_mapresults_4 = Unigram.predictionResults(mapProb_4, trainingWords_4);
		List<Double> train_mapresults = Unigram.predictionResults(mapProb, trainingWords);
		
		double train_map_pp_128 = Equations.perplexity(train_mapresults_128);
		double train_map_pp_64 = Equations.perplexity(train_mapresults_64);
		double train_map_pp_16 = Equations.perplexity(train_mapresults_16);
		double train_map_pp_4 = Equations.perplexity(train_mapresults_4);
		double train_map_pp = Equations.perplexity(train_mapresults);
		
		System.out.println(train_map_pp_128 + "  " + train_map_pp_64 + "  " + train_map_pp_16 + "  " + train_map_pp_4 + "  " + train_map_pp);
		
		
		List<Double> test_mapresults_128 = Unigram.predictionResults(mapProb_128, testWords);
		List<Double> test_mapresults_64 = Unigram.predictionResults(mapProb_64, testWords);
		List<Double> test_mapresults_16 = Unigram.predictionResults(mapProb_16, testWords);
		List<Double> test_mapresults_4 = Unigram.predictionResults(mapProb_4, testWords);
		List<Double> test_mapresults = Unigram.predictionResults(mapProb, testWords);
		
		double test_map_pp_128 = Equations.perplexity(test_mapresults_128);
		double test_map_pp_64 = Equations.perplexity(test_mapresults_64);
		double test_map_pp_16 = Equations.perplexity(test_mapresults_16);
		double test_map_pp_4 = Equations.perplexity(test_mapresults_4);
		double test_map_pp = Equations.perplexity(test_mapresults);
		
		System.out.println(test_map_pp_128 + "  " + test_map_pp_64 + "  " + test_map_pp_16 + "  " + test_map_pp_4 + "  " + test_map_pp);

		System.out.println("******* predictive distribution results *******");

		List<Double> train_pdresults_128 = Unigram.predictionResults(pdProb_128, trainingWords_128);
		List<Double> train_pdresults_64 = Unigram.predictionResults(pdProb_64, trainingWords_64);
		List<Double> train_pdresults_16 = Unigram.predictionResults(pdProb_16, trainingWords_16);
		List<Double> train_pdresults_4 = Unigram.predictionResults(pdProb_4, trainingWords_4);
		List<Double> train_pdresults = Unigram.predictionResults(pdProb, trainingWords);
		
		double train_pd_pp_128 = Equations.perplexity(train_pdresults_128);
		double train_pd_pp_64 = Equations.perplexity(train_pdresults_64);
		double train_pd_pp_16 = Equations.perplexity(train_pdresults_16);
		double train_pd_pp_4 = Equations.perplexity(train_pdresults_4);
		double train_pd_pp = Equations.perplexity(train_pdresults);

		System.out.println(train_pd_pp_128 + "  " + train_pd_pp_64 + "  " + train_pd_pp_16 + "  " + train_pd_pp_4 + "  " + train_pd_pp);
		
		List<Double> test_pdresults_128 = Unigram.predictionResults(pdProb_128, testWords);
		List<Double> test_pdresults_64 = Unigram.predictionResults(pdProb_64, testWords);
		List<Double> test_pdresults_16 = Unigram.predictionResults(pdProb_16, testWords);
		List<Double> test_pdresults_4 = Unigram.predictionResults(pdProb_4, testWords);
		List<Double> test_pdresults = Unigram.predictionResults(pdProb, testWords);
		
		double test_pd_pp_128 = Equations.perplexity(test_pdresults_128);
		double test_pd_pp_64 = Equations.perplexity(test_pdresults_64);
		double test_pd_pp_16 = Equations.perplexity(test_pdresults_16);
		double test_pd_pp_4 = Equations.perplexity(test_pdresults_4);
		double test_pd_pp = Equations.perplexity(test_pdresults);

		System.out.println(test_pd_pp_128 + "  " + test_pd_pp_64 + "  " + test_pd_pp_16 + "  " + test_pd_pp_4 + "  " + test_pd_pp);
	}
}
