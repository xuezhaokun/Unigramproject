package unigramproject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.io.*;

public class Unigram {
	
	public static List<String> readInputFile(String filename) throws IOException {
		List<String> words = new ArrayList<String>();
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
            String fileRead = br.readLine();

            while (fileRead != null) {

                String[] tokens = fileRead.split(" ");
                for(String token: tokens) {
                		words.add(token);
                }
                fileRead = br.readLine();
            }

            br.close();
        } catch (FileNotFoundException fnfe) {
            System.out.println("file not found");
        }
		
		return words;
	}
	
	public static Set<String> buildVocabulary(String trainFile, String testFile) throws IOException {
		Set<String> vocabulary = new HashSet<String>();
		List<String> allwords = new ArrayList<String>();
		List<String> trainingWords = Unigram.readInputFile(trainFile);
		List<String> testWords = Unigram.readInputFile(testFile);
		allwords.addAll(trainingWords);
		allwords.addAll(testWords);
		vocabulary.addAll(allwords);
		System.out.println("vocabulary size: " + vocabulary.size());
		return vocabulary;
	}
	
	public static HashMap<String, Integer> calculateFrequency (List<String> trainingData, Set<String> vocabulary) {
		HashMap<String, Integer> wordsFrequency = new HashMap<String, Integer>();
		
		for (String word : vocabulary) {
			wordsFrequency.put(word, 0);
		}
		
		for (String data : trainingData) {
			if (wordsFrequency.containsKey(data)) {
				int currentCount = wordsFrequency.get(data);
				int newCount = currentCount + 1;
				wordsFrequency.put(data, newCount);
			}
		}
		return wordsFrequency;
	} 

	public static int totalNoWords (HashMap<String, Integer> wordFrequency) {
	    int totalNoWords = 0;
		for (Map.Entry<String, Integer> entry : wordFrequency.entrySet()) {
	        Integer mk = entry.getValue();
	        totalNoWords += mk; 
	    }
		return totalNoWords;
	}
	
	public static HashMap<String, Double> calculateMLEst (HashMap<String, Integer> wordFrequency) {
		HashMap<String, Double> mlProb = new HashMap<String, Double>();
		int n = totalNoWords(wordFrequency);
		for (Map.Entry<String, Integer> entry : wordFrequency.entrySet()) {
			String word = entry.getKey();
	        Integer mk = entry.getValue();
	        Double mlest = Equations.mlEstimate(mk, n);
	        mlProb.put(word, mlest);
	    }
		return mlProb;
	}
	
	public static HashMap<String, Double> calculateMAPEst (HashMap<String, Integer> wordFrequency,double alphak) {
		HashMap<String, Double> mapProb = new HashMap<String, Double>();
		int n = totalNoWords(wordFrequency);
		int k = wordFrequency.size();
		double alpha0 = alphak * k;
		System.out.println("alph0 is " + alpha0 + " vocabulary k is " + k);
		for (Map.Entry<String, Integer> entry : wordFrequency.entrySet()) {
			String word = entry.getKey();
	        Integer mk = entry.getValue();
	        Double mapest = Equations.mapEstimate(mk, alphak, alpha0, n, k);
	        mapProb.put(word, mapest);
	    }
		return mapProb;
	}

	public static HashMap<String, Double> calculatePredictiveEst (HashMap<String, Integer> wordFrequency, double alphak) {
		HashMap<String, Double> predictiveProb = new HashMap<String, Double>();
		int n = totalNoWords(wordFrequency);
		int k = wordFrequency.size();
		double alpha0 = alphak * k;
		for (Map.Entry<String, Integer> entry : wordFrequency.entrySet()) {
			String word = entry.getKey();
	        Integer mk = entry.getValue();
	        Double predictiveEst = Equations.predictiveDis(mk, alphak, alpha0, n);
	        predictiveProb.put(word, predictiveEst);
	    }
		return predictiveProb;
	}
	
	public static List<Double> predictionResults (HashMap<String, Double> probs, List<String> testData) {
		List<Double> results = new ArrayList<Double>();
		for (String data : testData) {
			double prob = probs.get(data);
			results.add(prob);
		}
		System.out.println("results size: " + results.size());
		return results;
	}
	
	public static void main(String[] args) throws IOException {
		String dataPath = "data/";
		String trainingData = dataPath + "training_data.txt";
		String testData = dataPath + "test_data.txt";
		
		List<String> trainingWords = Unigram.readInputFile(trainingData);
		int trainingSize = trainingWords.size();
		//int[] trainingSizes = new int[]{trainingSize/128, trainingSize/64, trainingSize/16, trainingSize/4, trainingSize};
		List<String> trainingWords_128 = trainingWords.subList(0, trainingSize/128 - 1);
		List<String> trainingWords_64 = trainingWords.subList(0, trainingSize/64 - 1);
		List<String> trainingWords_16 = trainingWords.subList(0, trainingSize/16 - 1);
		List<String> trainingWords_4 = trainingWords.subList(0, trainingSize/4 - 1);
		
		List<String> testWords = Unigram.readInputFile(testData);
		System.out.println("training data length: " + trainingWords.size());
		System.out.println("test data length: " + testWords.size());
		
		Set<String> vocabulary = Unigram.buildVocabulary(trainingData, testData);
		
		HashMap<String, Integer> wordFrequency_128 = Unigram.calculateFrequency(trainingWords_128, vocabulary);
		HashMap<String, Integer> wordFrequency_64 = Unigram.calculateFrequency(trainingWords_64, vocabulary);
		HashMap<String, Integer> wordFrequency_16 = Unigram.calculateFrequency(trainingWords_16, vocabulary);
		HashMap<String, Integer> wordFrequency_4 = Unigram.calculateFrequency(trainingWords_4, vocabulary);
		HashMap<String, Integer> wordFrequency = Unigram.calculateFrequency(trainingWords, vocabulary);
//		int n = Unigram.totalNoWords(wordFrequency);
//		System.out.println("total count: " + n);

		HashMap<String, Double> mapProb_128 = Unigram.calculateMAPEst(wordFrequency_128, 2);
		HashMap<String, Double> mapProb_64 = Unigram.calculateMAPEst(wordFrequency_64, 2);
		HashMap<String, Double> mapProb_16 = Unigram.calculateMAPEst(wordFrequency_16, 2);
		HashMap<String, Double> mapProb_4 = Unigram.calculateMAPEst(wordFrequency_4, 2);
		HashMap<String, Double> mapProb = Unigram.calculateMAPEst(wordFrequency, 2);
		
		HashMap<String, Double> pdProb_128 = Unigram.calculatePredictiveEst(wordFrequency_128, 2);
		HashMap<String, Double> pdProb_64 = Unigram.calculatePredictiveEst(wordFrequency_64, 2);
		HashMap<String, Double> pdProb_16 = Unigram.calculatePredictiveEst(wordFrequency_16, 2);
		HashMap<String, Double> pdProb_4 = Unigram.calculatePredictiveEst(wordFrequency_4, 2);
		HashMap<String, Double> pdProb = Unigram.calculatePredictiveEst(wordFrequency, 2);
		
		
		List<Double> mapresults_128 = Unigram.predictionResults(mapProb_128, testWords);
		List<Double> mapresults_64 = Unigram.predictionResults(mapProb_64, testWords);
		List<Double> mapresults_16 = Unigram.predictionResults(mapProb_16, testWords);
		List<Double> mapresults_4 = Unigram.predictionResults(mapProb_4, testWords);
		List<Double> mapresults = Unigram.predictionResults(mapProb, testWords);
		
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
