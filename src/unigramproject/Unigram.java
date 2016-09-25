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
	
	public static HashMap<String, Double> calculateFrequency (List<String> trainingData, Set<String> vocabulary) {
		HashMap<String, Double> wordsFrequency = new HashMap<String, Double>();
		
		for (String word : vocabulary) {
			wordsFrequency.put(word, (double) 0);
		}
		
		for (String data : trainingData) {
			if (wordsFrequency.containsKey(data)) {
				double currentCount = wordsFrequency.get(data);
				double newCount = currentCount + 1;
				wordsFrequency.put(data, newCount);
			}
		}
		return wordsFrequency;
	} 

	public static double totalNoWords (HashMap<String, Double> wordFrequency) {
	    double totalNoWords = 0;
		for (Map.Entry<String, Double> entry : wordFrequency.entrySet()) {
	        double mk = entry.getValue();
	        totalNoWords += mk; 
	    }
		return totalNoWords;
	}
	
	public static HashMap<String, Double> calculateMLEst (HashMap<String, Double> wordFrequency) {
		HashMap<String, Double> mlProb = new HashMap<String, Double>();
		double n = totalNoWords(wordFrequency);
		for (Map.Entry<String, Double> entry : wordFrequency.entrySet()) {
			String word = entry.getKey();
	        double mk = entry.getValue();
	        double mlest = Equations.mlEstimate(mk, n);
	        mlProb.put(word, mlest);
	    }
		return mlProb;
	}
	
	public static HashMap<String, Double> calculateMAPEst (HashMap<String, Double> wordFrequency,double alphak) {
		HashMap<String, Double> mapProb = new HashMap<String, Double>();
		double n = totalNoWords(wordFrequency);
		double k = wordFrequency.size();
		System.out.println("totla no. words: " + n + " k:" + k);
		double alpha0 = alphak * k;
		for (Map.Entry<String, Double> entry : wordFrequency.entrySet()) {
			String word = entry.getKey();
	        double mk = entry.getValue();
	        double mapest = Equations.mapEstimate(mk, alphak, alpha0, n, k);
	        mapProb.put(word, mapest);
	    }
		return mapProb;
	}

	public static HashMap<String, Double> calculatePredictiveEst (HashMap<String, Double> wordFrequency, double alphak) {
		HashMap<String, Double> predictiveProb = new HashMap<String, Double>();
		double n = totalNoWords(wordFrequency);
		double k = wordFrequency.size();
		double alpha0 = alphak * k;
		for (Map.Entry<String, Double> entry : wordFrequency.entrySet()) {
			String word = entry.getKey();
	        double mk = entry.getValue();
	        double predictiveEst = Equations.predictiveDis(mk, alphak, alpha0, n);
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
		return results;
	}
	
	public static double evidence (Set<String> vocabulary, HashMap<String, Double> wordFrequency, double alphak) {
		double k = vocabulary.size();
		double alphak_mk_product = 1;
		double alphak_product = 1;
		double n = totalNoWords(wordFrequency);
		double alpha0 = alphak * k;
		double gamma_alph0 = Equations.gamma(alpha0);
		double gamma_alph0_N = Equations.gamma(alpha0+n);
		
		for (String word : vocabulary) {
			double mk = wordFrequency.get(word);
			alphak_mk_product = alphak_mk_product * Equations.gamma(alphak + mk);
			alphak_product = alphak_product * Equations.gamma(alphak);
		}
		
		return (gamma_alph0 * alphak_mk_product) / (gamma_alph0_N * alphak_product);
	}
	
	public static void main(String[] args) throws IOException {
		String dataPath = "data/";
		String trainingData = dataPath + "training_data.txt";
		String testData = dataPath + "test_data.txt";
		Task1.task1Tets(trainingData, testData);
		
	}

}
