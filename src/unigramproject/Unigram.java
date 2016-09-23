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
	        Double mlest = Prediction.mlEstimate(mk, n);
	        mlProb.put(word, mlest);
	    }
		return mlProb;
	}
	
	public static HashMap<String, Double> calculateMAPEst (HashMap<String, Integer> wordFrequency) {
		HashMap<String, Double> mapProb = new HashMap<String, Double>();
		int n = totalNoWords(wordFrequency);
		for (Map.Entry<String, Integer> entry : wordFrequency.entrySet()) {
			String word = entry.getKey();
	        Integer mk = entry.getValue();
	        Double mapest = Prediction.mapEstimate(mk, alphak, alpha0, n, k);
	        mapProb.put(word, mapest);
	    }
		return mapProb;
	}

	public static HashMap<String, Double> calculatePredictiveEst (HashMap<String, Integer> wordFrequency) {
		HashMap<String, Double> predictiveProb = new HashMap<String, Double>();
		int n = totalNoWords(wordFrequency);
		for (Map.Entry<String, Integer> entry : wordFrequency.entrySet()) {
			String word = entry.getKey();
	        Integer mk = entry.getValue();
	        Double predictiveEst = Prediction.predictiveDis(mk, alphak, alpha0, n);
	        predictiveProb.put(word, predictiveEst);
	    }
		return predictiveProb;
	}
	
	public static void main(String[] args) throws IOException {
		String dataPath = "data/";
		String trainingData = dataPath + "training_data.txt";
		String testData = dataPath + "test_data.txt";
		List<String> trainingWords = Unigram.readInputFile(trainingData);
		List<String> testWords = Unigram.readInputFile(testData);
		System.out.println("training data length: " + trainingWords.size());
		System.out.println("test data length: " + testWords.size());
		Set<String> vocabulary = Unigram.buildVocabulary(trainingData, testData);
		HashMap<String, Integer> wordFrequency = Unigram.calculateFrequency(trainingWords, vocabulary);
		int n = Unigram.totalNoWords(wordFrequency);
		System.out.println("total count: " + n);
	}

}
