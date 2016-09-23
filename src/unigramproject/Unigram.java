package unigramproject;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.io.*;

public class Unigram {
	private static Set<String> vocabulary = new HashSet<String>();
	
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
	
	public static void buildVocabulary(String trainFile, String testFile) throws IOException {
		List<String> allwords = new ArrayList<String>();
		List<String> trainingWords = Unigram.readInputFile(trainFile);
		List<String> testWords = Unigram.readInputFile(testFile);
		allwords.addAll(trainingWords);
		allwords.addAll(testWords);
		vocabulary.addAll(allwords);
		System.out.println("vocabulary size: " + vocabulary.size());
	}

	public static void main(String[] args) throws IOException {
		String dataPath = "data/";
		String trainingData = dataPath + "training_data.txt";
		String testData = dataPath + "test_data.txt";
		List<String> trainingWords = Unigram.readInputFile(trainingData);
		List<String> testWords = Unigram.readInputFile(testData);
		System.out.println("training data length: " + trainingWords.size());
		System.out.println("test data length: " + testWords.size());
		Unigram.buildVocabulary(trainingData, testData);
		
	}

}
