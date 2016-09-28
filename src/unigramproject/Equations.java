package unigramproject;

import java.util.List;

/**
 * class contains all helpful math equations we need for this project
 * @author Zhaokun Xue
 *
 */
public class Equations {
	
	/**
	 * The function calculates the perplexity
	 * @param probs the probabilities results from testing data
	 * @return the perplexity for this testing data
	 */
	public static double perplexity(List<Double> probs) {
		double lnsum = 0; 
		double n = probs.size();
		// iterate each probablities in the list
		for(double prob : probs) {
			if (prob == 0) { // to avoid ln(0) error, we set ln(0) = -inf
				lnsum += Double.NEGATIVE_INFINITY;
			} else {
				lnsum += Math.log(prob);
			}
		}
		return Math.exp((-1/n)*lnsum);
	}
	
	/**
	 * The function calculates the ML k-th word
	 * @param mk  No. of times k-th word of vocabulary appears in the document
	 * @param n Total no. of words in the document
	 * @return the ML for k-th word
	 */
	public static double mlEstimate (double mk, double n) {
		return (mk/n);
	}
	
	/**
	 * The function calculates the MAP k-th word
	 * @param mk No. of times k-th word of vocabulary appears in the document
	 * @param alphak a scalar for Dirichlet distribution
	 * @param alpha0 the sum of all alphak
	 * @param n Total no. of words in the document
	 * @param k Total no. of words in the vocabulary
	 * @return the MAP for k-th word
	 */
	public static double mapEstimate (double mk, double alphak, double alpha0, double n, double k) {
		return ((mk + alphak - 1)/(n + alpha0 - k));
	}

	/**
	 * The function calculates the Predictive Distribution k-th word
	 * @param mk No. of times k-th word of vocabulary appears in the document
	 * @param alphak a scalar for Dirichlet distribution
	 * @param alpha0 the sum of all alphak
	 * @param n Total no. of words in the document
	 * @return the predictive distribution for k-th word
	 */
	public static double predictiveDis (double mk, double alphak, double alpha0, double n) {
		return ((mk + alphak)/(n + alpha0));
	}
	
	/**
	 * The fucntion calculates the log of gamma function
	 * @param n parameter for gamma function 
	 * @return the log of gamma function
	 */
	public static double logGamma(double n) {
		double result = 0;
		// gamma(n) = (n - 1)!
		// iterate from 1 to (n-1) and calculate the log gamma
		for (double i = 1; i < n; i++) {
			result += Math.log(i);
		}
		return result;
	}
	
}
