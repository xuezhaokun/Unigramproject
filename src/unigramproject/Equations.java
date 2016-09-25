package unigramproject;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Equations {
	
	public static double perplexity(List<Double> probs) {
		double lnsum = 0;
		double n = probs.size();
		// N
		for(double prob : probs) {
			if (prob == 0) {
				lnsum += Double.NEGATIVE_INFINITY;
			} else {
				lnsum += Math.log(prob);
			}
		}
		return Math.exp((-1/n)*lnsum);
	}
	
	public static double mlEstimate (double mk, double n) {
		return (mk/n);
	}
	
	public static double mapEstimate (double mk, double alphak, double alpha0, double n, double k) {
		return ((mk + alphak - 1)/(n + alpha0 - k));
	}
	
	public static double predictiveDis (double mk, double alphak, double alpha0, double n) {
		return ((mk + alphak)/(n + alpha0));
	}
	
	public static BigDecimal gamma(double x) {
		return factorial(x-1);
	}
	
	public static BigDecimal factorial(double n) {
		 BigDecimal fact = BigDecimal.valueOf(1);
		 for (int i = 1; i <= n; i++){
		        fact = fact.multiply(BigDecimal.valueOf(i));
		 }
		return fact;
	}
	
}
