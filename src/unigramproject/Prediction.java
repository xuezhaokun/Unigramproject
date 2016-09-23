package unigramproject;

public class Prediction {
	
	public static double mlEstimate (int mk, int n) {
		return (mk/n);
	}
	
	public static double mapEstimate (int mk, int alphak, int alpha0, int n, int k) {
		return (mk + alphak - 1)/(n + alpha0 - k);
	}
	
	public static double predictiveDis (int mk, int alphak, int alpha0, int n) {
		return (mk + alphak)/(n + alpha0);
	}
}
