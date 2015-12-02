package project4;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import weka.core.Instance;
import weka.core.Instances;

public class Perceptron {
	//private static int iterations = 50; //I
	private List<PwMData> trainData;
	private List<PwMData> testData;
	
	
	public Perceptron(List<PwMData> trainData, List<PwMData> testData) {
		this.trainData = trainData;
		this.testData = testData;
	}


	// primal perceptron
	public double[] primalPwM(){
		List<PwMData> trainDataSet = this.getTrainData(); // get train data
		int trainSize = trainDataSet.size();
		int length = trainDataSet.get(0).getNum_attributes() + 1; // w size 
		double[] w = new double[length]; // init w to 0s
		// add constant feature with value 1 to data
		for(PwMData p : trainDataSet){
			double featureConstant = 1; // constant feature
			List<Double> currentData = p.getData();
			currentData.add(featureConstant);
			p.setData(currentData);
		}
		double gamma = calculateGamma(trainDataSet); // calculate gamma
		// repeat for 50 iterations
		for(int n = 0; n < 50; n++){
			for(int i = 0; i < trainSize; i++){
				PwMData pi = trainDataSet.get(i);
				List<Double> xi = pi.getData(); // xi
				int yi = pi.getClassifier(); // yi
				double innerProductOfwx = calculatePrimalInnerProduct(w, xi);
				int o = sign(innerProductOfwx);
				if((yi * innerProductOfwx) < gamma){
					for(int k = 0; k < w.length; k++){
						w[k] = w[k] + (yi * xi.get(k));
					}
				}
			}
		}
		return w;
	}

	// dual/kernel perceptron polyKernel = 0; rbfKernel = 1. param: d->polykernel;s->rbfkernel
	public double[] dualPwM(int kernel, double param){
		List<PwMData> trainDataSet = this.getTrainData(); // get trainning data 
		int trainSize = trainDataSet.size(); // training data size
		double[] alpha = new double[trainSize]; // init alpha with 0s
		double tau = calculateKernelGamma(trainDataSet, kernel, param);
		// repeat for 50 iterations
		for(int n = 0; n < 50; n++){
			// iterate each training data
			for(int i = 0; i < trainSize; i++){
				PwMData pi = trainDataSet.get(i); // current data object
				List<Double> xi = pi.getData(); // current data
				int yi = pi.getClassifier(); // current data classifier
				double sumForxi = 0; //sum for each x over k
				// iterate each training data
				for(int k = 0; k < trainSize; k++){
					PwMData pk = trainDataSet.get(k); // current data k
					List<Double> xk = pk.getData(); // xk
					int yk = pk.getClassifier(); // yk
					double tempSum = 0; // current sum
					if(kernel == 0){ // polykernel
						tempSum = alpha[k] * yk * PwMData.polyKernel(pk, pi, param); 
					}else if(kernel == 1){ // rbfkernel
						tempSum = alpha[k] * yk * PwMData.rbfKernel(pk, pi, param);
					}else{
						throw new IllegalArgumentException("No such kernel");
					}
					sumForxi = sumForxi + tempSum;
				}
				int o = sign(sumForxi);
				if((yi * sumForxi) < tau){
					alpha[i] = alpha[i] + 1;
				}
			}
		}
		return alpha;
	}

	// calculate gamma for primal perceptron
	public double calculateGamma(List<PwMData> trainDataSet){
		double normSum = 0; // init sum of norm to 0
		int trainSize = trainDataSet.size(); // training data size
		// iterate each data in the training data
		for(PwMData p : trainDataSet){
			List<Double> xi = p.getData();// current data
			double xi_norm = 0; // current data norm init to be 0
			// iterate each feature
			for(Double d : xi){ 
				xi_norm += Math.pow(d, 2); // each feature's square
			}
			xi_norm = Math.sqrt(xi_norm); // current data norm
			normSum = normSum + xi_norm; // add total norm
		}
		double gamma = 0.1 * (normSum / trainSize); // calc gamma
		return gamma;
	}

	// calculate inner product of w and xi
	public double calculatePrimalInnerProduct(double[] w, List<Double> xi){
		int w_length = w.length; // length of w
		int xi_size = xi.size(); // size of xi
		double result = 0; // init result to 0
		if(w_length == xi_size){ // if w and xi have same length
			for(int i = 0; i < w_length; i++){ // iterate each corresponding element
				result += w[i] * xi.get(i); // calc inner product
			}
		}else{
			System.out.println(xi);
			throw new IllegalArgumentException("Two vectors do not have same length");
		}
		return result;
	}
	
	// calculate gamma for dual/kernel perceptron
	public double calculateKernelGamma(List<PwMData> trainDataSet, int kernel, double param){
		double result = 0;
		int trainSize = trainDataSet.size();
		for(PwMData p : trainDataSet){
			if(kernel == 0){ // polykernel
				result += Math.sqrt(PwMData.polyKernel(p, p, param));
			}else if(kernel == 1){ // rbfkernel
				result += Math.sqrt(PwMData.rbfKernel(p, p, param));
			}else{
				throw new IllegalArgumentException("No such kernel");
			}
		}
		double gamma = 0.1 * (result / trainSize);
		return gamma;
	}

	// sign function
	public int sign(double test){
		if(test >= 0){
			return 1;
		}else{
			return -1;
		}
	}

	public List<PwMData> getTrainData() {
		return trainData;
	}

	public void setTrainData(List<PwMData> trainData) {
		this.trainData = trainData;
	}

	public List<PwMData> getTestData() {
		return testData;
	}

	public void setTestData(List<PwMData> testData) {
		this.testData = testData;
	}
	
}