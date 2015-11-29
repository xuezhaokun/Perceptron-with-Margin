package project4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;
public class Perceptron {
	private static int iterations = 50; //I
	/**
	 * read data from file using weka instance feature
	 * @param filename input file name
	 * @return a list of PwMData
	 * @throws IOException
	 */
	public static List<PwMData> readDataFile(String filename) throws IOException {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		Instances read_data = new Instances(inputReader);
		
		List<PwMData> list_data = new ArrayList<PwMData>();
		for(int i = 0; i < read_data.numInstances(); i ++){
			List<Double> attributes = new ArrayList<Double>();
			Instance current_instance = read_data.instance(i);
			int num_attributes = current_instance.numAttributes() ;
			int classifier = (int) current_instance.value(num_attributes - 1);
			for(int j = 0; j < num_attributes - 1; j++){
				attributes.add(current_instance.value(j));
			}
			PwMData knn_data = new PwMData(attributes, classifier); 
			list_data.add(knn_data);
		}
		return list_data;
	}
	
	private static double calculateTau(List<PwMData> data){
		double norm  = 0;
		for(PwMData d : data){
			for(Double x : d.getData()){
				norm += Math.pow(x, 2);
			}
		}
		norm = Math.sqrt(norm);
		double tau = 0.1*(norm/data.size());
		return tau;
	}
	
	public static double[] primalPwM(List<PwMData> data){
		int w_size = data.get(0).getNum_attributes();
		double[] w = new double[w_size];
		double tau = calculateTau(data);
		for(int i = 0; i < iterations; i++){
			for(PwMData d : data){
				double innerProductOfwAndx = calculatePrimalInnerProduct(w, d.getData());
				int predict = signFunction(innerProductOfwAndx);
				int dataClassifier = d.getClassifier();
				if(dataClassifier*innerProductOfwAndx < tau){
					for(int k = 0; k < w_size; k++){
						w[k] = w[k] + dataClassifier*d.getData().get(k);
					}
				}
			}
		}
		return w;
	}
	
	private static double calculatePrimalInnerProduct(double[] vector1, List<Double> data){
		double sum = 0;
		if(vector1.length != data.size()){
			throw new IllegalArgumentException("Two vectors do not have same length");
		}else{
			for(int k = 0; k < vector1.length; k++){
				sum += vector1[k] * data.get(k);
			}
		}
		return sum;
	}
	
	private static double[] dualPwM(List<PwMData> data){
		int trainingDataSize = data.size();
		double[] alpha = new double[trainingDataSize];
		double tau = calculateTau(data);
		for(int j = 0; j < iterations; j++){
			for(int i = 0; i < data.size(); i++){
				PwMData d = data.get(i);
				double dualSum = 0;
				int classifier = d.getClassifier();
				for(int k = 0; k < trainingDataSize; k++){
					double innerProductOfTwoData = calculateDualInnerProduct(data.get(k).getData(), d.getData());
					dualSum += alpha[k]*data.get(k).getClassifier()*innerProductOfTwoData;
				}
				int predict = signFunction(dualSum);
				if(classifier*dualSum < tau){
					alpha[i] = alpha[i] + 1;
				}
			}
		}
		return alpha;
	}
	
	private static double calculateDualInnerProduct(List<Double> data1, List<Double> data2){
		double sum = 0;
		if(data1.size() != data2.size()){
			throw new IllegalArgumentException("Two vectors do not have same length");
		}else{
			for(int k = 0; k < data1.size(); k++){
				sum += data1.get(k) * data2.get(k);
			}
		}
		return sum;
	}
	
	private static int signFunction(double d){
		if(d >= 0){
			return 1;
		}else{
			return -1;
		}
	} 
	
	private static double polyKernel(double[] u, double[] v, int d){
		return 0;
	}
	
	private static double RBFKernel(double[] u, double[] v, double s){
		return 0;
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
