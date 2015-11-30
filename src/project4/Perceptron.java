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
	private static int iterations = 50; //I
	private List<PwMData> trainData;
	private List<PwMData> testData;
	
	
	public Perceptron(List<PwMData> trainData, List<PwMData> testData) {
		this.trainData = trainData;
		this.testData = testData;
	}

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
			PwMData pwmData = new PwMData(attributes, classifier); 
			list_data.add(pwmData);
		}
		return list_data;
	}
	
	public double calculateTau(List<PwMData> data){
		double totalNorm = 0;
		for(PwMData d : data){
			double norm  = 0;
			for(Double x : d.getData()){
				norm += Math.pow(x, 2);
			}
			norm = Math.sqrt(norm);
			totalNorm += norm;
		}
		double tau = 0.1*(totalNorm/data.size());
		System.out.println("tau is " + tau);
		return tau;
	}

	public double calculateKernelTau(List<PwMData> data, int kernel, double param){
		double norm  = 0;
		for(PwMData d : data){
			if(kernel == 0){
				norm += Math.sqrt(polyKernel(d.getData(), d.getData(), param));
			}else if (kernel == 1){
				norm += Math.sqrt(rbfKernel(d.getData(), d.getData(), param));
			}else{
				throw new IllegalArgumentException("No such kernel");
			}
		}
		double tau = 0.1*(norm/data.size());
		return tau;
	}
	
	public double[] primalPwM(){
		List<PwMData> data = this.getTrainData();
		int w_size = data.get(0).getNum_attributes()+1;
		double[] w = new double[w_size];
		double tau = calculateTau(data);
		//System.out.println("iteration: " + iterations);
		for(PwMData pdata : data){
			double featureConstant = 1;
			pdata.getData().add(featureConstant);
		}
		for(int i = 0; i < iterations; i++){
			//System.out.println(i);
			for(PwMData pdata : data){
				List<Double> d = pdata.getData();
				//double featureConstant = 1;
				//d.add(featureConstant);
				double innerProductOfwAndx = calculatePrimalInnerProduct(w, d);
				int predict = signFunction(innerProductOfwAndx);
				int dataClassifier = pdata.getClassifier();
				if(dataClassifier*innerProductOfwAndx < tau){
					for(int k = 0; k < w_size; k++){
						w[k] = w[k] + dataClassifier*(d.get(k));
					}
				}
			}
		}
		return w;
	}
	
	public double calculatePrimalInnerProduct(double[] vector1, List<Double> data){
		double sum = 0;
		//double featureConstant = 1;
		//System.out.println(vector1.length + ":" + data.size());
		if(vector1.length != data.size()){
			System.out.println(data);
			throw new IllegalArgumentException("Two vectors do not have same length");
		}else{
			for(int k = 0; k < vector1.length; k++){
				sum += vector1[k] * data.get(k);
			}
		}
		return sum;
	}
	// kernel = 0 polyKernel, kernel = 1 dualKernel
	public double[] dualPwM(List<PwMData> data, int kernel, double param){
		int trainingDataSize = data.size();
		double[] alpha = new double[trainingDataSize];
		double tau = calculateKernelTau(data, kernel, param);
		for(int j = 0; j < iterations; j++){
			for(int i = 0; i < data.size(); i++){
				PwMData d = data.get(i);
				double dualSum = 0;
				int classifier = d.getClassifier();
				for(int k = 0; k < trainingDataSize; k++){
					double kernelResult = 0;
					if(kernel == 0){
						kernelResult = polyKernel(data.get(k).getData(), d.getData(), param);
					}else if (kernel == 1){
						kernelResult = rbfKernel(data.get(k).getData(), d.getData(), param);
					}else{
						throw new IllegalArgumentException("No such kernel");
					}
					//double innerProductOfTwoData = calculateDualInnerProduct(data.get(k).getData(), d.getData());
					dualSum += alpha[k]*data.get(k).getClassifier()*kernelResult;
				}
				int predict = signFunction(dualSum);
				if(classifier*dualSum < tau){
					alpha[i] = alpha[i] + 1;
				}
			}
		}
		return alpha;
	}
	
	/*private static double calculateDualInnerProduct(List<Double> data1, List<Double> data2){
		double sum = 0;
		if(data1.size() != data2.size()){
			throw new IllegalArgumentException("Two vectors do not have same length");
		}else{
			for(int k = 0; k < data1.size(); k++){
				sum += data1.get(k) * data2.get(k);
			}
		}
		return sum;
	}*/
	
	public int signFunction(double d){
		if(d >= 0){
			return 1;
		}else{
			return -1;
		}
	} 
	
	private static double polyKernel(List<Double> u, List<Double> v, double d){
		double innerProduct = 0;
		if(u.size() != v.size()){
			throw new IllegalArgumentException("Two vectors do not have same length");
		}else{
			for(int k = 0; k < u.size(); k++){
				innerProduct += u.get(k) * v.get(k);
			}
		}
		innerProduct = innerProduct + 1;
		double kernelResult = Math.pow(innerProduct, d);
		return kernelResult;
	}
	
	private static double rbfKernel(List<Double> u, List<Double> v, double s){
		double dist = 0;
		if(u.size() == v.size()){
			for(int i = 0; i < u.size(); i++){
				dist += Math.pow((u.get(i) - v.get(i)), 2);
			}
		}
		double power = -dist/(2*Math.pow(s, 2));
		double kernelResult = Math.exp(power);
		return kernelResult;
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

	public static void main(String[] args) throws IOException {
		List<PwMData> trainData = readDataFile("data/ATrain.arff");
		List<PwMData> testData = readDataFile("data/ATest.arff");
		Perceptron p1 = new Perceptron(trainData, testData);
		double[] w = p1.primalPwM();
		double correctPrediction = 0;
		for(PwMData instance : p1.getTestData()){
			List<Double> data = instance.getData();
			data.add((double) 1);
			double innerProductOfwAndx = p1.calculatePrimalInnerProduct(w, data);
			int predict = p1.signFunction(innerProductOfwAndx);
			if(predict == instance.getClassifier()){
				correctPrediction++;
			}
		}
		System.out.println(correctPrediction);
		System.out.println(testData.size());
		double accuracy = correctPrediction/(testData.size());
		System.out.println(accuracy);
		
		/*File folder = new File("data");
		File[] listOfFiles = folder.listFiles();
		for (int i = 0; i < listOfFiles.length; i++) {
			if (listOfFiles[i].isFile()) {
				System.out.println("File " + listOfFiles[i].getName());
				if(listOfFiles[i].getName().contains("Test")){
					System.out.println("test:" + listOfFiles[i].getName());
				}
			} else if (listOfFiles[i].isDirectory()) {
				System.out.println("Directory " + listOfFiles[i].getName());
		    }
		}*/
		
		
	}

}
