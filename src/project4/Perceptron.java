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
	
	
	public double[] primalPwM(){
		List<PwMData> data = this.getTrainData();
		int w_size = data.get(0).getNum_attributes()+1;
		double[] w = new double[w_size];
		double tau = calculateTau(data);
		for(PwMData p : data){
			double featureConstant = 1;
			List<Double> newData = p.getData();
			newData.add(featureConstant);
			p.setData(newData);
		}
		for(int i = 0; i < iterations; i++){
			for(PwMData pdata : data){
				List<Double> d = pdata.getData();
				double innerProductOfwAndx = calculatePrimalInnerProduct(w, d);
				int predict = signFunction(innerProductOfwAndx);
				int dataClassifier = pdata.getClassifier();
				if((dataClassifier * innerProductOfwAndx) < tau){
					//System.out.println("update w");
					for(int k = 0; k < w_size; k++){
						w[k] = w[k] + dataClassifier * (d.get(k));
					}
				}
			}
		}

		return w;
	}
	

	// kernel = 0 polyKernel, kernel = 1 rbfKernel
	public double[] dualPwM(int kernel, double param){
		List<PwMData> data = this.getTrainData(); 
		int trainingDataSize = data.size();
		double[] alpha = new double[trainingDataSize];
		double tau = calculateKernelTau(kernel, param);
		System.out.println(alpha.length +", tau: "+tau);
		
		for(int j = 0; j < iterations; j++){
			
			for(int i = 0; i < data.size(); i++){
				//PwMData d = data.get(i); 
				List<Double> xi = data.get(i).getData();
				double dualSum = 0;
				
				int yi = data.get(i).getClassifier();//classifier of xi
				
				for(int k = 0; k < trainingDataSize; k++){
					List<Double> xk = data.get(k).getData();
					int yk = data.get(k).getClassifier();
					double kernelResult = 0;
					
					if(kernel == 0){
						kernelResult = polyKernel(xk, xi, param);
					}else if (kernel == 1){
						kernelResult = rbfKernel(xk, xi, param);
					}else{
						throw new IllegalArgumentException("No such kernel");
					}
					dualSum += alpha[k] * yk * kernelResult;
				}
				
				int predict = signFunction(dualSum);
				if(yi*dualSum < tau){
					alpha[i] = alpha[i] + 1;
				}
			}
			
		}
		return alpha;
	}
	
	public double calculatePrimalInnerProduct(double[] vector1, List<Double> data){
		double sum = 0;
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
		double tau = 0.1 * (totalNorm / data.size());
		System.out.println("primal tau is " + tau);
		return tau;
	}

	public double calculateKernelTau(int kernel, double param){
		double norm  = 0;
		for(PwMData d : this.getTrainData()){
			if(kernel == 0){
				norm += Math.sqrt(polyKernel(d.getData(), d.getData(), param));
			}else if (kernel == 1){
				norm += Math.sqrt(rbfKernel(d.getData(), d.getData(), param));
			}else{
				throw new IllegalArgumentException("No such kernel");
			}
		}
		double tau = 0.1 * (norm / this.getTrainData().size());
		return tau;
	}
	
	public double polyKernel(List<Double> u, List<Double> v, double d){
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
	
	public double rbfKernel(List<Double> u, List<Double> v, double s){
		double dist = 0;
		if(u.size() == v.size()){
			for(int i = 0; i < u.size(); i++){
				dist += Math.pow((u.get(i) - v.get(i)), 2);
			}
		}
		double power = -dist/(2 * Math.pow(s, 2));
		double kernelResult = Math.exp(power);
		return kernelResult;
	}
	
	public int signFunction(double d){
		if(d >= 0){
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
		System.out.println("primal accuracy: " + accuracy);
		
		System.out.println("************ kernel version ***********");
		List<PwMData> trainData2 = readDataFile("data/backTrain.arff");
		List<PwMData> testData2 = readDataFile("data/backTest.arff");
		Perceptron p2 = new Perceptron(trainData2, testData2);
		int kernel = 1;
		
		System.out.println("train: " + trainData2.size() + "test: " + testData2.size());
		double[] rbfparam = new double[]{0.1, 0.5, 1}; 
		for(double d : rbfparam){
		//for(double d = 1; d < 6; d++){
			System.out.println(d);
			double[] alphaPoly = p2.dualPwM(kernel, d);
			double c = 0;
			//System.out.println(c);
			for(int i = 0; i < p2.getTestData().size(); i++){
				PwMData xi = p2.getTestData().get(i);
				double sum = 0;
				int yi = xi.getClassifier();
				
				for(int k = 0; k < p2.getTrainData().size(); k++){
					double kernelResult = 0;
					PwMData xk = p2.getTrainData().get(k);
					int yk = xk.getClassifier();
					if(kernel == 0){
						kernelResult = p2.polyKernel(xk.getData(), xi.getData(), d);
					}else{
						kernelResult = p2.rbfKernel(xk.getData(), xi.getData(), d);
					}
					sum += alphaPoly[k] * yk * kernelResult;
				}
				//System.out.println(dualSum);
				int predict = p2.signFunction(sum);
				if(predict == yi){
					c++;
				}
			}
				
			System.out.println(c);
			System.out.println(p2.getTestData().size());
			double a = c/(p2.getTestData().size());
			System.out.println("kernel accuracy:" + a);
			
		}
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
