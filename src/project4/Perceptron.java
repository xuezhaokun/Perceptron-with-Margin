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
			if(classifier == 0){
				classifier = -1;
			}
			for(int j = 0; j < num_attributes - 1; j++){
				attributes.add(current_instance.value(j));
			}
			PwMData pwmData = new PwMData(attributes, classifier); 
			list_data.add(pwmData);
		}
		inputReader.close();
		return list_data;
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
						tempSum = alpha[k] * yk * polyKernel(xk, xi, param); 
					}else if(kernel == 1){ // rbfkernel
						tempSum = alpha[k] * yk * rbfKernel(xk, xi, param);
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
				result += Math.sqrt(polyKernel(p.getData(), p.getData(), param));
			}else if(kernel == 1){ // rbfkernel
				result += Math.sqrt(rbfKernel(p.getData(), p.getData(), param));
			}else{
				throw new IllegalArgumentException("No such kernel");
			}
		}
		double gamma = 0.1 * (result / trainSize);
		return gamma;
	}

	// poly kernel
	public double polyKernel(List<Double> u, List<Double> v, double d){
		double result = 0;
		if(u.size() == v.size()){ // inner product of u and v
			for(int i = 0; i < u.size(); i++){
				result += u.get(i) * v.get(i);
			}
		}else{
			throw new IllegalArgumentException("Two vectors do not have same length");	
		}
		result = result + 1; // + 1
		result = Math.pow(result, d); // power d
		return result;
	}

	// RBF kernel
	public double rbfKernel(List<Double> u, List<Double> v, double s){
		double dist = 0;
		if(u.size() == v.size()){ 
			for(int i = 0; i < u.size(); i++){ // calc ||u-v||^2
				double diff = u.get(i) - v.get(i);
				dist += Math.pow(diff, 2);
			}
		}
		double power = -(dist / (2 * Math.pow(s, 2)));
		double result = Math.exp(power); 
		return result;
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
	
	public static void main(String[] args) throws IOException {
		List<PwMData> trainData = readDataFile("data/BTrain.arff");
		List<PwMData> testData = readDataFile("data/BTest.arff");
		Perceptron p1 = new Perceptron(trainData, testData);
		double[] w = p1.primalPwM();
		double correctPrediction = 0;
		for(PwMData instance : p1.getTestData()){
			List<Double> data = instance.getData();
			data.add((double) 1);
			double innerProductOfwAndx = p1.calculatePrimalInnerProduct(w, data);
			int predict = p1.sign(innerProductOfwAndx);
			if(predict == instance.getClassifier()){
				correctPrediction++;
			}
		}
		System.out.println(correctPrediction);
		System.out.println(testData.size());
		double accuracy = correctPrediction/(testData.size());
		System.out.println("primal accuracy: " + accuracy);
		

		List<PwMData> trainData2 = readDataFile("data/BTrain.arff");
		List<PwMData> testData2 = readDataFile("data/BTest.arff");
		//double[] rbfparam = new double[]{0.1, 0.5, 1}; 
		//for(double d : rbfparam){
		for (double d = 1; d < 6; d++){
			Perceptron p2 = new Perceptron(trainData2, testData2);
			List<PwMData> trainDataSet = p2.getTrainData();
			int trainSize = trainDataSet.size();
			int testSize = p2.getTestData().size();
			int kernel = 0;
			double[] alpha = p2.dualPwM(kernel, d);
			double correct_predictions = 0;
			for(PwMData test : p2.getTestData()){
				List<Double> test_data = test.getData();
				int yi = test.getClassifier();
				double tester = 0;
				for(int k = 0; k < trainSize; k++){
					double kernelResult = 0;
					//System.out.println(trainDataSet.get(k));
					List<Double> xk = trainDataSet.get(k).getData();
					int yk = trainDataSet.get(k).getClassifier();
					if(kernel == 0){
						kernelResult = p2.polyKernel(xk, test_data, d);
					}else{
						kernelResult = p2.rbfKernel(xk, test_data, d);
					}
					tester += alpha[k] * yk * kernelResult;
				}
				int prediction = p2.sign(tester);
				if(prediction == yi){
					correct_predictions++;
				}
			}
			System.out.println("*********** kernel perceptron *********** d=" + d);
			double arr_rate = correct_predictions / testSize;
			System.out.println(correct_predictions);
			System.out.println(testSize);
			System.out.println("dual accuracy: " + arr_rate);
		}
	}
}