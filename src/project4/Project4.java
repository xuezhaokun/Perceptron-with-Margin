package project4;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

public class Project4 {

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
	
	/**
	 * test primal perceptron
	 * @param trainFileName training data file name
	 * @param testFileName test data file name
	 * @return test accuracy
	 * @throws IOException
	 */
	public static double testPrimalPerceptron(String trainFileName, String testFileName) throws IOException{
		List<PwMData> trainDataSet = readDataFile(trainFileName); // get training data set 
		List<PwMData> testDataSet = readDataFile(testFileName); // get test data set
		Perceptron p = new Perceptron(trainDataSet, testDataSet); // init perceptron
		int testSize = p.getTestData().size(); // test data size
		double[] w = p.primalPwM(); // calc weight vector w
		double correctPredictions = 0; // correct prediction
		for(PwMData instance : p.getTestData()){ // iterate each test data
			List<Double> xi = instance.getData(); // current test data
			xi.add((double) 1); // add constant feature with value of 1 
			double innerProductOfwAndx = p.calculatePrimalInnerProduct(w, xi); // calc inner product of w and xi
			int predict = p.sign(innerProductOfwAndx); // prediction
			if(predict == instance.getClassifier()){
				correctPredictions++; // increase correct prediction
			}
		}
		double accuracy = correctPredictions / testSize; //accuracy
		return accuracy;
	}
	
	/**
	 * test kernel perceptron
	 * @param trainFileName training data file name
	 * @param testFileName test data file name
	 * @return test accuracy
	 * @throws IOException
	 */
	public static double testDualPerceptron(String trainFileName, String testFileName, int kernel, double param) throws IOException{
		List<PwMData> trainDataSet = readDataFile(trainFileName); // get training data set
		List<PwMData> testDataSet = readDataFile(testFileName); // get test data set
		Perceptron p = new Perceptron(trainDataSet, testDataSet); // init perceptron
		int trainSize = p.getTrainData().size(); // training data size
		int testSize = p.getTestData().size(); // test data size 
		double correctPredictions = 0; // correct predictrions
		double[] alpha = p.dualPwM(kernel, param); // init alpha
		
		for(PwMData test : p.getTestData()){ // iterate each test data
			List<Double> test_data = test.getData(); // get test data
			int yi = test.getClassifier(); // get current test data label
			double tester = 0; // tester to 0
			for(int k = 0; k < trainSize; k++){ // iterate training data set
				double kernelResult = 0; // init to 0
				PwMData pk = trainDataSet.get(k); // current training data
				List<Double> xk = pk.getData(); // get data
				int yk = pk.getClassifier(); // traing data label
				if(kernel == 0){ // poly kernel
					kernelResult = PwMData.polyKernel(pk, test, param);
				}else{ // rbf kernel
					kernelResult = PwMData.rbfKernel(pk, test, param);
				}
				tester += alpha[k] * yk * kernelResult;
			}
			int prediction = p.sign(tester); // predict
			if(prediction == yi){
				correctPredictions++;
			}
		}
		double accuracy = correctPredictions / testSize; // accuracy
		return accuracy;
	}
	
	/**
	 * test primal knn
	 * @param trainFileName training data file name
	 * @param testFileName test data file name
	 * @return test accuracy
	 * @throws IOException
	 */
	public static double testPrimalKnn(String trainFileName, String testFileName) throws IOException{
		List<PwMData> trainDataSet = readDataFile(trainFileName); // get training data set
		List<PwMData> testDataSet = readDataFile(testFileName); // get test data set
		Knn knn = new Knn(trainDataSet, testDataSet); // init knn
		int testSize = testDataSet.size(); // test data size 
		double correctPredictions = 0; 
		for(PwMData kd : knn.getTest_data_set()){ // iterate test data
			int kd_classifier = kd.getClassifier(); // current data label
			List<PwMData> nearest_neighbor = knn.getKNearestNeighbors(kd); // find nearest neighbor
			int prediction = knn.determineClass(nearest_neighbor); // determine class
			if(prediction == kd_classifier){
				correctPredictions++;
			}
		}
		double accuracy = correctPredictions / testSize; // accuracy
		return accuracy;
	}
	
	/**
	 * test kernel knn
	 * @param trainFileName training data file name
	 * @param testFileName test data file name
	 * @return test accuracy
	 * @throws IOException
	 */
	public static double testDualKnn(String trainFileName, String testFileName, int kernel, double param) throws IOException{
		List<PwMData> trainDataSet = readDataFile(trainFileName); // get training data set
		List<PwMData> testDataSet = readDataFile(testFileName); // get test data set
		Knn knn = new Knn(trainDataSet, testDataSet); // init knn
		int testSize = testDataSet.size(); // test data size
		double correctPredictions = 0; 
		for(PwMData kd : knn.getTest_data_set()){ //iterate each test data
			int kd_classifier = kd.getClassifier(); // get current data label
			List<PwMData> nearest_neighbor = knn.getKNearestNeighbors(kd, kernel, param); // calc nearest neighbor
			int prediction = knn.determineClass(nearest_neighbor); //determin class
			if(prediction == kd_classifier){
				correctPredictions++;
			}
		}
		double accuracy = correctPredictions / testSize;
		return accuracy;
	}	
	
	/**
	 * main function
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		double[] rbfS = new double[]{0.1, 0.5, 1};
		File folder = new File(".");
		File[] listOfFiles = folder.listFiles();
		List<String> trainFileNames = new ArrayList<String>();
		List<String> testFileNames = new ArrayList<String>();
		for (int i = 0; i < listOfFiles.length; i++) {
			if (listOfFiles[i].isFile()) {
				if(listOfFiles[i].getName().contains("Train")){
					String trainFileName = listOfFiles[i].getName();
					trainFileNames.add(trainFileName);
				}else if(listOfFiles[i].getName().contains("Test")){
					String testFileName = listOfFiles[i].getName();
					testFileNames.add(testFileName);
				}
			}
		}
		Collections.sort(trainFileNames.subList(1, trainFileNames.size()));
		Collections.sort(testFileNames.subList(1, testFileNames.size()));
		if(trainFileNames.size() == testFileNames.size()){
			for(int i = 0; i < trainFileNames.size(); i++){
				String[] trainFileName = trainFileNames.get(i).split("(?=\\p{Upper})");
				String[] testFileName = testFileNames.get(i).split("(?=\\p{Upper})");
				if(trainFileName[0].equals(testFileName[0])){
					System.out.println("*************** results for " + trainFileNames.get(i) + " : " + testFileNames.get(i) + " *****************");
					System.out.println("order: primal | polyKernel d=1 | polyKernel d=2 | polyKernel d=3 | polyKernel d=4 | polyKernel d=5 | rbfKernel s=0.1 | rbfKernel s=0.5 | rbfKernel s=1");
					double knnPrimalAccuracy = testPrimalKnn(trainFileNames.get(i), testFileNames.get(i));
					System.out.print("KNN: " + knnPrimalAccuracy + " | ");
					for(int d = 1; d < 6; d++){
						double knnPolyKernelResult = testDualKnn(trainFileNames.get(i), testFileNames.get(i), 0, d);
						System.out.print(knnPolyKernelResult + " | ");
					}
					for(double s : rbfS){
						double knnrbfKernelResult = testDualKnn(trainFileNames.get(i), testFileNames.get(i), 1, s);
						System.out.print(knnrbfKernelResult + " | ");
					}
					System.out.println("");
					double perceptronPrimalAccuracy = testPrimalPerceptron(trainFileNames.get(i), testFileNames.get(i));
					System.out.print("Perceptron: " + perceptronPrimalAccuracy + " | ");
					for(int d = 1; d < 6; d++){
						double perceptronpolyKernelResult = testDualPerceptron(trainFileNames.get(i), testFileNames.get(i), 0, d);
						System.out.print(perceptronpolyKernelResult + " | ");
					}
					for(double s : rbfS){
						double perceptronrbfKernelResult = testDualPerceptron(trainFileNames.get(i), testFileNames.get(i), 1, s);
						System.out.print(perceptronrbfKernelResult + " | ");
					}
					System.out.println("");
				}else{
					throw new IllegalArgumentException("Your corresponding Train and Test files' names do not have the same prefix, please check it");
				}
			}
		}else{
			throw new IllegalArgumentException("Test and Train files are not paired, please check your files");
		}
	}
}
