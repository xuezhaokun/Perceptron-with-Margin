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
	
	public static double testPrimalPerceptron(String trainFileName, String testFileName) throws IOException{
		List<PwMData> trainDataSet = readDataFile(trainFileName);
		List<PwMData> testDataSet = readDataFile(testFileName);
		Perceptron p = new Perceptron(trainDataSet, testDataSet);
		int testSize = p.getTestData().size();
		double[] w = p.primalPwM();
		double correctPredictions = 0;
		for(PwMData instance : p.getTestData()){
			List<Double> xi = instance.getData();
			xi.add((double) 1);
			double innerProductOfwAndx = p.calculatePrimalInnerProduct(w, xi);
			int predict = p.sign(innerProductOfwAndx);
			//System.out.println(predict);
			if(predict == instance.getClassifier()){
				correctPredictions++;
			}
		}
		double accuracy = correctPredictions / testSize;
		return accuracy;
	}
	
	public static double testDualPerceptron(String trainFileName, String testFileName, int kernel, double param) throws IOException{
		List<PwMData> trainDataSet = readDataFile(trainFileName);
		List<PwMData> testDataSet = readDataFile(testFileName);
		Perceptron p = new Perceptron(trainDataSet, testDataSet);
		int trainSize = p.getTrainData().size();
		int testSize = p.getTestData().size();
		double correctPredictions = 0;
		double[] alpha = p.dualPwM(kernel, param);
		
		for(PwMData test : p.getTestData()){
			List<Double> test_data = test.getData();
			int yi = test.getClassifier();
			double tester = 0;
			for(int k = 0; k < trainSize; k++){
				double kernelResult = 0;
				//System.out.println(trainDataSet.get(k));
				PwMData pk = trainDataSet.get(k);
				List<Double> xk = pk.getData();
				int yk = pk.getClassifier();
				if(kernel == 0){
					//System.out.println("pk: " +  pk);
					//System.out.println("test: " + test);
					kernelResult = PwMData.polyKernel(pk, test, param);
				}else{
					kernelResult = PwMData.rbfKernel(pk, test, param);
				}
				tester += alpha[k] * yk * kernelResult;
			}
			int prediction = p.sign(tester);
			if(prediction == yi){
				correctPredictions++;
			}
		}
		double accuracy = correctPredictions / testSize;
		return accuracy;
	}
	
	public static double testPrimalKnn(String trainFileName, String testFileName) throws IOException{
		List<PwMData> trainDataSet = readDataFile(trainFileName);
		List<PwMData> testDataSet = readDataFile(testFileName);
		Knn knn = new Knn(trainDataSet, testDataSet);
		int testSize = testDataSet.size();
		double correctPredictions = 0;
		for(PwMData kd : knn.getTest_data_set()){
			int kd_classifier = kd.getClassifier();
			List<PwMData> nearest_neighbor = knn.getKNearestNeighbors(kd);
			int prediction = knn.determineClass(nearest_neighbor);
			if(prediction == kd_classifier){
				correctPredictions++;
			}
		}
		double accuracy = correctPredictions / testSize;
		return accuracy;
	}
	
	public static double testDualKnn(String trainFileName, String testFileName, int kernel, double param) throws IOException{
		List<PwMData> trainDataSet = readDataFile(trainFileName);
		List<PwMData> testDataSet = readDataFile(testFileName);
		Knn knn = new Knn(trainDataSet, testDataSet);
		int testSize = testDataSet.size();
		double correctPredictions = 0;
		for(PwMData kd : knn.getTest_data_set()){
			int kd_classifier = kd.getClassifier();
			List<PwMData> nearest_neighbor = knn.getKNearestNeighbors(kd, kernel, param);
			int prediction = knn.determineClass(nearest_neighbor);
			if(prediction == kd_classifier){
				correctPredictions++;
			}
		}
		double accuracy = correctPredictions / testSize;
		return accuracy;
	}	
	
	public static void main(String[] args) throws IOException {
		double a = testPrimalPerceptron("data/BTrain.arff", "data/BTest.arff");
		double b = testDualKnn("data/BTrain.arff", "data/BTest.arff", 1, 0.1);
		double c = testDualKnn("data/BTrain.arff", "data/BTest.arff", 1, 0.5);
		double d = testDualKnn("data/BTrain.arff", "data/BTest.arff", 1, 1);
		/*double c = testDualPerceptron("/ATrain.arff", "ATest.arff", 0, 2);
		double d = testDualPerceptron("ATrain.arff", "ATest.arff", 0, 3);
		double e = testPrimalKnn("ATrain.arff", "ATest.arff");*/
		System.out.println(b);
		System.out.println(c);

		System.out.println(d);
		
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
