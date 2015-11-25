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
	private static double tau = Double.NEGATIVE_INFINITY;
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
	
	public static void calculateTau(List<PwMData> data){
		double norm  = 0;
		for(PwMData d : data){
			for(Double x : d.getData()){
				norm += Math.pow(x, 2);
			}
		}
		norm = Math.sqrt(norm);
		tau = 0.1*(norm/data.size());
	}
	
	public static double[] primalPwM(List<PwMData> data){
		for(PwMData d : data){
			for(Double x : d.getData()){
				
			}
		}
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
