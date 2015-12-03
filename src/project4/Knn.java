package project4;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
/**
 * Implement KNN Method
 * @author Zhaokun Xue
 *
 */
public class Knn {
	private List<PwMData> train_data_set;
	private List<PwMData> test_data_set;
	private int k;
	
	/**
	 * constructor for KNN
	 * @param train_data_set train data set
	 * @param test_data_set	test data set
	 * @param k k value
	 */
	public Knn(List<PwMData> train_data_set, List<PwMData> test_data_set) {
		this.train_data_set = train_data_set;
		this.test_data_set = test_data_set;
		this.k = 1;
	}

	
	/**
	 * get the k nearest neighbors for test example
	 * @param test the test example
	 * @return a list of PwMData contains k nearest neighbors
	 */
	public List<PwMData> getKNearestNeighbors(PwMData test){
		Comparator<PwMData> comparator = new DistanceComparator();
		int initial_size = train_data_set.size();
	    PriorityQueue<PwMData> queue = new PriorityQueue<PwMData>(initial_size, comparator);
		List<PwMData> k_nearest_neighbors = new ArrayList<PwMData>();
		for(int i = 0; i < train_data_set.size(); i++){
			PwMData current_train_data = train_data_set.get(i);
			double dis = PwMData.primalDist(test, current_train_data);
			current_train_data.setDistance(dis);
			queue.add(current_train_data);
		}
		int i = 0;
		while(i < this.getK()){
			k_nearest_neighbors.add(queue.remove());
			i++;
		}
		return k_nearest_neighbors;
	}

	/**
	 * find the knn using kernels
	 * @param test test data
	 * @param kernel kernel option
	 * @param param kernel param
	 * @return nearest neighbor
	 */
	public List<PwMData> getKNearestNeighbors(PwMData test, int kernel, double param){
		Comparator<PwMData> comparator = new DistanceComparator();
		int initial_size = train_data_set.size();
	    PriorityQueue<PwMData> queue = new PriorityQueue<PwMData>(initial_size, comparator);
		List<PwMData> k_nearest_neighbors = new ArrayList<PwMData>();
		for(int i = 0; i < train_data_set.size(); i++){
			PwMData current_train_data = train_data_set.get(i);
			double dis = PwMData.kernelDist(test, current_train_data, kernel, param);
			current_train_data.setDistance(dis);
			queue.add(current_train_data);
		}
		int i = 0;
		while(i < this.getK()){
			k_nearest_neighbors.add(queue.remove());
			i++;
		}
		return k_nearest_neighbors;
	}
	
	/**
	 * determine the class based on the test data's k nearest neighbors
	 * @param k_nearest_neighbors the test data's k nearest neighbors
	 * @return 0 or 1 for the class
	 */
	public int determineClass(List<PwMData> k_nearest_neighbors){
		int counter0 = 0;
		int counter1 = 0;
		for(PwMData kd : k_nearest_neighbors){
			int kd_class = kd.getClassifier();
			if(kd_class == -1){
				counter0++;
			}
			if(kd_class == 1){
				counter1++;
			}
		}
		if(counter0 >= counter1){
			return -1;
		}else{
			return 1;
		}
	}
	
	// getters and setters
	public List<PwMData> getTrain_data_set() {
		return train_data_set;
	}
	public void setTrain_data_set(List<PwMData> train_data_set) {
		this.train_data_set = train_data_set;
	}
	public List<PwMData> getTest_data_set() {
		return test_data_set;
	}
	public void setTest_data_set(List<PwMData> test_data_set) {
		this.test_data_set = test_data_set;
	}
	public int getK() {
		return k;
	}
	public void setK(int k) {
		this.k = k;
	}
}
