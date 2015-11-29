package project4;

import java.util.List;
/**
 * KNN data for each record of data
 * @author Zhaokun Xue
 *
 */
public class PwMData {
	private List<Double> data;
	private int classifier;
	private double distance; // the distance from test PwM Data 
	private int num_attributes;


	/**
	 * constructor for PwMData
	 * @param data the record's data
	 * @param classifier the class of the data record
	 */
	public PwMData(List<Double> data, int classifier) {
		this.data = data;
		this.classifier = classifier;
		this.distance = Double.POSITIVE_INFINITY;
		this.num_attributes = data.size()-1;
	}
	
	// getters and setters
	public List<Double> getData() {
		return data;
	}
	public void setData(List<Double> data) {
		this.data = data;
	}
	
	public int getClassifier() {
		return classifier;
	}
	public void setClassifier(int classifier) {
		this.classifier = classifier;
	}
	public double getDistance() {
		return distance;
	}
	public void setDistance(double distance) {
		this.distance = distance;
	}
	
	public int getNum_attributes() {
		return num_attributes;
	}

	public void setNum_attributes(int num_attributes) {
		this.num_attributes = num_attributes;
	}
	
}
