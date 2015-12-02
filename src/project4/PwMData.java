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
		this.num_attributes = data.size();
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



	@Override
	public String toString() {
		return "PwMData [data=" + data + ", classifier=" + classifier + ", distance=" + distance + ", num_attributes="
				+ num_attributes + "]";
	}

	// poly kernel
	public static double polyKernel(PwMData p1, PwMData p2, double d){
		List<Double> u = p1.getData();
		List<Double> v = p2.getData();
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
	public static double rbfKernel(PwMData p1, PwMData p2, double s){
		List<Double> u = p1.getData();
		List<Double> v = p2.getData();
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
	
	// primal distance
	public static double primalDist(PwMData p1, PwMData p2){
		List<Double> p1_data = p1.getData();
		List<Double> p2_data = p2.getData();
		double distance = 0;
		if(p1_data.size() == p2_data.size()){
			for (int i = 0; i < p1_data.size(); i++){
				double diff = p1_data.get(i) - p2_data.get(i);
				distance += Math.pow(diff, 2);
			}
			distance = Math.sqrt(distance);
		}
		return distance;
	}
	
	// kernel distance
	// 0 -> polykernel; 1 -> rbfkernel
	public static double kernelDist(PwMData p1, PwMData p2, int kernel, double param){
		double distance = 0;
		double kernelResult = 0;
		if(kernel == 0){
			kernelResult = polyKernel(p1, p1, param) + polyKernel(p2, p2, param) - 2 * polyKernel(p1, p2, param);
			distance = Math.sqrt(kernelResult);
		}else if(kernel == 1){
			kernelResult = rbfKernel(p1, p1, param) + rbfKernel(p2, p2, param) - 2 * rbfKernel(p1, p2, param);
			distance = Math.sqrt(kernelResult);
		}else{
			throw new IllegalArgumentException("No such kernel");
		}
		return distance;
	}
}
