package project4;

import java.util.Comparator;
/**
 * comparator for helping compare the distances among KNN Data
 * @author Zhaokun Xue
 *
 */
public class DistanceComparator implements Comparator<PwMData> {

	@Override
	public int compare(PwMData o1, PwMData o2) {
        if (o1.getDistance() < o2.getDistance())
        {
            return -1;
        }
        if (o1.getDistance() > o2.getDistance())
        {
            return 1;
        }
        return 0;
	}

}
