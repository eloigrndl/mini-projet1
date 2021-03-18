package cs107KNN;

//import com.sun.tools.corba.se.idl.toJavaPortable.Helper;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane;

import java.util.Arrays;

public class KNN {
	public static void main(String[] args) {

		int TESTS = 1000;
		int K = 7;

		byte[][][] trainImages = parseIDXImages(Helpers.readBinaryFile("datasets/100-per-digit_images_train"));
		byte[] trainLabels = parseIDXLabels(Helpers.readBinaryFile("datasets/100-per-digit_labels_train"));

		byte[][][] testImages = parseIDXImages(Helpers.readBinaryFile("datasets/10k_images_test"));
		byte[] testLabels = parseIDXLabels(Helpers.readBinaryFile("datasets/10k_labels_test"));

		assert testImages != null && trainImages != null && testLabels != null;

		byte[] predictions = new byte[TESTS];
		long start = System.currentTimeMillis();
		for (int i = 0; i < TESTS; i++) {
			predictions[i] = knnClassify(testImages[i], trainImages, trainLabels, K);
		}
		long end = System.currentTimeMillis();
		double time = (end - start) / 1000d;
		System.out.println("Accuracy = " + accuracy(predictions, Arrays.copyOfRange(testLabels, 0, TESTS))*100 + " %");
		System.out.println("Time = " + time + " seconds");
		System.out.println("Time per test image = " + (time / TESTS));


		Helpers.show("Test", testImages, predictions, testLabels, 10, 10) ;
	}

	/**
	 * Composes four bytes into an integer using big endian convention.
	 * @param "bXToBY" The byte containing the bits to store between positions X and Y
	 * @return the integer having form [ b31ToB24 | b23ToB16 | b15ToB8 | b7ToB0 ]
	 */
	public static int extractInt(byte b31ToB24, byte b23ToB16, byte b15ToB8, byte b7ToB0) {

		String bits = Helpers.byteToBinaryString(b31ToB24) + Helpers.byteToBinaryString(b23ToB16) + Helpers.byteToBinaryString(b15ToB8) + Helpers.byteToBinaryString(b7ToB0);
		
		int sum = 0;
		for (int i= bits.length()-1; i>=0;--i){
			int power = bits.length()-i-1;
			int binaryVal = (int)bits.charAt(i) - 48;
			sum+= binaryVal * (Math.pow(2, power));

		}
		return sum;
	}

	/**
	 * Parses an IDX file containing images
	 * @param data the binary content of the file
	 * @return A tensor of images
	 */
	public static byte[][][] parseIDXImages(byte[] data) {

		assert data != null;
		int nbMagic = extractInt(data[0], data[1], data[2], data[3]);
		if (nbMagic != 2051) {
			return null;
		}

		//Get the number of images as well as their height and width
		int nbImages = extractInt(data[4], data[5], data[6], data[7]);
		int imageHeight = extractInt(data[8], data[9], data[10], data[11]);
		int imageWidth = extractInt(data[12], data[13], data[14], data[15]);

		//Convert unsigned bytes to signed bytes
		byte[][][] imageTab = new byte[nbImages][imageHeight][imageWidth];
		int pxlNumber = nbImages * imageHeight * imageWidth;
		byte []signedImages = new byte[pxlNumber];
		for (int i = 16; i < 16 + pxlNumber; ++i) {
			byte unsignedImage = data[i];
			byte signedImage = (byte) ((unsignedImage & 0xFF) - 128);
			signedImages[i-16] = signedImage;
		}

		//Put signed bytes in image tensor
		for (int k = 0; k < nbImages; ++k) {
			for (int j = 0; j < imageHeight; ++j) {
				for (int l = 0; l < imageWidth; ++l) {
					imageTab[k][j][l] = signedImages[k * imageWidth * imageHeight + j * imageWidth + l];
				}
			}
		}
		return imageTab;
	}

	/**
	 * Parses an idx images containing labels
	 * @param data the binary content of the file
	 * @return the parsed labels
	 */
	public static byte[] parseIDXLabels(byte[] data) {

		assert data != null;
		int magicNumber = extractInt(data[0], data[1], data[2], data[3]);
		if (magicNumber != 2049) {
		    return null;
        }

        int labelNumber = extractInt(data[4], data[5], data[6], data[7]);
		byte[] labels = new byte[labelNumber];

        for (int i = 8; i < 8 + labelNumber; ++i) {
            byte unsignedLabel = data[i];
            byte intLabel = (byte) ((unsignedLabel & 0xFF));

			labels[i-8] = intLabel;
        }
		return labels;
	}

	/**
	 * Computes the squared L2 distance of two images
	 * @param a, b two images of same dimensions
	 * @return the squared euclidean distance between the two images
	 */
	public static float squaredEuclideanDistance(byte[][] a, byte[][] b) {

		assert a != null;
		assert b != null;

		int height = a.length;
		int width = a[0].length;

		float e = 0;

		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {

				float aij = a[i][j];
				float bij = b[i][j];

				e += Math.pow((aij - bij), 2);
			}
		}
		return e;
	}

	/**
	 * Computes the inverted similarity between 2 images.
	 * @param a, b two images of same dimensions

	 * @return the inverted similarity between the two images
	 */
	public static float invertedSimilarity(byte[][] a, byte[][] b) {

		assert a != null;
		assert b!= null;

		int height = a.length;
		int width = a[0].length;

		//Computation of numerator
		float numerator = 0;
		float term1 = 0;
		float term2 = 0;

		double[] averageVal = averageValCalc(a, b);

		for (int i = 0; i < height-1; ++i){
			for (int j = 0; j < width - 1; ++j){
				numerator += (a[i][j] - averageVal[0]) * (b[i][j] - averageVal[1]);
				term1 += (a[i][j] - averageVal[0]) * (a[i][j] - averageVal[0]);
				term2 += (b[i][j] - averageVal[1]) * (b[i][j] - averageVal[1]);
			}
		}

		float denominator = (float) Math.sqrt((term1 * term2));
		if(denominator==0){
			denominator = (float) 2.0;
		}

		//Computation of inverted similarity
		return 1 - (numerator / denominator);
	}

	//Helper method to compute average value
	public static double[] averageValCalc(byte[][] a, byte[][]b){

		assert a != null;
		assert b != null;

		double avgValA = 0;
		double avgValB = 0;
		double  height = a.length;
		double width = a[0].length;
		for(int i =0; i<height-1;++i){
			for(int j = 0; j<width-1;++j){
				avgValA +=a[i][j];
				avgValB +=b[i][j];
			}
		}
		avgValA = avgValA / (height * width);
		avgValB = avgValB / (height * width);


		return new double[] {avgValA, avgValB};
	}

	/**
	 * Quick-sorts and returns the new indices of each value.
	 * @param values the values whose indices have to be sorted in non decreasing order
	 * @return the array of sorted indices
	 *         Example: values = quicksortIndices([3, 7, 0, 9]) gives [2, 0, 1, 3]
	 */
	public static int[] quicksortIndices(float[] values) {
		//ex: values = {3, 7, 0, 9};
		//indices = {0, 1, 2, 3};

		assert values != null;

		int[] indices = new int[values.length];
		for (int i = 0; i < values.length; ++i) {
			indices[i] = i;
		}

		quicksortIndices(values, indices, 0, (values.length-1));
		return indices;
	}

	/**
	 * Sorts the provided values between two indices while applying the same
	 *        transformations to the array of indices
	 * @param values  the values to sort
	 * @param indices the indices to sort according to the corresponding values
	 * @param         low, high are the **inclusive** bounds of the portion of array
	 *                to sort
	 */
	public static void quicksortIndices(float[] values, int[] indices, int low, int high) {

		assert values != null;
		assert indices != null;

		int l = low;
		int h = high;
		float pivot = values[low];

		while (l <= h) {
			if (values[l] < pivot) {
				++l;
			} else if (values[h] > pivot) {
				--h;
			} else {
				swap(l, h, values, indices);
				++l;
				--h;
			}
		}

		if (low < h) {
			quicksortIndices(values, indices, low, h);
		}

		if (high > l) {
			quicksortIndices(values, indices, l, high);
		}

	}

	/**
	 * Swaps the elements of the given arrays at the provided positions
	 * @param         i, j the indices of the elements to swap
	 * @param values  the array floats whose values are to be swapped
	 * @param indices the array of ints whose values are to be swapped
	 */
	public static void swap(int i, int j, float[] values, int[] indices) {

		//values[i][j] => values[j][i]
		//indices[i][j] => indices[j][i]

		assert values != null;
		assert indices != null;

		float valuesTemp = values[i];
		values[i] = values[j];
		values[j] = valuesTemp;

		int indicesTemp = indices[i];
		indices[i] = indices[j];
		indices[j] = indicesTemp;

	}

	/**
	 * Returns the index of the largest element in the array
	 * @param array an array of integers
	 * @return the index of the largest integer
	 */
	public static int indexOfMax(int[] array) {

		assert array != null;

	    int k = 0;
		for(int i = 0; i < array.length;++i){
		    if(k<array[i]){
		        k = i;
            }
        }
		return k;
	}

	/**
	 * The k first elements of the provided array vote for a label
	 * @param sortedIndices the indices sorted by non-decreasing distance
	 * @param labels        the labels corresponding to the indices
	 * @param k             the number of labels asked to vote
	 * @return the winner of the election
	 */
	public static byte electLabel(int[] sortedIndices, byte[] labels, int k) {

		assert sortedIndices != null;
		assert labels != null;

		//Verification of the bounds of the array
        if(k > sortedIndices.length){
            throw new IllegalArgumentException();
        }

	    int[] tab = new int[10];
        for(int i = 0;i<k;++i){
            int j = labels[sortedIndices[i]];
            tab[j]+=1;
        }
        return (byte) indexOfMax(tab);
	}

	/**
	 * Classifies the symbol drawn on the provided image
	 * @param image       the image to classify
	 * @param trainImages the tensor of training images
	 * @param trainLabels the list of labels corresponding to the training images
	 * @param k           the number of voters in the election process
	 * @return the label of the image
	 */
	public static byte knnClassify(byte[][] image, byte[][][] trainImages, byte[] trainLabels, int k) {

		assert image != null;
		assert trainImages != null;
		assert trainLabels != null;

		float[] distances = new float[trainImages.length];

		for (int i = 0; i < trainImages.length; ++i) {
			distances[i] = invertedSimilarity(image, trainImages[i]);
		}
		return electLabel(quicksortIndices(distances), trainLabels, k);
	}

	/**
	 * Computes accuracy between two arrays of predictions
	 * @param predictedLabels the array of labels predicted by the algorithm
	 * @param trueLabels      the array of true labels
	 * @return the accuracy of the predictions. Its value is in [0, 1]
	 */
	public static double accuracy(byte[] predictedLabels, byte[] trueLabels) {

		assert predictedLabels != null;
		assert trueLabels != null;

		//
		if(predictedLabels.length != trueLabels.length || predictedLabels.length == 0){
            throw new IllegalArgumentException();
        }

		double accuracy = 0;
		for(int i = 0; i<trueLabels.length;++i){
			if(predictedLabels[i]==trueLabels[i]){
				accuracy+=1;
			}
		}
		accuracy/=trueLabels.length;
		return accuracy;
	}
}
