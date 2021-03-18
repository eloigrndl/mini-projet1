package cs107KNN;

import java.util.Arrays;

public class KNNTest {
	public static void main(String[] args) {
		// TODO: Adapt path to data files in parsing test
		// Uncomment wanted tests
		extractIntTest();
		parsingTest();
		squaredEuclideanDistanceTest();
		invertedSimilarityTest();
		quicksortTest();
		indexOfMaxTest();
		electLabelTest();
		knnClassifyTest();
		accuracyTest();
	}

	public static void extractIntTest() {
		byte b1 = 40; // 00101000
		byte b2 = 120; // 00010100
		byte b3 = 70; // 00001010
		byte b4 = -117; // 00000101

		String expected = Helpers.byteToBinaryString(b1) +
			Helpers.byteToBinaryString(b2) +
			Helpers.byteToBinaryString(b3) +
			Helpers.byteToBinaryString(b4);

		int obtained = KNN.extractInt(b1, b2, b3, b4);

		System.out.println("=== Test extractInt ===");
		System.out.println("Expected:\t " + expected);
		System.out.println("Result  :\t " + "00101000011110000100011010001011");
	}

	public static void parsingTest() {
		System.out.println("=== Test parsing ===");
		byte[][][] images = KNN.parseIDXImages(Helpers.readBinaryFile("datasets/10-per-digit_images_train"));
		byte[] labels = KNN.parseIDXLabels(Helpers.readBinaryFile("datasets/10-per-digit_labels_train"));

		assert(images != null && labels != null);

		System.out.println("Number of images: " + images.length);
		System.out.println("Height: " + images[0].length);
		System.out.println("Width: " + images[0][0].length);

		Helpers.show("Test parsing", images, labels, 10, 10);
	}


	public static void squaredEuclideanDistanceTest() {
		System.out.println("=== Test euclidean distance ===");
		byte[][] a = new byte[][] {{1, 1}, {2, 2}};
		byte[][] b = new byte[][] {{3, 3}, {4, 4}};

		System.out.println("Expected: " + KNN.squaredEuclideanDistance(a, b));
		System.out.println("Result  : 16.0");
	}

	public static void invertedSimilarityTest() {
		System.out.println("=== Test Inverted Similarity ===");
		byte[][] a = new byte[][] {{1, 1}, {1, 2}};
		byte[][] b = new byte[][] {{50, 50}, {50, 100}};

		System.out.println("Expected: " + KNN.invertedSimilarity(a, b));
		System.out.println("Result  : 0.0");
	}

	public static void quicksortTest() {
		System.out.println("=== Test quicksort ===");
		float[] data = new float[] {3, 7, 0, 9};
		int[] result = KNN.quicksortIndices(data);

		System.out.println("Sorted indices: " + Arrays.toString(result));
	}

	public static void indexOfMaxTest() {
		System.out.println("=== Test indexOfMax ===");
		int[] data = new int[]{0, 5, 9, 1};

		int indexOfMax = KNN.indexOfMax(data);
		System.out.println("Indices: [0, 1, 2, 3]");
		System.out.println("Data: " + Arrays.toString(data));
		System.out.println("Max element index: " + indexOfMax);
	}


	public static void electLabelTest() {
		System.out.println("=== Test electLabel ===");
		int[] sortedIndices = new int[]{0, 3, 2, 1};
		byte[] labels = new byte[]{2, 1, 1, 2};
		int k = 3;

		System.out.println("Elected label : " + KNN.electLabel(sortedIndices, labels, k));
		System.out.println("Expected label: 2");
	}

	public static void knnClassifyTest() {
		System.out.println("=== Test predictions ===");
		byte[][][] imagesTrain = KNN.parseIDXImages(Helpers.readBinaryFile("datasets/10-per-digit_images_train"));
		byte[] labelsTrain = KNN.parseIDXLabels(Helpers.readBinaryFile("datasets/10-per-digit_labels_train"));

		byte[][][] imagesTest = KNN.parseIDXImages(Helpers.readBinaryFile("datasets/10k_images_test"));
		byte[] labelsTest = KNN.parseIDXLabels(Helpers.readBinaryFile("datasets/10k_labels_test"));

		assert imagesTrain != null && labelsTrain != null && imagesTest != null && labelsTest != null;

		byte[] predictions = new byte[60];
		for (int i = 0; i < 60; i++) {
			predictions[i] = KNN.knnClassify(imagesTest[i], imagesTrain, labelsTrain, 7);
		}
		Helpers.show("Test predictions", imagesTest, predictions, labelsTest, 10, 6);
	}


	public static void accuracyTest() {
		System.out.println("=== Test precision ===");
		byte[] a = new byte[] {1, 1, 1, 1};
		byte[] b = new byte[] {1, 1, 1, 9};

		System.out.println("Expected: " + KNN.accuracy(a, b));
		System.out.println("Result  : 0.75");
	}
}

