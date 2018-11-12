package cs107KNN;

//import com.sun.tools.corba.se.idl.toJavaPortable.Helper;

import java.util.*;

public class KMeansClustering {
	public static void main(String[] args) {
		int K = 5000;
		int maxIters = 20;

		byte[][][] images = KNN.parseIDXimages(Helpers.readBinaryFile("datasets/100-per-digit_images_train"));
		byte[] labels = KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/100-per-digit_labels_train"));

		byte[][][] reducedImages = KMeansReduce(images, K, maxIters);
//
//		byte[] reducedLabels = new byte[reducedImages.length];
//		for (int i = 0; i < reducedLabels.length; i++) {
//			reducedLabels[i] = KNN.knnClassify(reducedImages[i], images, labels, 5);
//			System.out.println("Classified " + (i + 1) + " / " + reducedImages.length);
//		}
//
//		Helpers.writeBinaryFile("datasets/reduced10Kto1K_images", encodeIDXimages(reducedImages));
//		Helpers.writeBinaryFile("datasets/reduced10Kto1K_labels", encodeIDXlabels(reducedLabels));
	}

    /**
     * @brief Encodes a tensor of images into an array of data ready to be written on a file
     * 
     * @param images the tensor of image to encode
     * 
     * @return the array of byte ready to be written to an IDX file
     */
	public static byte[] encodeIDXimages(byte[][][] images) {

		//TESTEE ET VALIDEE
		//Test : diff testEncodeIDXImages 10-per-digit_images_train (commande Terminal)
		//si retour sans message : VALIDE
		//sinon message "Binary files [file1] and [file2] differ"

		//afficher un fichier en binaire: xxd -b [filename]

		//images structure:
		// 0-3 : magic number 2051
		// 4-7 : number of images
		// 8-11 : height of images
		// 12-15 : width of images
		// 16-... : image by image, line by line, column by column

		assert images != null;

		byte[] encodedImages = new byte[(images.length*images[0].length*images[0][0].length)+16];
		int currentByte = 16;

		encodeInt(2051, encodedImages, 0);
		encodeInt(images.length, encodedImages, 4);
		encodeInt(images[0].length, encodedImages, 8);
		encodeInt(images[0][0].length, encodedImages, 12);

		for (int k = 0; k < images.length; ++k) {
			for (int j = 0; j < images[0].length; ++j) {
				for (int l = 0; l < images[0][0].length; ++l) {
					//tabImages[k][j][l] = signedImages[k*largeurImages*hauteurImages+j*largeurImages+l];
					byte unsignedByte = images[k][j][l];
					encodedImages[16+(k*images[0].length*images[0][0].length+j*images[0][0].length+l)] = (byte) ((unsignedByte & 0xFF) - 128);
					//System.out.println(k*images[0].length*images[0][0].length+j*images[0][0].length+l);
				}
			}
		}

		Helpers.writeBinaryFile("datasets/encodedIDXImages", encodedImages);

		return encodedImages;
	}

    /**
     * @brief Prepares the array of labels to be written on a binary file
     * 
     * @param labels the array of labels to encode
     * 
     * @return the array of bytes ready to be written to an IDX file
     */
	public static byte[] encodeIDXlabels(byte[] labels) {

		//TESTEE ET VALIDEE
		//Test : diff testEncodeIDXLabels 10-per-digit_labels_train (commande Terminal)
		//si retour sans message : VALIDE
		//sinon message "Binary files [file1] and [file2] differ"

		//afficher un fichier en binaire: xxd -b [filename]

		//labels structure :
		// 0-3 : magic number 2049
		// 4-7 : number of labels
		// 8-... : labels

		assert labels != null;

		byte[] encodedLabels = new byte[labels.length + 8];

		encodeInt(2049,encodedLabels,0);
		encodeInt(labels.length, encodedLabels, 4);

		for (int i=8; i<labels.length+8; ++i) {
			byte unsignedByte = labels[i-8];
			byte signedByte = (byte) ((unsignedByte & 0xFF));
			encodedLabels[i] = signedByte;
		}

		Helpers.writeBinaryFile("datasets/encodedIDXLabels", encodedLabels);

		return encodedLabels;
	}

    /**
     * @brief Decomposes an integer into 4 bytes stored consecutively in the destination
     * array starting at position offset
     * 
     * @param n the integer number to encode
     * @param destination the array where to write the encoded int
     * @param offset the position where to store the most significant byte of the integer,
     * the others will follow at offset + 1, offset + 2, offset + 3
     */
	public static void encodeInt(int n, byte[] destination, int offset) {

		assert destination != null;

		String integerByte = Integer.toBinaryString(n);
		if(integerByte.length()!=32){
			int nbAAjouter = 32 - integerByte.length();
			for (int i = 0; i<nbAAjouter; ++i){
				integerByte ='0' + integerByte;
			}
		}

		for (int j = 0; j<4;++j) {
			String byte8 = "";

			for (int i = 0; i < 8; ++i) {
				byte8 = byte8 + integerByte.charAt(j * 8 + i);
			}

			destination[offset+j] = Helpers.binaryStringToByte(byte8);

		}

	}

    /**
     * @brief Runs the KMeans algorithm on the provided tensor to return size elements.
     * 
     * @param tensor the tensor of images to reduce
     * @param size the number of images in the reduced dataset
     * @param maxIters the number of iterations of the KMeans algorithm to perform
     * 
     * @return the tensor containing the reduced dataset
     */
	public static byte[][][] KMeansReduce(byte[][][] tensor, int size, int maxIters) {

		assert tensor != null;

		System.out.println("Starting KMeansReduce");

		int[] assignments = new Random().ints(tensor.length, 0, size).toArray();
		byte[][][] centroids = new byte[size][][];
		initialize(tensor, assignments, centroids);

		int nIter = 0;
		while (nIter < maxIters) {

			System.out.println("Itering - Current : " + nIter);
			// Step 1: Assign points to closest centroid
			recomputeAssignments(tensor, centroids, assignments);
			System.out.println("Recomputed assignments");
			// Step 2: Recompute centroids as average of points
			//recomputeCentroids(tensor, centroids, assignments);
			//System.out.println("Recomputed centroids");

			System.out.println("KMeans completed iteration " + (nIter + 1) + " / " + maxIters);

			nIter++;
		}

		return centroids;
	}

   /**
     * @brief Assigns each image to the cluster whose centroid is the closest.
     * It modifies.
     * 
     * @param tensor the tensor of images to cluster
     * @param centroids the tensor of centroids that represent the cluster of images
     * @param assignments the vector indicating to what cluster each image belongs to.
     *  if j is at position i, then image i belongs to cluster j
     */
	public static void recomputeAssignments(byte[][][] tensor, byte[][][] centroids, int[] assignments) {

		assert tensor != null;
		assert centroids != null;
		assert assignments != null;

		//chaque image appartient au cluster le plus proche (euclidean distance)

		for (int i=0; i<tensor.length; ++i) {
			byte[][] currentImage = tensor[i];

			float smallestDistance = KNN.squaredEuclideanDistance(currentImage, centroids[0]);
			int currentCentroid = 0;
			//Compare current image to all the centroids
			for (int j=1; j<centroids.length; ++j) {

				float distance = KNN.squaredEuclideanDistance(currentImage, centroids[j]);
				if (distance < smallestDistance) {
					//We found a smaller Euclidean distance
					smallestDistance = distance;
					currentCentroid = j;
				}

				assignments[i] = currentCentroid;

			}

		}

	}

    /**
     * @brief Computes the centroid of each cluster by averaging the images in the cluster
     * 
     * @param tensor the tensor of images to cluster
     * @param centroids the tensor of centroids that represent the cluster of images
     * @param assignments the vector indicating to what cluster each image belongs to.
     *  if j is at position i, then image i belongs to cluster j
     */
	public static void recomputeCentroids(byte[][][] tensor, byte[][][] centroids, int[] assignments) {
	}

    /**
     * Initializes the centroids and assignments for the algorithm.
     * The assignments are initialized randomly and the centroids
     * are initialized by randomly choosing images in the tensor.
     * 
     * @param tensor the tensor of images to cluster
     * @param assignments the vector indicating to what cluster each image belongs to.
     * @param centroids the tensor of centroids that represent the cluster of images
     *  if j is at position i, then image i belongs to cluster j
     */
	public static void initialize(byte[][][] tensor, int[] assignments, byte[][][] centroids) {

		assert tensor != null;
		assert assignments != null;
		assert centroids != null;

		System.out.println("Initialising...");

		Set<Integer> centroidIds = new HashSet<>();
		Random r = new Random("cs107-2018".hashCode());
		while (centroidIds.size() != centroids.length)
			centroidIds.add(r.nextInt(tensor.length));
		Integer[] cids = centroidIds.toArray(new Integer[] {});
		for (int i = 0; i < centroids.length; i++)
			centroids[i] = tensor[cids[i]];
		for (int i = 0; i < assignments.length; i++)
			assignments[i] = cids[r.nextInt(cids.length)];

		System.out.println("Initialised.");
	}
}
