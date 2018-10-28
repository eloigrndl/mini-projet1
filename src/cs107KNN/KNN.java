package cs107KNN;

import com.sun.tools.corba.se.idl.toJavaPortable.Helper;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane;

public class KNN {
	public static void main(String[] args) {
		//byte b1 = 40; // 00101000
		//byte b2 = 20; // 00010100
		//byte b3 = 10; // 00001010
		//byte b4 = 5; // 00000101

		// [00101000 | 00010100 | 00001010 | 00000101] = 672401925
		//int result = extractInt(b1, b2, b3, b4);
		//System.out.println(result);
        KNNTest.electLabelTest();
		//Exemple de lecture du dataset IDX
		// Charge les étiquettes depuis le disque
		byte[] labelsRaw = Helpers.readBinaryFile("datasets/10-per-digit_labels_train");
		// Parse les étiquettes
		byte[] labelsTrain = parseIDXlabels(labelsRaw);
		// Affiche le nombre de labels
		System.out.println("Number of labels : " + labelsTrain.length);
		// Affiche le premier label
		System.out.println("First label : " +labelsTrain[0]);

		// Charge les images depuis le disque
		byte[] imagesRaw = Helpers.readBinaryFile("datasets/10-per-digit_images_train");
		// Parse les images
		byte[][][] imagesTrain = parseIDXimages(imagesRaw);
		// Affiche les dimensions des images
		System.out.println("Number of images : " + imagesTrain.length); System.out.println("Height : " + imagesTrain[0].length); System.out.println("Width : " + imagesTrain[0][0].length);
		// Affiche les 30 premières images et leurs étiquettes
		Helpers.show("Test", imagesTrain, labelsTrain, 2, 15);
	}

	/**
	 * Composes four bytes into an integer using big endian convention.
	 *
	 * @param "bXToBY" The byte containing the bits to store between positions X and Y
	 *
	 * @return the integer having form [ b31ToB24 | b23ToB16 | b15ToB8 | b7ToB0 ]
	 */
	public static int extractInt(byte b31ToB24, byte b23ToB16, byte b15ToB8, byte b7ToB0) {

		String bits = Helpers.byteToBinaryString(b31ToB24) + Helpers.byteToBinaryString(b23ToB16) + Helpers.byteToBinaryString(b15ToB8) + Helpers.byteToBinaryString(b7ToB0);
		
		int sum = 0;
		for (int i= bits.length()-1; i>=0;--i){
			int puissance = bits.length()-i-1;
			int valBinaire = (int)bits.charAt(i) - 48;
			sum+=valBinaire*(Math.pow(2,puissance));

		}
		return sum;
	}

	/**
	 * Parses an IDX file containing images
	 *
	 * @param data the binary content of the file
	 *
	 * @return A tensor of images
	 */
	public static byte[][][] parseIDXimages(byte[] data) {

		int nbMagic = extractInt(data[0], data[1], data[2], data[3]);
		if (nbMagic != 2051) {
			return null;
		}
		//récupère nombre d'images/hauteur/largeur
		int nbImages = extractInt(data[4], data[5], data[6], data[7]);
		int hauteurImages = extractInt(data[8], data[9], data[10], data[11]);
		int largeurImages = extractInt(data[12], data[13], data[14], data[15]);


		//passe de bytes non-signés à signés
		byte[][][] tabImages = new byte[nbImages][hauteurImages][largeurImages];
		int nbpixels = nbImages * hauteurImages * largeurImages;
		byte []signedImages = new byte[nbpixels];
		for (int i = 16; i <16+nbpixels; ++i) {
			byte unsignedImage = data[i];
			byte signedImage = (byte) ((unsignedImage & 0xFF) - 128);
			signedImages[i-16] = signedImage;
		}

		//transfère bytes signés dans tensor d'images
		for (int k = 0; k < nbImages; ++k) {
			for (int j = 0; j < hauteurImages; ++j) {
				for (int l = 0; l < largeurImages; ++l) {
					tabImages[k][j][l] = signedImages[k*largeurImages*hauteurImages+j*largeurImages+l];
				}
			}
		}
		return tabImages;
	}

	/**
	 * Parses an idx images containing labels
	 *
	 * @param data the binary content of the file
	 *
	 * @return the parsed labels
	 */
	public static byte[] parseIDXlabels(byte[] data) {
		int nombreMagique = extractInt(data[0], data[1], data[2], data[3]);

		if (nombreMagique != 2049) {
		    return null;
        }

        int nombreEtiquettes = extractInt(data[4], data[5], data[6], data[7]);

		byte[] etiquettes = new byte[nombreEtiquettes];
        for (int i=8; i<8+nombreEtiquettes; ++i) {
            byte unsignedEtiquette = data[i];
            byte intEtiquette = (byte) ((unsignedEtiquette & 0xFF));

            etiquettes[i-8] = intEtiquette;
        }

		return etiquettes;
	}

	/**
	 * @brief Computes the squared L2 distance of two images
	 *
	 * @param a, b two images of same dimensions
	 *
	 * @return the squared euclidean distance between the two images
	 */
	public static float squaredEuclideanDistance(byte[][] a, byte[][] b) {

		//On vérifie qu'on a des tenseurs de dimensions équivalentes
		if (a.length != b.length && a[0].length != b[0].length) {
			return 0;
		}

		int hauteur = a.length; //nombre de lignes
		int largeur = a[0].length; //nombre de colonnes

		float e = 0;

		for (int i=0; i<hauteur; ++i) {
			for (int j=0; j<largeur; ++j) {

				float aij = a[i][j];
				float bij = b[i][j];

				e += Math.pow((aij - bij), 2);
			}
		}

		return e;
	}

	/**
	 * @brief Computes the inverted similarity between 2 images.
	 *
	 * @param a, b two images of same dimensions
	 *
	 * @return the inverted similarity between the two images
	 */
	public static float invertedSimilarity(byte[][] a, byte[][] b) {
		//On vérifie qu'on a des tenseurs équivalents
		if (a.length != b.length && a[0].length != b[0].length) {
			return 0;
		}

		int hauteur = a.length;
		int largeur = a[0].length;

		//Calcul du numérateur
		float numerateur = 0;
		for (int i = 0; i < hauteur-1; ++i){
			for (int j = 0; j < largeur - 1; ++j){
				numerateur += (a[i][j] - calculValeurMoyenne(a))*(b[i][j] - calculValeurMoyenne(b));
			}
		}
		//Calcul du premier membre du dénominateur
		float membre1 = 0;
		for (int i =0;i<hauteur-1;++i){
			for (int j = 0;j<largeur - 1;++j){
				membre1 += (a[i][j] - calculValeurMoyenne(a))*(a[i][j] - calculValeurMoyenne(a));
			}
		}
		//Calcul du deuxième membre du dénominateur

		float membre2 = 0;
		for (int i =0;i<hauteur-1;++i){
			for (int j = 0;j<largeur - 1;++j){
				membre2 += (b[i][j] - calculValeurMoyenne(b))*(b[i][j] - calculValeurMoyenne(b));
			}
		}
		float denominateur = (float) Math.sqrt((membre1*membre2));

		if(denominateur==0){
			denominateur = (float) 2.0;
		}

		//Calcul final Similarité inversée
		float SI = 1-(numerateur/denominateur);

		return SI;
	}
	// méthode pour la valeur moyenne
	public static double calculValeurMoyenne(byte[][] a){

		double valMoyenne = 0;
		double  hauteur = a.length;
		double largeur = a[0].length;
		for(int i =0; i<hauteur-1;++i){
			for(int j = 0; j<largeur-1;++j){
				valMoyenne +=a[i][j];
			}
		}
		valMoyenne= valMoyenne/(hauteur*largeur);


		return valMoyenne;
	}

	/**
	 * @brief Quicksorts and returns the new indices of each value.
	 *
	 * @param values the values whose indices have to be sorted in non decreasing
	 *               order
	 *
	 * @return the array of sorted indices
	 *
	 *         Example: values = quicksortIndices([3, 7, 0, 9]) gives [2, 0, 1, 3]
	 */
	public static int[] quicksortIndices(float[] values) {
		//ex: values = {3, 7, 0, 9};
		//indices = {0, 1, 2, 3};

		int[] indices = new int[values.length];

		for (int i=0; i<values.length; ++i) {
			indices[i] = i;
		}

		quicksortIndices(values, indices, 0, (values.length-1));

		return indices;
	}

	/**
	 * @brief Sorts the provided values between two indices while applying the same
	 *        transformations to the array of indices
	 *
	 * @param values  the values to sort
	 * @param indices the indices to sort according to the corresponding values
	 * @param         low, high are the **inclusive** bounds of the portion of array
	 *                to sort
	 */
	public static void quicksortIndices(float[] values, int[] indices, int low, int high) {

		int l = low;
		int h = high;
		int pivot = (int) values[low];

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
	 * @brief Swaps the elements of the given arrays at the provided positions
	 *
	 * @param         i, j the indices of the elements to swap
	 * @param values  the array floats whose values are to be swapped
	 * @param indices the array of ints whose values are to be swapped
	 */
	public static void swap(int i, int j, float[] values, int[] indices) {

		//values[i][j] => values[j][i]
		//indices[i][j] => indices[j][i]

		float valuesTemp = values[i];
		values[i] = values[j];
		values[j] = valuesTemp;

		int indicesTemp = indices[i];
		indices[i] = indices[j];
		indices[j] = indicesTemp;

	}

	/**
	 * @brief Returns the index of the largest element in the array
	 *
	 * @param array an array of integers
	 *
	 * @return the index of the largest integer
	 */
	public static int indexOfMax(int[] array) {
	    int k = 0;
		for(int i = 0; i<array.length;++i){
		    if(k<array[i]){
		        k=i;
            }
        }
		return k;
	}

	/**
	 * The k first elements of the provided array vote for a label
	 *
	 * @param sortedIndices the indices sorted by non-decreasing distance
	 * @param labels        the labels corresponding to the indices
	 * @param k             the number of labels asked to vote
	 *
	 * @return the winner of the election
	 */
	public static byte electLabel(int[] sortedIndices, byte[] labels, int k) {
	    //On vérifie que l'indice k est inférieur ou égal à la taille du tableau d'indices
        if(k>sortedIndices.length){
            return 0; //Est-ce que peut retourner ca si c'est faux ?
        }
        //Processus de votes (j'ai mis un switch : ca fonctionne mais ya mieux ?)

	    int[] tab = new int[10];
        for(int i = 0;i<k;++i){
            int j = labels[sortedIndices[i]];
            tab[j]+=1;
        }
        byte winnerOfElection = (byte) indexOfMax(tab);
        return winnerOfElection;
	}

	/**
	 * Classifies the symbol drawn on the provided image
	 *
	 * @param image       the image to classify
	 * @param trainImages the tensor of training images
	 * @param trainLabels the list of labels corresponding to the training images
	 * @param k           the number of voters in the election process
	 *
	 * @return the label of the image
	 */
	public static byte knnClassify(byte[][] image, byte[][][] trainImages, byte[] trainLabels, int k) {


		return 0;
	}

	/**
	 * Computes accuracy between two arrays of predictions
	 *
	 * @param predictedLabels the array of labels predicted by the algorithm
	 * @param trueLabels      the array of true labels
	 *
	 * @return the accuracy of the predictions. Its value is in [0, 1]
	 */
	public static double accuracy(byte[] predictedLabels, byte[] trueLabels) {
		// TODO: Implémenter
		return 0d;
	}
}
