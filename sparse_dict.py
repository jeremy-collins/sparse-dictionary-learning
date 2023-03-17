import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import yaml

class SparseDictLearning:
    """
    A class to implement sparse dictionary learning for images using numpy.
    
    Attributes:
    -----------
    atoms : int
        The number of atoms (columns) in the dictionary matrix.
    max_iter : int, optional
        The maximum number of iterations for the optimization process.
    alpha : float, optional
        The sparsity constraint parameter.
    tol : float, optional
        The tolerance for convergence.
    dictionary : numpy array
        The dictionary matrix.
    """

    def __init__(self, atoms, max_iter=200, alpha=5.0, tol=1e-6):
        """
        Initialize the SparseDictLearning class with the given parameters.

        Parameters:
        -----------
        atoms : int
            The number of atoms (columns) in the dictionary matrix.
        max_iter : int, optional
            The maximum number of iterations for the optimization process.
        alpha : float, optional
            The sparsity constraint parameter.
        tol : float, optional
            The tolerance for convergence.
        """

        self.atoms = atoms
        self.max_iter = max_iter
        self.alpha = alpha
        self.tol = tol
        self.dictionary = None

    def _init_dict(self, X, n_atoms):
        """
        Initializes self.dictionary using PCA.
        
        Args:
            X : numpy array
                The input image data with shape (height, width)
            n_atoms : int
                The number of atoms for the dictionary.
            
        Returns:
            None
        """

        # Calculating the mean of the data
        mean = np.mean(X, axis=0)

        # Subtracting the mean from the data
        centered_X = X - mean

        # Computing the covariance matrix of the centered data
        covariance= np.dot(centered_X.T, centered_X) / (X.shape[0] - 1)

        # Computing the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Sorting the eigenvectors by the corresponding eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Selecting the top K eigenvectors to initialize the dictionary
        self.dictionary = sorted_eigenvectors[:, :n_atoms]

        # Normalizing the dictionary
        dictionary_norms = np.linalg.norm(self.dictionary, axis=0)
        self.dictionary /= dictionary_norms[np.newaxis, :]

    def _update_codes(self, X):
        """
        Update the sparse codes using the orthogonal matching pursuit (OMP) algorithm, which we implement from scratch.

        Parameters:
        -----------
        X : numpy array
            The input image data with shape (height, width)

        Returns:
        --------
        codes : numpy array
            The updated sparse codes.
        """
            
        rows, cols = X.shape
        codes = np.zeros((rows, self.atoms))

        # OMP algorithm
        for i in range(rows):
            # The residual is the ith row of X
            x = X[i, :]
            residual = x.copy()
            support = []

            for _ in range(int(self.alpha)):
                # Finding the atom with maximum correlation to the residual and adding it to the support
                correlations = self.dictionary.T @ residual
                chosen_atom = np.argmax(np.abs(correlations))
                support.append(chosen_atom)

                # Updating the residual using the selected atoms
                selected_atoms = self.dictionary[:, support]
                coefficients, _, _, _ = np.linalg.lstsq(selected_atoms, x, rcond=None)
                residual = x - selected_atoms @ coefficients

                if np.linalg.norm(residual) < self.tol:
                    break

            codes[i, support] = coefficients

        return codes

    def _update_dict(self, X, codes):
        """
        Update the dictionary using the method of optimal directions (MOD).

        Parameters:
        -----------
        X : numpy array
            The input data matrix.
        codes : numpy array
            The sparse codes.
        """

        for i in range(self.atoms):
            indices = np.nonzero(codes[:, i])[0]
            if len(indices) > 0:
                atom_codes = np.zeros_like(codes)
                atom_codes[:, i] = codes[:, i]
                residual = X.T[:, indices] - self.dictionary @ atom_codes[indices, :].T
                u, s, vt = np.linalg.svd(residual @ atom_codes[indices, i].reshape(-1, 1))
                self.dictionary[:, i] = u[:, 0]

        # Normalizing the dictionary
        dictionary_norms = np.linalg.norm(self.dictionary, axis=0)
        self.dictionary /= dictionary_norms[np.newaxis, :]

    def _converged(self, X, old_dictionary):
        """
        Check for convergence using the Frobenius norm of the difference between
        the current and the previous dictionary matrices.

        Parameters:
        -----------
        X : numpy array
            The input data matrix.
        old_dictionary : numpy array
            The previous dictionary matrix.

        Returns:
        --------
        converged : bool
            True if the optimization process has converged, False otherwise.
        """

        difference = np.linalg.norm(self.dictionary - old_dictionary, 'fro')
        return difference < self.tol

    def fit(self, X):
        """
        Fit the dictionary to the input data X.

        Parameters:
        -----------
        X : numpy array
            The input data matrix.
        """

        if self.dictionary is None:
            self._init_dict(X, self.atoms)
        for iteration in range(self.max_iter):
            codes = self._update_codes(X)
            old_dictionary = self.dictionary.copy()
            self._update_dict(X, codes)

            if self._converged(X, old_dictionary):
                break

def main(image_path):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Reading the image
    image = plt.imread(image_path)

    # converting from uint8 to float32 to make the dictionary and code matrix math cleaner
    image = image.astype(np.float32) / 255

    # Separating the channels so we can learn a representation for each channel
    R, G, B = cv2.split(image)

    # Flattening each channel to a 2D array
    X = [R.reshape(R.shape[0], -1), G.reshape(G.shape[0], -1), B.reshape(B.shape[0], -1)] # R, G, B
    
    # Defining the number of atoms to test
    # atoms_list = [10, 50, 100, 200]
    # atoms_list = [10, 50]
    # atoms_list = [10]
    # atoms_list = config['atoms_list']

    fig, axes = plt.subplots(1, len(config['atoms_list']) + 1, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    for i, atoms in enumerate(config['atoms_list']):
        model = SparseDictLearning(atoms=atoms, alpha=config['alpha'], max_iter=config['max_iter'], tol=config['tol'])

        # intializing the reconstructed image
        reconstructed = np.zeros_like(image, dtype=np.float32)

        for j in range(3):
            model.fit(X[j])
            code = model._update_codes(X[j])
            # Reconstruct the image
            reconstructed[:, :, j] = (model.dictionary @ code.T).T

        # clipping the values to be between 0 and 1
        reconstructed = np.clip(reconstructed, 0, 1)

        # converting the image to uint8 to display it
        reconstructed = (reconstructed * 255).astype(np.uint8)

        # Displaying the reconstructed image
        axes[i + 1].imshow(reconstructed)
        axes[i + 1].set_title(f'{atoms} Atoms')
        axes[i + 1].axis('off')

    plt.show()

if __name__ == '__main__':
    image_path = sys.argv[1]
    main(image_path)