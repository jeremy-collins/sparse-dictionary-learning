# Sparse Dictionary Learning Report

This report discusses the results of applying sparse dictionary learning to image representation using different numbers of atoms.

## Results

The original image was reconstructed using different numbers of atoms specified in the `config.yaml` file. The reconstructed images were displayed alongside the original image for comparison.

![alt text](https://github.com/jeremy-collins/sparse-dictionary-learning/upload/main/images/jeremy_and_yann_results.png "Results")

## Discussion

As the number of atoms increases, the quality of the reconstructed image improves. However, increasing the number of atoms also increases the computational complexity of the algorithm. Therefore, a balance must be struck between image quality and computational complexity.

The quality of the reconstructed image can also be affected by the choice of the sparsity constraint parameter (alpha), the maximum number of iterations for the optimization process, and the tolerance for convergence. Tuning these hyperparameters can potentially improve the quality of the reconstructed image.

In summary, sparse dictionary learning can provide a useful representation for image compression and denoising. Further improvements can be made by adjusting the number of atoms and hyperparameters.