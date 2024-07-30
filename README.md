# Optimization Techniques for t-SNE Visualizations

## Project Overview

This project explores various optimization techniques for improving the performance and reliability of t-SNE (t-distributed Stochastic Neighbor Embedding), a popular method for dimensionality reduction. The study focuses on addressing the limitations of t-SNE, particularly its sensitivity to initialization and the challenges in preserving global data structure.

## Table of Contents

1. [Introduction](#introduction)
2. [Key Concepts](#key-concepts)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Improvements and Techniques](#improvements-and-techniques)
5. [Experiments and Results](#experiments-and-results)
6. [Conclusion](#conclusion)
7. [How to Use](#how-to-use)
8. [Contributors](#contributors)
9. [License](#license)

## Introduction

Dimensionality reduction is crucial for visualizing and interpreting high-dimensional data. While methods like PCA and UMAP are widely used, t-SNE offers unique benefits in preserving local structure. This project delves into the challenges of t-SNE, proposing enhancements to its optimization process.

## Key Concepts

- **Dimensionality Reduction**: Reducing the number of random variables under consideration by obtaining a set of principal variables.
- **t-SNE**: A non-linear technique that helps in visualizing high-dimensional data by mapping it into a lower-dimensional space.

## Mathematical Formulation

t-SNE works by minimizing the Kullback-Leibler divergence between two distributions, one that measures pairwise similarities of the input objects in the high-dimensional space and another that measures pairwise similarities of the corresponding low-dimensional points.

### SNE and t-SNE

- **SNE**: Focuses on preserving local structure.
- **t-SNE**: Uses a Student-t distribution in the low-dimensional space to better capture global structure.

## Improvements and Techniques

Several methods were explored to enhance the performance of t-SNE:

- **Momentum and Adaptive Learning Rates**: To stabilize and accelerate convergence.
- **Early Compression and Exaggeration**: To enhance the separability of clusters.
- **Noise Contrastive Estimation (NCE)**: Reformulating the objective function for better optimization.

## Experiments and Results

Experiments were conducted using synthetic datasets like circles, blobs, and clusters. The performance of various methods was compared, highlighting the effectiveness of the proposed improvements.

![Model Accuracy Comparison](images/Model_Accuracy_Comparison.png)

## Conclusion

Despite improvements, the project identifies that t-SNE's sensitivity to initial conditions and difficulty in learning complex landscapes remain significant challenges. Future work will explore alternative optimization strategies.

## How to Use

1. **Installation**: Instructions for setting up the environment and dependencies.
2. **Data**: Information on datasets used and how to acquire them.
3. **Running the Code**: Steps to execute the experiments and reproduce the results.
4. **Customization**: Guidelines for tweaking the parameters and settings.

## Contributors

- **Liam Brinker** - Project Lead
- **Prof. Ofra Amir** - Supervisor
- **Dr. Nir Rosenfeld** - Co-Supervisor

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
