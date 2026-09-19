# Analysis Tool for Multivariable Calculus

This repository contains a small Python tool for exploring topics from analysis coursework in a more visual and intuitive way. It is designed to help students understand scalar functions, vector fields, and common concepts from multivariable calculus by combining numerical computation with interactive plots.

## What this tool does

The script helps you:

- visualize functions of two variables as 3D surfaces
- compute partial derivatives numerically
- calculate gradients, Hessians, Jacobians, divergence, and rotation
- inspect Fourier spectra and Taylor approximations
- try out example functions inspired by typical analysis exercises

## Features

- Interactive 3D surface plots for functions and partial derivatives
- Numerical differentiation using finite differences
- Gradient and Hessian computation for scalar functions
- Jacobian, divergence and rotation/curl for vector fields
- Fourier spectrum visualization
- Taylor polynomial approximation around a point
- Ready-to-run example functions for coursework-style exploration

## Installation

Make sure you have Python installed, then install the required packages:

```bash
pip install numpy matplotlib
```

## Core Functionality

### Function Evaluation

```evaluate_function(func, x, y)```
Evaluates a function for scalar or array inputs. Supports both vectorized and non-vectorized functions.

```partial_derivative(func, x, y, axis="x", h=1e-4)```
Computes numerical partial derivatives using the central difference method.

### Scalar Functions

```gradient(func, x0, y0)```
Computes the gradient at a given point.

```hessian_matrix(func, x0, y0)```
Computes the Hessian matrix using second-order finite differences.

### Vector Fields

```evaluate_vector_field(vector_field, x, y)```
Evaluates a 2D vector field.

```jacobian_matrix(vector_field, x0, y0)```
Computes the Jacobian matrix.

```divergence(vector_field, x0, y0)```
Computes the divergence as the trace of the Jacobian.

```rotation(vector_field, x0, y0)```
Computes the 2D curl (rotation).

### Transformations and Approximation

```compute_fourier_spectrum(values)```
Computes the 2D Fourier spectrum of sampled data.

```taylor_polynomial(func, x0, y0, degree=2)```
Constructs a Taylor approximation around a point.

### Visualization

```analyze_function(...)```
Creates plots for:

- the function surface
- partial derivatives
- optional Fourier spectrum
- optional Taylor approximation and error

Returns all computed data as NumPy arrays.

## Included Examples

```exam_style_function(x, y)```
Smooth oscillatory function with Gaussian decay.

```exam_style_function_2(x, y)```
Combination of rational and exponential components.

```exam_vector_field(x, y)```
Simple vector field for testing divergence and rotation.

## Notes

- All derivatives are computed numerically using finite differences.
- Results are approximations and may depend on the step size h.
- The tool is intended for learning and visualization rather than symbolic computation.
- Plot interactivity depends on the matplotlib backend.

## Possible Extensions

- Support for higher-dimensional functions
- Interactive controls for parameters
- Improved visualization options
- Integration with symbolic tools

---

### Development Note
Parts of the codebase were created with AI assistance ("vibe coding"), but the underlying ideas, research direction, experimental design, mathematical reasoning, and interdisciplinary extensions are my own. AI was used as an implementation and exploration tool, with all major decisions, modifications, and interpretations guided by the author.

---
