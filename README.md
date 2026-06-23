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

## Main functions

### 1. evaluate_function(func, x, y)
Evaluates a user-defined function of two variables at scalar or array inputs.

### 2. partial_derivative(func, x, y, axis="x", h=1e-4)
Computes a numerical partial derivative with respect to x or y using a central difference scheme.

### 3. reference_note(concept)
Returns a short note pointing the user to the relevant definition or theorem in their course PDF or notes.

### 4. gradient(func, x0, y0, h=1e-4)
Returns the gradient of a scalar function at a point $(x_0, y_0)$.

### 5. hessian_matrix(func, x0, y0, h=1e-4)
Returns the Hessian matrix of a scalar function at a point $(x_0, y_0)$.

### 6. evaluate_vector_field(vector_field, x, y)
Evaluates a vector field given either as two component functions or as a callable returning two outputs.

### 7. jacobian_matrix(vector_field, x0, y0, h=1e-4)
Computes the Jacobian matrix of a 2D vector field at a point.

### 8. divergence(vector_field, x0, y0, h=1e-4)
Computes the divergence of a 2D vector field.

### 9. rotation(vector_field, x0, y0, h=1e-4)
Computes the 2D rotation/curl of a vector field.

### 10. compute_fourier_spectrum(values)
Computes a 2D Fourier spectrum from sampled values.

### 11. taylor_polynomial(func, x0, y0, degree=2, h=1e-4)
Builds a Taylor polynomial of degree 0, 1, or 2 around a point.

### 12. analyze_function(func, x_range=(-2, 2), y_range=(-2, 2), resolution=301, h=1e-4, show=True, save_path=None, include_fourier=True, include_taylor=True, taylor_point=(0, 0), taylor_degree=2)
Creates a set of plots for:

- the original function
- the partial derivative with respect to x
- the partial derivative with respect to y
- optionally a Fourier spectrum
- optionally a Taylor approximation and its difference from the original function

## Included example functions

The script already contains a few example functions inspired by typical analysis exercises:

### exam_style_function(x, y)
A smooth oscillatory function that is useful for testing gradients, Hessians, and visual behavior.

### exam_vector_field(x, y)
A simple 2D vector field used for Jacobian, divergence, and rotation examples.

### exam_style_function_2(x, y)
A second example function with different structure, useful for comparing two different analysis cases.

## Example usage

```python
from Tool import analyze_function, gradient, hessian_matrix


def f(x, y):
    return x**2 + y**2

analyze_function(f)
print(gradient(f, 0, 0))
print(hessian_matrix(f, 0, 0))
```

You can also run the script directly:

```bash
python Tool.py
```

## Notes

This project is meant for learning and visualization. The calculations are numerical rather than symbolic, so they are especially helpful for building intuition about how functions behave and how key concepts from multivariable analysis look in practice.

As the project grows, it could be extended to support more variables, more advanced plots, and a more interactive user interface.
