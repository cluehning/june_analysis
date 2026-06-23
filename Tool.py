import numpy as np
import matplotlib.pyplot as plt


def evaluate_function(func, x, y):
    """Evaluate a user-defined function on arrays or scalars."""
    try:
        values = func(x, y)
    except TypeError:
        values = np.vectorize(func, otypes=[float])(x, y)
    return np.asarray(values, dtype=float)


def partial_derivative(func, x, y, axis="x", h=1e-4):
    """Numerical partial derivative using the central difference method."""
    if axis == "x":
        return (evaluate_function(func, x + h, y)
                - evaluate_function(func, x - h, y)) / (2 * h)
    if axis == "y":
        return (evaluate_function(func, x, y + h)
                - evaluate_function(func, x, y - h)) / (2 * h)
    raise ValueError("axis must be 'x' or 'y'")


def reference_note(concept):
    """Return a short note pointing the user to the relevant definition or theorem in their PDF."""
    notes = {
        "Gradient": "Siehe die Definition 3.2. im Analysis 2, Differential- und Integralrechnung für Funktionen mehrerer reeller Veränderlichen — Rolf Rannacher",
        "Hessian": "Siehe die Definition 3.3. im Analysis 2, Differential- und Integralrechnung für Funktionen mehrerer reeller Veränderlichen — Rolf Rannacher",
        "Jacobian": "Siehe die Definition 3.4 im Analysis 2, Differential- und Integralrechnung für Funktionen mehrerer reeller Veränderlichen — Rolf Rannacher",
        "Divergence": "Siehe die Definition 3.5 im Analysis 2, Differential- und Integralrechnung für Funktionen mehrerer reeller Veränderlichen — Rolf Rannacher",
        "Rotation": "Siehe die Definition 3.6 im Analysis 2, Differential- und Integralrechnung für Funktionen mehrerer reeller Veränderlichen — Rolf Rannacher",
    }
    return notes.get(concept, "Siehe den relevanten Satz oder die Definition in Ihrem PDF.")


def gradient(func, x0, y0, h=1e-4):
    """Return the gradient of a scalar function at (x0, y0)."""
    try:
        gx = partial_derivative(func, x0, y0, axis="x", h=h)
        gy = partial_derivative(func, x0, y0, axis="y", h=h)
        return np.array([gx, gy], dtype=float)
    except Exception as exc:
        raise ValueError(f"Gradient konnte nicht bestimmt werden. {reference_note('gradient')}") from exc


def hessian_matrix(func, x0, y0, h=1e-4):
    """Return the Hessian matrix of a scalar function at (x0, y0)."""
    try:
        f0 = evaluate_function(func, x0, y0)
        fxx = (evaluate_function(func, x0 + h, y0) - 2 * f0 + evaluate_function(func, x0 - h, y0)) / (h**2)
        fyy = (evaluate_function(func, x0, y0 + h) - 2 * f0 + evaluate_function(func, x0, y0 - h)) / (h**2)
        fxy = (evaluate_function(func, x0 + h, y0 + h)
               - evaluate_function(func, x0 + h, y0 - h)
               - evaluate_function(func, x0 - h, y0 + h)
               + evaluate_function(func, x0 - h, y0 - h)) / (4 * h**2)
        return np.array([[fxx, fxy], [fxy, fyy]], dtype=float)
    except Exception as exc:
        raise ValueError(f"Hessematrix konnte nicht bestimmt werden. {reference_note('hessian')}") from exc


def evaluate_vector_field(vector_field, x, y):
    """Evaluate a vector field given as a list/tuple of component functions."""
    if isinstance(vector_field, (list, tuple)):
        return np.array([evaluate_function(component, x, y)
                         for component in vector_field], dtype=float)
    if callable(vector_field):
        values = vector_field(x, y)
        if isinstance(values, (list, tuple, np.ndarray)):
            return np.asarray(values, dtype=float)
    raise TypeError("vector_field must be a list/tuple of two component functions")


def jacobian_matrix(vector_field, x0, y0, h=1e-4):
    """Return the Jacobian matrix of a 2D vector field at (x0, y0)."""
    try:
        if callable(vector_field):
            values = vector_field(x0, y0)
            if isinstance(values, (list, tuple, np.ndarray)) and len(values) == 2:
                p = lambda a, b: vector_field(a, b)[0]
                q = lambda a, b: vector_field(a, b)[1]
            else:
                raise TypeError("callable vector_field must return two components")
        elif isinstance(vector_field, (list, tuple)) and len(vector_field) == 2:
            p, q = vector_field
        else:
            raise TypeError("vector_field must be a callable or a list/tuple with two component functions")

        p_x = (evaluate_function(p, x0 + h, y0)
               - evaluate_function(p, x0 - h, y0)) / (2 * h)
        p_y = (evaluate_function(p, x0, y0 + h)
               - evaluate_function(p, x0, y0 - h)) / (2 * h)
        q_x = (evaluate_function(q, x0 + h, y0)
               - evaluate_function(q, x0 - h, y0)) / (2 * h)
        q_y = (evaluate_function(q, x0, y0 + h)
               - evaluate_function(q, x0, y0 - h)) / (2 * h)
        return np.array([[p_x, p_y], [q_x, q_y]], dtype=float)
    except Exception as exc:
        raise ValueError(
            f"Jacobi-Matrix konnte nicht bestimmt werden. {reference_note('jacobian')}") from exc


def divergence(vector_field, x0, y0, h=1e-4):
    """Return the divergence of a 2D vector field at (x0, y0)."""
    try:
        jac = jacobian_matrix(vector_field, x0, y0, h=h)
        return float(jac[0, 0] + jac[1, 1])
    except Exception as exc:
        raise ValueError(
            f"Divergenz konnte nicht bestimmt werden. {reference_note('divergence')}") from exc


def rotation(vector_field, x0, y0, h=1e-4):
    """Return the 2D rotation/curl of a vector field at (x0, y0)."""
    try:
        jac = jacobian_matrix(vector_field, x0, y0, h=h)
        return float(jac[1, 0] - jac[0, 1])
    except Exception as exc:
        raise ValueError(
            f"Rotation konnte nicht bestimmt werden. {reference_note('rotation')}") from exc


def compute_fourier_spectrum(values):
    """Compute a 2D Fourier spectrum from sampled values."""
    spectrum = np.fft.fft2(values)
    spectrum = np.fft.fftshift(spectrum)
    return np.abs(spectrum)


def taylor_polynomial(func, x0, y0, degree=2, h=1e-4):
    """Build a Taylor polynomial of degree 0, 1 or 2
    around a point (x0, y0)."""
    if degree not in (0, 1, 2):
        raise ValueError("degree must be 0, 1 or 2")

    x0 = float(x0)
    y0 = float(y0)

    f0 = float(evaluate_function(func, np.array([x0]), np.array([y0]))[0])
    fx = float(partial_derivative(func, np.array([x0]),
                                  np.array([y0]), axis="x",
                                  h=h)[0])
    fy = float(partial_derivative(func, np.array([x0]),
                                  np.array([y0]), axis="y",
                                  h=h)[0])

    if degree >= 2:
        fxx = float((evaluate_function(
            func, x0 + h, y0) - 2 * f0 + evaluate_function(func, x0 - h, y0))
                    / (h**2))
        fyy = float((evaluate_function(
            func, x0, y0 + h) - 2 * f0 + evaluate_function(func, x0, y0 - h))
                    / (h**2))
        fxy = float((evaluate_function(
            func, x0 + h, y0 + h) - evaluate_function(func, x0 + h, y0 - h)
                     - evaluate_function(func, x0 - h, y0 + h)
                     + evaluate_function(func, x0 - h, y0 - h))
                    / (4 * h**2))
    else:
        fxx = fyy = fxy = 0.0

    def approx(x, y):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        dx = x_arr - x0
        dy = y_arr - y0
        poly = f0 + fx * dx + fy * dy
        if degree >= 2:
            poly += 0.5 * fxx * dx**2 + fxy * dx * dy + 0.5 * fyy * dy**2
        return poly

    return approx


def analyze_function(func, x_range=(-2, 2), y_range=(-2, 2), resolution=301,
                     h=1e-4, show=True, save_path=None, include_fourier=True,
                     include_taylor=True,
                     taylor_point=(0, 0), taylor_degree=2):
    """Create interactive 3D plots for a function,
    its partial derivatives, and optional transforms."""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z = evaluate_function(func, X, Y)
    dZ_dx = partial_derivative(func, X, Y, axis="x", h=h)
    dZ_dy = partial_derivative(func, X, Y, axis="y", h=h)

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(X, Y, Z, cmap="turbo", edgecolor="none")
    ax1.set_title("f(x, y)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot_surface(X, Y, dZ_dx, cmap="turbo", edgecolor="none")
    ax2.set_title("∂f/∂x")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("df/dx")

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot_surface(X, Y, dZ_dy, cmap="turbo", edgecolor="none")
    ax3.set_title("∂f/∂y")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("df/dy")

    fourier_spectrum = None
    taylor_values = None

    if include_fourier:
        fourier_spectrum = compute_fourier_spectrum(Z)
        fig2 = plt.figure(figsize=(6, 5))
        ax4 = fig2.add_subplot(111)
        im = ax4.imshow(np.log10(
            fourier_spectrum + 1e-12),
                        cmap="viridis",
                        origin="lower",
                        aspect="auto")
        ax4.set_title("Fourier spectrum")
        ax4.set_xlabel("frequency x")
        ax4.set_ylabel("frequency y")
        plt.colorbar(im, ax=ax4, shrink=0.9)
        fig2.tight_layout()

    if include_taylor:
        taylor_approx = taylor_polynomial(
            func, taylor_point[0], taylor_point[1], degree=taylor_degree, h=h)
        taylor_values = taylor_approx(X, Y)
        fig3 = plt.figure(figsize=(10, 5))

        ax5 = fig3.add_subplot(121, projection="3d")
        ax5.plot_surface(X, Y, taylor_values, cmap="turbo", edgecolor="none")
        ax5.set_title(f"Taylor approx. (degree {taylor_degree})")
        ax5.set_xlabel("x")
        ax5.set_ylabel("y")
        ax5.set_zlabel("T")

        ax6 = fig3.add_subplot(122, projection="3d")
        ax6.plot_surface(X, Y, Z - taylor_values,
                         cmap="turbo", edgecolor="none")
        ax6.set_title("Difference to original")
        ax6.set_xlabel("x")
        ax6.set_ylabel("y")
        ax6.set_zlabel("f - T")
        fig3.tight_layout()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
        if include_fourier:
            plt.close(fig2)
        if include_taylor:
            plt.close(fig3)

    return {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "Z": Z,
        "dZ_dx": dZ_dx,
        "dZ_dy": dZ_dy,
        "fourier_spectrum": fourier_spectrum,
        "taylor_values": taylor_values,
    }


# General structure of a function for this tool:
# def example_function(x, y):
#     x = np.asarray(x, dtype=float)
#     y = np.asarray(y, dtype=float)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         z = function_expression
#     return np.nan_to_num(z, nan=0.0)


def exam_style_function(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.exp(-0.5 * (x**2 + y**2)) * np.cos(3 * x - y)
    return np.nan_to_num(z, nan=0.0)


def exam_vector_field(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.sin(x) + y
    q = np.cos(y) - x
    return p, q


def exam_style_function_2(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.sin(x) / (1 + y**2) + np.exp(-0.5 * (x**2 + y**2))
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":
    for name, func in [("exam_style_function", exam_style_function),
                       ("exam_style_function_2", exam_style_function_2)]:
        analyze_function(
            func=func,
            x_range=(-2, 2),
            y_range=(-2, 2),
            resolution=301,
            h=1e-4,
            show=True,
            save_path=None,
            include_fourier=True,
            include_taylor=True,
            taylor_point=(0, 0),
            taylor_degree=2,
        )
        print(f"Displayed plots for {name}")

    point = (0.0, 0.0)
    try:
        print("Gradient (function 1):", gradient(exam_style_function, *point))
        print("Hessian matrix (function 1):\n",
              hessian_matrix(exam_style_function, *point))
        print("Jacobian matrix (vector field 1):\n",
              jacobian_matrix(exam_vector_field, *point))
        print("Divergence (vector field 1):",
              divergence(exam_vector_field, *point))
        print("Rotation (vector field 1):",
              rotation(exam_vector_field, *point))
        print("\n---\n")
        print("Gradient (function 2):",
              gradient(exam_style_function_2, *point))
        print("Hessian matrix (function 2):\n",
              hessian_matrix(exam_style_function_2, *point))
        print("Jacobian matrix (vector field 2):\n",
              jacobian_matrix(exam_vector_field, *point))
        print("Divergence (vector field 2):",
              divergence(exam_vector_field, *point))
        print("Rotation (vector field 2):",
              rotation(exam_vector_field, *point))
    except ValueError as exc:
        print(exc)
    print("An interactive figure window has been opened.")
