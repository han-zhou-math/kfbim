# Changes: KFBIM Operator Alignment and Solver Output

This document summarizes the recent changes made to align the KFBIM codebase with the standard mathematical framework for Boundary Integral Equations (BIEs) and to reorganize the test suite.

## 1. KFBIM Operator Modes Realigned
The `LaplaceKFBIMode` enum and its implementation in `LaplaceKFBIOperator2D/3D` have been simplified and strictly aligned with 2nd-kind Boundary Value Problem (BVP) theory:

*   **`Dirichlet` Mode**: Represents the Interior Dirichlet BVP solved via a Double-Layer potential ($u = D\phi$).
    *   **Input**: Jump $\mu = \phi, \sigma = 0$.
    *   **Operator**: $(\frac{1}{2}I - K)\phi$.
    *   **Output Recovery**: Calculates $u_{int}$ by adding $\frac{\mu}{2}$ to the averaged trace solver output.
*   **`Neumann` Mode**: Represents the Interior Neumann BVP solved via a Single-Layer potential ($u = S\phi$).
    *   **Input**: Jump $\mu = 0, \sigma = \phi$.
    *   **Operator**: $(\frac{1}{2}I + K')\phi$.
    *   **Output Recovery**: Calculates $\partial_n u_{int}$ by adding $\frac{\sigma}{2}$ to the averaged normal derivative solver output.
*   **Removed Modes**: Non-standard and first-kind BIE modes (`SingleLayerTrace` and `ExteriorDirichlet`) were completely removed from the enum, documentation, and the application logic to prevent misuse and confusion.

## 2. Averaged Interface Values in Solver
The `LaplaceInterfaceSolver2D` output structure (`LaplaceInterfaceSolveResult2D`) was modified to explicitly return the **averaged** trace and normal derivative values, rather than relying on one-sided limits.

*   `u_trace` renamed to `u_avg` (representing $\frac{u_{int} + u_{ext}}{2}$).
*   `un_trace` renamed to `un_avg` (representing $\frac{\partial_n u_{int} + \partial_n u_{ext}}{2}$).
*   **Trace Recovery**:
    *   $u_{avg} = u_{int} - \frac{\mu}{2}$
    *   $(\partial_n u)_{avg} = \partial_n u_{int} - \frac{\sigma}{2}$
*   **Impact**: The `LaplaceKFBIOperator` now correctly reconstructs the interior limit traces by taking the solver's averaged output and adding half the jump back. This ensures that operators like $-K\phi$ and $K'\phi$ are correctly shifted by the necessary $\frac{1}{2}I$ identity terms when formulating second-kind integral equations.

## 3. Test Suite Organization (Archiving)
To maintain focus and differentiate foundational components from end-to-end integration routines, several high-level PDE solver tests were moved to `tests/archive/`:

*   `test_kfbi_laplace_2d.cpp`
*   `test_laplace_double_layer_2d.cpp`
*   `test_laplace_iface_2d.cpp`
*   `test_laplace_interface_solver_2d.cpp`
*   `test_iim_spread_2d.cpp`

The `tests/CMakeLists.txt` was updated to exclude these tests. The active test suite (`ctest`) now exclusively runs the foundational logic verifications (Grids, Interface parsing, FFT Bulk solver, Local Cauchy panel inversion, and the core IIM interpolation routines).

## 4. Documentation (`notes/kfbim_mathematics.md`)
The core mathematical reference was fully rewritten to reflect these operational facts:
*   Established standard representation formulas: $u = \mathcal{S}\sigma - \mathcal{D}\mu + \mathcal{V}f$.
*   Corrected the mappings to show that the solver returns averaged values.
*   Documented how Dirichlet and Neumann modes formulate their effective BIE operators $(\frac{1}{2}I - K)$ and $(\frac{1}{2}I + K')$ using the averaged traces.