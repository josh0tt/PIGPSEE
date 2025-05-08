[![arXiv](https://img.shields.io/badge/PIGPSEE:%20arXiv-2501.01000-b31b1b.svg)](https://arxiv.org/abs/2501.01000)

# Physics-informed Gaussian processes for Safe Envelope Expansion (PIGPSEE)

PIGPSEE is a Julia module that provides tools for aerodynamic analysis of aircraft using Gaussian processes (GPs) with physics-informed mean functions. The module allows for the estimation of aerodynamic quantities from arbitrary flight test data, significantly reducing the need for extensive and repetitive experimental campaigns.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Examples](#examples)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

Accurately determining key aerodynamic quantities without exhaustive data collection is a central challenge in flight testing. Traditional approaches often require extensive flight test campaigns with methodically selected and revisited points, which are time-consuming and labor-intensive.

PIGPSEE introduces a novel approach that leverages Gaussian processes with physics-informed mean functions to calculate aerodynamic quantities. In this repository, we demonstrate one such example with the pitching moment coefficient $`C_m`$, from arbitrary flight test data. This method allows for precise estimation of $`C_m`$ across diverse aircraft states without the need for meticulously predefined data points, thereby streamlining the flight test process.

### Getting started

Use the julia package manager to add the PIGPSEE module:
```julia
] add https://github.com/josh0tt/PIGPSEE
using PIGPSEE
```

## Usage

1. **Prepare the Data**:

   - Ensure that the flight test data CSV file (e.g., `Rollercoasters_20240821.csv`) is located in the `data` directory or adjust the file path accordingly in the script.

2. **Run the Analysis**:

   - In the Julia REPL, include the module and run the main function:

     ```julia
     using PIGPSEE
     PIGPSEE.run()
     ```

3. **View Results**:

   - The script will generate plots and save them in the project directory.
   - Numerical results and tables will be displayed in the console or saved as CSV files.

## Examples

An example usage is provided in the `run()` function within the module. This function performs the entire analysis pipeline:

- Loads and preprocesses data.
- Computes moments of inertia and moments.
- Fits linear and Gaussian Process models.
- Computes aerodynamic derivatives.
- Generates tables and plots.

## Results

![](https://github.com/josh0tt/PIGPSEE/blob/main/img/surface_plot.png)

The module produces several outputs:

- **Aerodynamic Derivatives**: Estimates of $`C_m`$, $`C_Z`$, and their derivatives with respect to angle of attack and control inputs.
- **Short-Period Dynamics**: Calculations of the short-period frequency ($`\omega_{sp}`$) and damping ratio ($`\zeta_{sp}`$) across different dynamic pressures and Mach numbers.
- **Plots**:
  - **$`\omega_{sp}`$ vs Dynamic Pressure**: Comparison between GP model predictions, linear model predictions, and empirical data.
  - **$`\zeta_{sp}`$ vs Dynamic Pressure**: Similar comparison as above.
  - **3D Surface Plot of $`C_m`$**: Visualization of the pitch moment coefficient over a range of angle of attack and pitch rates.
  - **$`C_m`$ Prediction Plots**: Comparison of GP and linear model predictions against empirical data for specific flight events.

These results demonstrate the effectiveness of the physics-informed GP model in accurately predicting the short-period frequency and damping for the T-38 aircraft across various Mach numbers and dynamic pressure profiles.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

Tested with Julia 1.11.1
