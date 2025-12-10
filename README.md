# Scaling of the Variance of the Error Term in the Dirichlet Divisor Problem

**High-precision numerical investigation of the fluctuations of Δ(x)**  
Kenneth P. Perez¹ · Warren I. Luzano²  
¹ Independent Researcher  
² University of Science and Technology of Southern Philippines  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281/zenodo.17880291-blue)](https://doi.org/10.5281/zenodo.17880291)

## Overview

This repository contains the code.

We perform a high-precision computation of the Dirichlet divisor error term  
Δ(x) = ∑_{n≤x} d(n) − x(log x + 2γ − 1)  
for approximately 6.5 million values of x evenly distributed (in logarithmic scale) from x = 10³ up to x = 10¹⁵.

A log–log regression of the variance Var(Δ(x)) ∼ x^{2H} yields the Hurst exponent  
**H ≈ 0.251267 ± 0.0000133** (95 % confidence interval from bootstrap).

This value is remarkably close to 1/4, supporting (but not proving nothing about) the conjecture that the true asymptotic is Δ(x) = O(x^{1/4 + ε}) for every ε > 0.

## Repository contents
