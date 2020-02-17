# MCBB-InertialOscillators

This repository contains the code to analyze a network of inertial oscillators
using MCBB, and contains the script that generates the figures that appear in
the accompanying publication (TODO).

# Inertial Kuromato Network

We investigate the network equation

$\ddot{\theta}_i = -\alpha\dot{\theta}_i + \Omega_i + \lambda\sum_{j=1}^N A_{ij}\, sin(\theta_j - \theta_i)$

with the adjacency matrix $A_{ij} = 1$ if two nodes are connected, otherwise $A_{ij}=0$.


## ODE Formulation

$\dot{\theta} = \omega$

$\dot{\omega} = \Omega_i - \alpha\omega + \lambda\sum_{j=1}^N A_{ij}\, sin(\theta_j - \theta_i)$

* $\Omega_i$ -> drive[i]
* $\alpha$ -> damping
* $\lambda$ -> coupling

### Default Configuration

* Drive: $\mathbf{\Omega} = \left[ -1, 1, -1, 1, -1, ...\right]$
* Damping: $\alpha = 0.1$
* **Coupling**: $\lambda$, varied here


* we vary the global coupling $\lambda$
