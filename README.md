# 📡 Dynamic Systems Estimation

A comprehensive repository for **Estimation Theory for Dynamic Systems**, containing algorithm implementations, tutorials, and course assignments. Covers classical and modern estimation techniques including Kalman filtering, Bayesian methods, and particle filters.

---

## 📁 Repository Structure

```
dynamic-systems-estimation/
├── estimation/          # Core algorithm implementations
├── tutorials/           # Step-by-step tutorial notebooks
├── Assignments/         # Course assignments and solutions
└── README.md
```

---

## 🧠 Topics Covered

### Filtering & Estimation Algorithms
- **Kalman Filter (KF)** — Optimal linear estimator for Gaussian noise
- **Extended Kalman Filter (EKF)** — Nonlinear systems via first-order linearization
- **Unscented Kalman Filter (UKF)** — Nonlinear systems via sigma-point propagation
- **Particle Filter (PF)** — Sequential Monte Carlo for non-Gaussian distributions
- **Bayesian Estimator** — Full posterior state estimation

### Supporting Concepts
- State-space modeling of dynamic systems
- Observability and controllability
- Noise modeling (process noise & measurement noise)
- CRLB (Cramér-Rao Lower Bound)
- MAP and MMSE estimation

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy scipy matplotlib jupyter
```

### Clone the Repository
```bash
git clone https://github.com/NK278/dynamic-systems-estimation.git
cd dynamic-systems-estimation
```

### Run a Tutorial
```bash
cd tutorials/
jupyter notebook
```

---

## 📚 Tutorials

Tutorials are structured as Jupyter notebooks, walking through concepts with derivations, code, and visualizations.

| # | Topic |
|---|-------|
| 1 | Introduction to State Estimation |
| 2 | Kalman Filter — Derivation & Implementation |
| 3 | Extended Kalman Filter |
| 4 | Unscented Kalman Filter |
| 5 | Particle Filter |
| 6 | Bayesian Estimation Framework |

---

## 📝 Assignments

The `Assignments/` folder contains problem sets and solutions corresponding to the course curriculum.

---

## 📦 Core Implementations (`estimation/`)

All algorithms are implemented from scratch in Python using `numpy` and `scipy`.

```python
from estimation.kalman_filter import KalmanFilter
from estimation.ekf import ExtendedKalmanFilter
from estimation.particle_filter import ParticleFilter
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 👤 Author

**Nishchal Gaur (NK278)**  
Graduate Student | IIIT Delhi  
[GitHub Profile](https://github.com/NK278)

---

## 📄 License

This repository is for academic and educational purposes.  
Licensed under the [MIT License](LICENSE).
