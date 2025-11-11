# Deep Reinforcement Learning Course

This course aims to teach from the basics of RL to advanced algorithms such as GAE.

[![GitHub stars](https://img.shields.io/github/stars/xycoord/deep-rl-course?style=social)](https://github.com/xycoord/deep-rl-course/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Prerequisites

- Machine Learning (Gradient Descent, Neural Networks)
- Basic Probability Theory (Expectations and Distributions)
- Multivariate Calculus
- Python and PyTorch

## 📚 Course Style

Each module consists of:
- Formal mathematical definitions and theory
- Step-by-step algorithm derivations
  
Parts 1 and 2 also include:
- Complete PyTorch implementations
- Runnable experiments

I recommend working through the notebooks carefully, especially the mathematical derivations and proofs, and ensuring you understand each concept before moving on. This material is designed to be precise and concise so that you can learn efficiently - without rushing through. 

## 🗺️ Course Roadmap

| Module | Topic | Colab/Post | Key Concepts |
|--------|-------|--------|--------------|
| Part 1 | RL Basics & Policy Gradients | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Lm_TI-Vrzai-WZQeZL3o7US07vVKWXlQ) | MDPs, Policies, Trajectories, Policy Gradient Theorem, Reward-to-go |
| Part 2 | Discounting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UULTQYnymQOpa7nuaw6mDXnvWRV9R_2y) | Temporal Discounting, Convergence of Infinite Horizons, Variance Reduction |
| Part 3 | Baselines, Actor-Critic & GAE | [![Read Blog Post](https://img.shields.io/badge/Read%20Blog%20Post-blue?style=flat)](https://loganthomson.com/Baselines-Actor-Critic-GAE/) | Constant Baselines, Q-value, Value and Advantage Functions, Actor-Critic, Group-dependent Baselines, GAE |