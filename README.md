# Probabilistic Artificial Intelligence (PAI) - Course Projects

This repository contains the solutions for the projects of the **Probabilistic Artificial Intelligence** course, taught by Prof. Dr. Andreas Krause at **ETH Zürich**.

---

## 📝 Project Descriptions

The repository includes implementations for four main tasks of the course, each focusing on a different aspect of probabilistic artificial intelligence.

### 💨 Task 1: Gaussian Process Regression for Air Pollution Prediction

* **Objective:** To develop a model to predict the concentration of fine particulate matter (PM2.5) in locations without measurement stations, using Gaussian Process (GP) regression.
* **Challenges:**
    1.  **Model Selection:** Finding the most suitable kernel and its hyperparameters to accurately model the data.
    2.  **Scalability:** Addressing the computational complexity of GPs (which grows with O(n³)) on large datasets.
    3.  **Asymmetric Cost:** Avoiding underestimation of pollution concentration in candidate residential areas, where underestimation errors are penalized 50 times more than other errors.
* **Approach:** The solution implements a model which use multiple local GPs to face scalability. To handle the asymmetric cost function, multiple local GPs the final prediction is adjusted by adding a term proportional to the GP's standard deviation in the specific areas of interest.

---

### 🛰️ Task 2: Image Classification with Bayesian Neural Networks

* **Objective:** To classify satellite images into different land use categories. The training set consists of single-label images, while the test set includes multi-label images and images with atmospheric conditions like clouds or snow, which are not present in the training data.
* **Approach:** This task was solved using a **Bayesian Neural Network (BNN)**. Unlike standard neural networks that learn a single set of optimal weights, a BNN learns a probability distribution over its weights. This approach is ideal for quantifying **epistemic uncertainty**—the model's uncertainty about its own predictions.
* **Implementation:** Since computing the true posterior distribution over the weights is intractable, an approximation method called **Stochastic Weight Averaging Gaussian (SWAG)** was used. The **MultiSWAG** variant combines several independent SWAG models into a mixture of Gaussians for a more robust approximation.
* **Evaluation:** The ability to quantify uncertainty allows the model to abstain (predict -1) on out-of-distribution images, which is handled by a custom **cost function**. The model's performance was also evaluated

---

### 💊 Task 3: Hyperparameter Tuning with Bayesian Optimization

* [cite_start]**Objective:** To optimize the structural features of a drug candidate to maximize its bioavailability (measured by logP), while respecting a constraint on its synthesis difficulty (Synthetic Accessibility - SA)[cite: 11, 17].
* [cite_start]**Formalization:** The problem is defined as maximizing a noisy objective function $f(x)$ subject to an unknown constraint $v(x) < \kappa$[cite: 23].
* [cite_start]**Approach:** A **Constrained Bayesian Optimization** strategy was implemented[cite: 791].
    * [cite_start]Both the objective function and the constraint were modeled using **Gaussian Processes**[cite: 794].
    * [cite_start]The acquisition function used is the **Expected Improvement (EI)**, modified to account for the probability of satisfying the constraint[cite: 795, 798].
    * [cite_start]To improve numerical stability and exploration in a noisy environment, techniques such as **LogEI** and a *plug-in* estimate for the current optimum were used[cite: 804, 813].

---

### 🤖 Task 4: Implementing an Off-policy RL algorithm

* [cite_start]**Objective:** To implement an off-policy Reinforcement Learning (RL) algorithm to solve the **Cartpole** problem, which involves balancing a pole mounted on a controllable cart[cite: 612, 617, 618].
* [cite_start]**Algorithm:** The solution is based on **Soft Actor-Critic (SAC)**, a modern off-policy algorithm that maximizes a combination of expected reward and policy entropy[cite: 233]. [cite_start]The entropy encourages greater exploration[cite: 233].
* [cite_start]**Architecture:** The agent consists of three neural networks[cite: 234]:
    1.  [cite_start]**Actor:** Approximates the stochastic policy $\pi_{\theta}$, which maps states to actions[cite: 235].
    2.  [cite_start]**Critic:** Models two Q-functions ($Q_1, Q_2$) to mitigate overestimation bias[cite: 236, 249].
    3.  [cite_start]**Critic Target:** Separate target networks, updated slowly (bootstrapping), to stabilize training[cite: 237, 250].
* [cite_start]**Objective Function:** The agent maximizes the entropy-regularized expected reward, where the *temperature parameter* $\alpha$ balances the trade-off between exploitation and exploration[cite: 239, 240].
    $$ \pi^{*} = \arg\max_{\pi} \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t (R(x_t, a_t, x_{t+1}) + \alpha H(\pi(\cdot|x_t))) \right] $$
