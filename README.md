# Deep Reinforcement Learning for Preventive Maintenance of Repairable Machines in a Flow Line System - A Simulation Study
Final report of Operations Research Applications, Dec 2025 @ NCKU Institute of Manufacturing Information and Systems 



<div align="center">
 
### Group C
| Name                 | Student ID    |
|----------------------|---------------|
| **Jen-Chien, Tseng** | **C24106121** |
| **Sing-Yu, Bao**     | **N46144056** |
| **Josue Fernandez**  | **P76147051** |
</div>

![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)

# 1. Title

**Deep Reinforcement Learning for PM of Repairable Machines in a Flow Line System Simulation Study**

---

# 2. Background and Motivation
## 2.1 Motivation

In flow-line manufacturing systems, unexpected machine failures lead to production loss, buffer starvation or blockage, and high corrective maintenance (CM) costs. Preventive maintenance (PM) decisions must balance maintenance expenses against reliability and throughput. Traditional rule-based or threshold-based PM policies (e.g., age-based rules) are often suboptimal in multi-machine systems with stochastic failures and interactions through buffers. This motivates the use of reinforcement learning (RL) to learn adaptive PM policies directly from system dynamics.

## 2.2 Background

A flow-line system consists of sequential machine stages connected by buffers. Machine degradation is stochastic, failures increase with age, and upstream/downstream interactions amplify local decisions into system-wide effects. Discrete-event simulation is well-suited for modeling such systems, while RL provides a data-driven approach to decision-making under uncertainty without requiring explicit analytical solutions.

## 2.3 Problem Definition

This study formulates preventive maintenance scheduling for a repairable flow-line production system as a Markov Decision Process (MDP) and proposes a Double Deep Q-Network (DDQN) to minimize long-run expected total cost arising from maintenance actions and production losses due to starvation and blockage.

---

# 3. Methodology
## 3.1 Scenario Design and Comparison Framework

This study is explicitly designed as a controlled comparison between two scenarios with different system complexity, aiming to identify when and why deep reinforcement learning becomes necessary for preventive maintenance (PM) decisions.

Rather than evaluating a single system in isolation, we construct two scenarios that share the same modeling philosophy but differ in structural and stochastic characteristics. This allows us to isolate the effect of system complexity on policy learning and performance, as well as measure the added value from training reinforcement learning model for different levels of complexity.

### Scenario A: Simple Flow-Line System (Benchmark Scenario)

- One buffer and one machine stage
- Single machine
- Deterministic production capacity
- Linear age-dependent failure probability
- Minimal buffer interaction (no upstream/downstream propagation)

This scenario serves as a benchmark. Its purpose is not to maximize RL performance, but to verify whether the DDQN can learn intuitive PM behavior comparable to classical age-based or threshold-based policies.

### Scenario B: Complex Flow-Line System (Target Scenario)

- Two buffers and two sequential stages
- Two parallel machines per stage (4 machines total)
- Deterministic production capacity per machine
- Weibull failure distribution (k > 1), capturing aging-induced failure acceleration
- Strong interdependence through buffers, allowing blockage and starvation to propagate

This scenario represents a realistic manufacturing environment, where local maintenance decisions generate non-local system impacts.

**Comparison Objectives The two scenarios are compared along the following dimensions:**

- Preventive maintenance timing behavior
- Frequency and distribution of corrective maintenance events
- Accumulated total system cost
- Sensitivity of learned policies to buffer congestion and starvation risk

## 3.2 Method Justification

**Method Justification Assumptions & Conditions**

- Time is discretized into equal timesteps.
- Machine degradation follows a stochastic process with age-dependent failure probability.
- Corrective maintenance is automatically triggered upon failure.
- No limit is imposed on the number of machines under maintenance simultaneously.

**Why Reinforcement Learning?**

- The system is stochastic, non-linear, and high-dimensional.
- Buffer-machine interactions differ substantially between Scenario A and B.
- RL enables learning adaptive policies that respond to system-level states.

**Pros**

- Allows direct comparison of policy behavior under different structural complexities.
- Scales from simple to complex systems without reformulating the optimization problem.

**Cons / Limitations**

- Training time increases significantly in Scenario B.
- Learned policies are scenario-specific: retraining is needed for different scenarios.

## 3.3 MDP Formulation and DDQN
The preventive maintenance decision problem is formulated as a Markov Decision Process (MDP).

**State space**
At time step *t*, the system state is defined as the composition of three vectors:

$$s_t = [A_t, B_t, M_t]$$

where `A_t` denotes machine ages, `B_t` denotes buffer inventory levels, and `M_t` denotes remaining maintenance time for each machine.

**Action space**
The action is a vector of binary decisions:

$$a_t = (a_t^1, a_t^2, ..., a_t^M), a_t^i ∈ [0,1]$$

where `a_t^i = 1` indicates performing preventive maintenance on machine `i`. Corrective maintenance is triggered automatically upon failure.

**State transition**
System transitions are governed by production capacity, buffer constraints, stochastic arrivals, and probabilistic machine failures. Transition probabilities are unknown and learned implicitly through interaction.

**Uncertainty**
- Scenario A: Linear aging failure probability (factor of 0.01)
- Scenario B: Weibull-distributed failure time (k = 2, λ = 30)

WIP arrivals follow a periodic Poisson process in both scenarios.

**Reward function**
The immediate reward is defined as the negative total cost incurred in each time step:

$$r_t = - (C_{PM,t} + C_{CM,t} + C_{block,t} + C_{starv,t} + C_{idle,t})$$

The objective is to learn a policy `π(a|s)` that maximizes the expected discounted cumulative reward.

**Learning Algorithm**

- Double Deep Q-Network (DDQN)
- Shared feature layers + machine-specific independent linear layer with decision heads
- Experience replay buffer size: 5000
- Training episodes: 800
- Steps per episode: 1,000
- Training frequency: once every 4 steps
- Batch size: 128
- Discount factor (γ): 0.95
- Learning rate starts at 0.0004 and decays by a factor of 0.5 every 200 episodes
- ε-greedy exploration with logarithmic decay
- Target network updates every 10 episodes
---

# 4. Data Collection and Analysis Results
## 4.1 Data Collection

Two datasets are generated via simulation, corresponding to the two scenarios.
- Scenario A dataset: Low-dimensional state space, linear failure process
- Scenario B dataset: High-dimensional state space, Weibull failure process

All data are generated online during agent–environment interaction. This ensures consistency between training and evaluation conditions and allows fair comparison across scenarios.

<div align="center">

| Dimension     | Scenario A | Scenario B |
| ------------- | ---------- | ---------- |
| Machines      | 1          | 4          |
| Buffers       | 1          | 2          |
| Failure model | Linear     | Weibull    |
| Interaction   | None       | Strong     |
| Value of RL   | Low        | High       |

</div>

## 4.2 Analysis

The DDQN is trained and evaluated separately under Scenario A and Scenario B using identical learning hyperparameters. This ensures that observed performance differences are attributable to system structure rather than algorithmic tuning.

For each scenario, we track:
- Episode-level cumulative reward (total cost)
- Preventive vs corrective maintenance counts
- Average machine age at PM execution
- Frequency of starvation and blockage events
- Training convergence behavior is also compared to examine how state dimensionality and stochastic complexity affect learning stability.

## 4.3 Results and Managerial Implications

**Comparative Results**

In Scenario A, the DDQN converges quickly and learns a PM policy closely resembling an age-threshold rule, indicating limited marginal benefit of complex decision-making.
<img width="1336" height="855" alt="圖片" src="https://github.com/user-attachments/assets/4d861efe-b90e-4c29-be17-e8d6ebfa7d81" />

In Scenario B, the learned policy deviates significantly from pure age-based behavior. PM actions are influenced by buffer occupancy, downstream congestion, and failure risk propagation.
<img width="1311" height="855" alt="圖片" src="https://github.com/user-attachments/assets/c0295df6-af05-47b0-8ca8-77ad8cd90355" />

Corrective maintenance frequency is substantially reduced in Scenario B compared to reactive maintenance, despite similar average PM effort.

Learning convergence in Scenario B is slower and exhibits higher variance, reflecting increased state dimensionality and stochasticity.

**Managerial Implications**

For low-complexity production systems, simple heuristic PM rules may be sufficient.

As system complexity increases, ignoring buffer interactions leads to inefficient maintenance timing.

RL-based PM policies are most valuable in environments where maintenance decisions have system-wide ripple effects.

<img width="984" height="583" alt="圖片" src="https://github.com/user-attachments/assets/043311d4-1642-40de-bfa8-bb994906c170" />

5. Conclusion

This study demonstrates that preventive maintenance scheduling in repairable flow-line systems can be effectively modeled as an MDP and solved using DDQN. By integrating machine degradation, buffer dynamics, and stochastic demand, the proposed approach provides a flexible and adaptive alternative to traditional maintenance heuristics. Future work may incorporate maintenance capacity constraints, partial observability, or multi-objective optimization.

---

# 6. References

Alrabghi, A., & Tiwari, A. (2016). Simulation-based optimisation of maintenance systems. Computers & Industrial Engineering, 82, 167–182.

Cassady, C. R., & Kutanoglu, E. (2005). Integrating preventive maintenance planning and production scheduling. IEEE Transactions on Reliability, 54(2), 304–309.

Hung, YH., Shen, HY. & Lee, CY. Deep reinforcement learning-based preventive maintenance for repairable machines with deterioration in a flow line system. Ann Oper Res (2024).https://doi.org/10.1007/s10479-024-06207-x

Jianyu Su, Jing Huang, Stephen Adams, Qing Chang, and Peter A. Beling. 2022. Deep multi-agent reinforcement learning for multi-level preventive maintenance in manufacturing systems▪. Expert Syst. Appl. 192, C (Apr 2022). https://doi.org/10.1016/j.eswa.2021.116323

Li, H., & Meerkov, S. M. (2009). Production Systems Engineering. Springer.

Liu, Y., Wang, W., Hu, Y., Hao, J., Chen, X., & Gao, Y. (2019). Multi-agent game abstraction via graph attention neural network. arXiv preprint arXiv:1911.10715. https://doi.org/10.48550/arXiv.1911.10715

Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.

Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. AAAI Conference on Artificial Intelligence.

Wang, J., & Zhang, H. (2019). Reinforcement learning approaches for preventive maintenance: A review. Computers & Industrial Engineering, 135, 28–41.


