# Deep Reinforcement Learning for Preventive Maintenance in Flow-Line Manufacturing Systems with scenario comparison
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

**Deep Reinforcement Learning for Preventive Maintenance in Flow-Line Manufacturing Systems with scenario comparison**

---

# 2. Background and Motivation
## 2.1 Motivation

In flow-line manufacturing systems, machine failures not only incur direct repair costs but also propagate disruptions through upstream starvation and downstream blockage, resulting in significant production losses. Preventive maintenance (PM) policies are therefore critical for maintaining system reliability and throughput.

However, in practice, manufacturing systems operate under diverse conditions, such as different machine reliability characteristics, buffer capacities, and cost structures. Traditional preventive maintenance strategies, including run-to-failure or fixed age-based policies, are typically designed under restrictive assumptions and often lack adaptability across varying operational environments. Recent studies have shown that such rule-based policies can be suboptimal in flow-line systems with stochastic deterioration and strong machine interactions, motivating the need for more adaptive decision-making frameworks (Hung et al., 2024).

Recent advances in reinforcement learning (RL) offer a promising data-driven approach to learning maintenance policies directly from system dynamics through sequential interactions with the environment. RL-based approaches have been increasingly applied to manufacturing maintenance problems, demonstrating their ability to handle uncertainty and complex system interactions, including multi-level and multi-agent preventive maintenance settings (Su et al., 2022). Motivated by this potential, this study aims to examine the effectiveness and robustness of RL-based preventive maintenance policies across multiple flow-line manufacturing scenarios.

## 2.2 Background

A flow-line manufacturing system consists of sequential production stages connected by finite buffers. Machine deterioration is stochastic, and failure probabilities typically increase with operating age. Due to buffer constraints and machine interactions, failures at one stage can cause cascading effects throughout the system, amplifying local disruptions into system-wide performance losses. Analytical optimization of maintenance policies in such systems is challenging, particularly when failure processes and production flows are uncertain and interdependent.

Discrete-event simulation provides a flexible tool for modeling complex flow-line dynamics under stochastic deterioration and production variability. When combined with reinforcement learning, simulation-based environments enable adaptive maintenance policies to be learned without requiring explicit analytical system models. Recent surveys have highlighted the growing role of reinforcement learning and deep reinforcement learning in maintenance planning, scheduling, and optimization problems, especially in settings where traditional optimization approaches become intractable (Ogunfowora & Najjaran, 2023).

Several recent studies have successfully applied simulation-based deep reinforcement learning frameworks to production systems with deterioration. For example, Ferreira Neto et al. (2024) demonstrated the effectiveness of deep reinforcement learning for maintenance optimization in a scrap-based steel production line, while Geurtsen et al. (2026) investigated deep reinforcement learning for optimal maintenance planning in deteriorating flow-line systems. These studies illustrate the potential of integrating simulation and deep reinforcement learning to address complex maintenance decision problems. Building on this line of research, the present study adopts a simulation–learning framework to systematically compare learned preventive maintenance policies with traditional rule-based approaches under different system configurations.

## 2.3 Problem Definition

- This study formulates the preventive maintenance scheduling problem in a repairable flow-line manufacturing system as a Markov Decision Process (MDP), Double Deep Q-Network (DDQN) is proposed to learn maintenance decisions that minimize the long-run expected total cost, including preventive and corrective maintenance costs as well as production losses due to starvation and blockage. 

- The learned policy is evaluated across different scenarios and compared with maintenance strategies to assess its performance and robustness.

---

# 3. Methodology
## 3.1 Scenario Design and Comparison Framework

The general structural of flow-Line systems show as below:

<img width="5330" height="2250" alt="圖片" src="https://github.com/user-attachments/assets/0320ef7b-7a9e-4cb8-9fb1-fed2107ef59e" />

Our study is explicitly designed as a controlled comparison between two scenarios with different system complexity, aiming to identify when and why deep reinforcement learning becomes necessary for preventive maintenance (PM) decisions.

Rather than evaluating a single system in isolation, we construct two scenarios that share the same modeling philosophy but differ in structural and stochastic characteristics. This allows us to isolate the effect of system complexity on policy learning and performance, as well as measure the added value from training reinforcement learning model for different levels of complexity.

### Scenario A: Simple Flow-Line System (Benchmark Scenario)

- One buffer and one machine stage
- Single machine
- Deterministic production capacity
- Linear age-dependent failure probability
- Minimal buffer interaction (no upstream/downstream propagation)

This scenario serves as a benchmark. Its purpose is not to maximize RL performance, but to verify whether the DDQN can learn intuitive PM behavior comparable to classical age-based or threshold-based policies.

<img width="1656" height="540" alt="圖片" src="https://github.com/user-attachments/assets/02f95fe7-415a-42ab-8cdf-bda2a09f787d" />

### Scenario B: Complex Flow-Line System (Target Scenario)

- Two buffers and two sequential stages
- Two parallel machines per stage (4 machines total)
- Deterministic production capacity per machine
- Weibull failure distribution (k > 1), capturing aging-induced failure acceleration
- Strong interdependence through buffers, allowing blockage and starvation to propagate

This scenario represents a realistic manufacturing environment, where local maintenance decisions generate non-local system impacts.

<img width="2951" height="975" alt="圖片" src="https://github.com/user-attachments/assets/4045ec30-7d25-4317-b177-7a603b656cf2" />


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

The DDQN is trained and evaluated separately under Scenario A and Scenario B using identical network architectures and learning hyperparameters. This design ensures that observed performance differences can be attributed to differences in system structure and stochastic characteristics rather than algorithmic tuning.

For each scenario, we track:
- Episode-level cumulative reward (total cost)
- Preventive vs corrective maintenance counts
- Average machine age at PM execution
- Frequency of starvation and blockage events
- Training convergence behavior is also compared to examine how state dimensionality and stochastic complexity affect learning stability.

During training, the agent interacts with the simulated environment and makes maintenance decisions based on the observed system state. Performance is evaluated using episode-level cumulative reward, which represents the total expected cost incurred from preventive maintenance, corrective maintenance, and production losses due to starvation and blockage. In addition to cumulative reward, operational indicators are recorded to provide insights into policy behavior, including the frequency of preventive and corrective maintenance actions, the average machine age at which preventive maintenance is performed, and the occurrence of starvation and blockage events.

To assess learning dynamics, convergence behavior and reward variance across episodes are also analyzed. Comparing learning stability between the two scenarios allows us to examine how state-space dimensionality and stochastic failure processes influence the effectiveness and efficiency of reinforcement learning in preventive maintenance problems.


## 4.3 Results and Managerial Implications

**Comparative Results**

In Scenario A, characterized by a low-dimensional state space and a linear failure process, the DDQN converges rapidly. The learned preventive maintenance policy closely resembles a simple age-threshold rule, with preventive actions triggered primarily by machine age rather than system-level conditions. This result suggests that in low-complexity environments with minimal machine interactions, sophisticated decision-making provides limited additional benefit over traditional heuristic policies.

<img width="1503" height="1018" alt="圖片" src="https://github.com/user-attachments/assets/bb0db56e-0940-41b2-a6cc-e5e78208ad41" />

In Scenario A, the accumulated rewards (negative costs) over 1000 steps show that the Double DQN, Biweekly, and Run-to-Fail policies generally achieve higher rewards, indicating lower costs, while the Weekly and Monthly strategies tend to incur slightly higher costs. The variability across strategies is relatively similar, although the Monthly policy exhibits slightly larger fluctuations. Some outliers appear in the Double DQN and Biweekly strategies, reflecting occasional extreme low-cost events.

<img width="984" height="583" alt="圖片" src="https://github.com/user-attachments/assets/0292a175-469b-4d8f-b6f7-5d2f7f429cc3" />

In contrast, Scenario B exhibits fundamentally different behavior. Due to the presence of multiple machines, finite buffers, and a Weibull failure process, maintenance decisions have pronounced system-wide effects. The learned policy deviates significantly from a pure age-based strategy. Preventive maintenance actions are influenced not only by machine deterioration but also by buffer occupancy and downstream congestion, reflecting the agent’s ability to anticipate failure propagation and production disruptions.

<img width="1476" height="1018" alt="圖片" src="https://github.com/user-attachments/assets/57677447-119e-4cbd-9705-e925f1ab8a40" />

In Scenario B, the scale of accumulated costs is much larger. The Double DQN policy clearly outperforms the others, with the lowest and most stable costs. Weekly maintenance shows the worst performance, with several extreme high-cost outliers, indicating high variability and potential instability. Biweekly and Monthly strategies perform moderately, while Run-to-Fail yields intermediate costs with moderate stability. These results suggest that Double DQN not only minimizes costs but also provides the most predictable outcomes, whereas frequent scheduled maintenance, particularly Weekly, may lead to unpredictable high-cost events in this scenario.

<img width="984" height="583" alt="圖片" src="https://github.com/user-attachments/assets/9855ae3a-6dcf-4831-b2cd-00d170466fc1" />

Learning convergence in Scenario B is slower and exhibits higher variance, highlighting the increased difficulty of training reinforcement learning agents in high-dimensional and highly stochastic environments.

**Comparison Table** 

<div align="center">
 
| Dimension / Feature         | Scenario A (Simple)        | Scenario B (Complex)                         |
| --------------------------- | -------------------------- | -------------------------------------------- |
| Machines                    | 1                          | 4 (2 per stage)                              |
| Buffers                     | 1                          | 2                                            |
| Failure Model               | Linear age-dependent       | Weibull (k > 1, aging-induced)               |
| Machine Interactions        | Minimal / None             | Strong (starvation & blockage propagate)     |
| PM Strategy Learned         | Age-based (heuristic-like) | System-aware (considers buffer & congestion) |
| Corrective Maintenance (CM) | Few                        | Reduced compared to heuristic                |
| Convergence Speed (RL)      | Fast                       | Slower, higher variance                      |
| Value of RL                 | Low                        | High                                         |
| Total Cost Impact           | Minimal                    | Significant reduction                        |

</div>

**Managerial Implications**

The results provide several insights for maintenance decision-makers. 

First, for low-complexity production systems with limited interactions, simple heuristic preventive maintenance rules may be sufficient and cost-effective. Implementing advanced learning-based approaches in such environments may yield marginal improvements that do not justify additional complexity.
Second, as system complexity increases, maintenance decisions based solely on local information, such as machine age, become increasingly inefficient. Ignoring buffer interactions and downstream effects can lead to poorly timed maintenance actions and elevated production losses.
Finally, reinforcement learning-based preventive maintenance policies offer the greatest value in environments where maintenance decisions generate system-wide ripple effects. In such settings, data-driven policies that explicitly account for interactions among machines and buffers can significantly reduce corrective maintenance and improve overall system performance.


# 5. Conclusion

This study investigated preventive maintenance scheduling in repairable flow-line manufacturing systems using a deep reinforcement learning approach. By formulating the problem as a Markov Decision Process and applying a Double Deep Q-Network, the proposed framework integrates machine degradation, buffer dynamics, and stochastic production effects within a unified decision-making model.

Through a scenario-based comparison, the results demonstrate that the effectiveness of reinforcement learning-based preventive maintenance strongly depends on system complexity. In low-complexity environments, the learned policy converges to behavior similar to simple age-based heuristics, indicating limited marginal benefit from advanced learning approaches. In contrast, for high-complexity systems with strong machine interactions and nonlinear failure processes, the reinforcement learning agent learns nontrivial maintenance strategies that substantially reduce corrective maintenance and mitigate system-wide disruptions.

These findings suggest that reinforcement learning should be viewed as a targeted decision-support tool rather than a universal replacement for traditional maintenance rules. Its value is most pronounced in complex manufacturing systems where local maintenance decisions have far-reaching operational consequences. Future research may extend this framework to incorporate maintenance capacity constraints, partial observability, or multi-objective considerations to further enhance its practical applicability.

---

# 6. References

Ferreira Neto, W. A., Cavalcante, C. A. V., & Do, P. (2024). Deep reinforcement learning for maintenance optimization of a scrap-based steel production line. Reliability Engineering & System Safety, 249, Article 110199. https://doi.org/10.1016/j.ress.2024.110199

Geurtsen, M., Leenen, C., Adan, I., & Atan, Z. (2026). Deep reinforcement learning for optimal planning of production line maintenance with deterioration. Reliability Engineering & System Safety, 266(Part B), Article 111767. https://doi.org/10.1016/j.ress.2025.111767

Hung, YH., Shen, HY. & Lee, CY. Deep reinforcement learning-based preventive maintenance for repairable machines with deterioration in a flow line system. Ann Oper Res (2024).https://doi.org/10.1007/s10479-024-06207-x

Jianyu Su, Jing Huang, Stephen Adams, Qing Chang, and Peter A. Beling. 2022. Deep multi-agent reinforcement learning for multi-level preventive maintenance in manufacturing systems▪. Expert Syst. Appl. 192, C (Apr 2022). https://doi.org/10.1016/j.eswa.2021.116323

Ogunfowora, O., & Najjaran, H. (2023). Reinforcement and deep reinforcement learning-based solutions for machine maintenance planning, scheduling policies, and optimization. Journal of Manufacturing Systems, 70, 244–263. https://doi.org/10.1016/j.jmsy.2023.07.014
