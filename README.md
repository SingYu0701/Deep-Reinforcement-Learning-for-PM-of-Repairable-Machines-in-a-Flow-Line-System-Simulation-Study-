# Deep-Reinforcement-Learning-for-PM-of-Repairable-Machines-in-a-Flow-Line-System-Simulation-Study-
Final report of Operations Research Applications, 2025 @ NCKU Institute of Manufacturing Information and Systems 

## 1. Background and Motivation

### 1.1 Motivation
Effective production and maintenance scheduling is critical for minimizing production losses and maintenance costs in manufacturing systems. Machine failures and buffer overflows can result in significant financial losses and decreased operational efficiency. By optimizing maintenance policies and operational scheduling, manufacturers can improve productivity, reduce waste, and enhance resource utilization. This problem is especially relevant to industries like semiconductor, automotive, or food processing, where downtime is costly.

### 1.2 Background
Machines in modern production systems experience stochastic deterioration and random failures. Preventive maintenance (PM) can reduce the probability of failure but incurs maintenance costs and temporary downtime. Corrective maintenance (CM) is performed after a failure and typically has higher costs and production loss. Buffers store work-in-progress (WIP), but exceeding buffer limits results in waste. The trade-off between maintenance actions, production continuity, and buffer management is a classical operations research problem.

### 1.3 Problem Definition
This study proposes a reinforcement learning framework using Double Deep Q-Networks (DDQN) to optimize production and maintenance scheduling for a single-machine system with stochastic WIP arrivals. The objective is to maximize cumulative reward by balancing production output, maintenance cost, and WIP losses.

---

## 2. Methodology

### 2.1 Method Justification
**Assumptions:**
- Single machine with 2-stage production (T1 → T2 → T3).  
- Machine deterioration is proportional to operating time; failure probability increases as Health Index (HI) decreases.  
- PM and CM restore machine HI to maximum.  
- Buffer level is limited; WIP arrivals follow a Poisson distribution.

**Limitations:**
- Single-machine system; multi-machine extension increases state space.  
- Maintenance times are simplified (fixed PM/CM duration).  

**Why DDQN?**
- Traditional optimization methods struggle with stochastic failures and discrete multi-stage states.  
- DDQN handles high-dimensional, discrete state-action spaces and learns an optimal policy through simulation.  

**Pros and Cons:**
- **Pros:** Adaptive to stochastic WIP arrivals, learns from experience, incorporates long-term rewards.  
- **Cons:** Requires simulation and training; convergence can be slow for large state spaces.

### 2.2 Methodology
The problem is modeled as a **Markov Decision Process (MDP)**:

**State space:**  

 $$s_t = [B_t, M_t, HI_t]$$

- `B_t`: buffer level at time t  
- `M_t`: remaining maintenance time (0 if idle)  
- `HI_t`: machine health index (0–1)

**Action space:**  

$$a_t ∈ [ do nothing, PM, CM ] $$

**Reward function:**  

 $$r_t = - (production loss) - (WIP waste) - (maintenance cost) - (invalid actions)$$


**Transition dynamics:**  
- HI decreases during operation; failure occurs probabilistically depending on HI.  
- PM/CM restore HI and consume maintenance time.  
- WIP arrives stochastically; buffer overflow causes WIP loss.

**Learning algorithm: Double Deep Q-Network (DDQN)**  
- Q-network estimates action-value function `Q(s,a)`  
- Target network stabilizes training  
- Experience replay buffer stores transitions `(s,a,r,s')` for minibatch updates  
- Action masking prevents invalid actions in given states

---

## 3. Data Collection and Analysis

### 3.1 Data Collection
Simulation is used to generate data:
<div align="center">
 
| Parameter | Value |
|-----------|-------|
| WIP arrival | Poisson(0.5), capped at 1 per step |
| Machine deterioration | HI decreases by 0.05 per operation step |
| PM time | 2 steps |
| CM time | 4 steps |
| Buffer max | 6 |
| Training episodes | 300 |
| Max steps per episode | 50 |

</div>
Simulation provides realistic stochastic behavior while remaining computationally feasible.

### 3.2 Analysis
Python & PyTorch are used for training the DDQN agent.

**Environment:** `MachineEnv()`  
**Agent:** `DDQNAgent(state_size=3, action_size=3)`

**Hyperparameters:**
- Replay buffer size: 1000  
- Batch size: 16  
- Learning rate: 0.001  
- Discount factor γ = 0.99  
- Epsilon-greedy exploration with decay

**Training procedure:**
1. Select action using epsilon-greedy policy (mask illegal actions).  
2. Step the environment, observe reward and next state.  
3. Store transition in replay buffer.  
4. Sample minibatch and update Q-network.  
5. Update target network every episode.

**Python Illustration (Reward and Loss over Episodes):**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(reward_hist)
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.subplot(1,2,2)
plt.plot(loss_hist)
plt.title("Loss per Step")
plt.xlabel("Training Steps")
plt.ylabel("MSE Loss")

plt.show()
```

### 3.3 Results and Managerial Implications

The agent learns to perform PM before failures, avoiding costly CM and production loss.

WIP buffer management is implicitly learned: avoid overproduction when buffer is full.

Managerial insight: Reinforcement learning provides adaptive maintenance policies that balance cost and production continuity without explicit optimization models.

### 4. Conclusion

DDQN-based reinforcement learning effectively solves production-maintenance scheduling under stochastic WIP arrivals. The trained agent maximizes cumulative reward by balancing production, maintenance, and WIP loss. This approach is flexible, data-driven, and can be extended to multi-machine or more complex production systems.

### 5. References

Hung, YH., Shen, HY. & Lee, CY. Deep reinforcement learning-based preventive maintenance for repairable machines with deterioration in a flow line system. Ann Oper Res (2024).https://doi.org/10.1007/s10479-024-06207-x

Jianyu Su, Jing Huang, Stephen Adams, Qing Chang, and Peter A. Beling. 2022. Deep multi-agent reinforcement learning for multi-level preventive maintenance in manufacturing systems▪. Expert Syst. Appl. 192, C (Apr 2022). https://doi.org/10.1016/j.eswa.2021.116323

Liu, Y., Wang, W., Hu, Y., Hao, J., Chen, X., & Gao, Y. (2019). Multi-agent game abstraction via graph attention neural network. arXiv preprint arXiv:1911.10715. https://doi.org/10.48550/arXiv.1911.10715
