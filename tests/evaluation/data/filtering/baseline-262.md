# Quantum approximate optimization algorithm 
_Usage estimate: 22 minutes on a Heron r3 processor (NOTE: This is an estimate only. Your runtime might vary.)_
## Learning outcomes 
  * How to map a classical combinatorial optimization problem (max-cut) to a quantum Hamiltonian
  * How to implement and run the Quantum Approximate Optimization Algorithm (QAOA) using IBM Quantum Compute Service sessions
  * How to scale a QAOA workflow from a small simulator example to utility-scale hardware execution


## Prerequisites 
  * [QAOA in depth](https://quantum.cloud.ibm.com/learning/en/courses/quantum-computing-in-practice/utility-scale-qaoa) — for a comprehensive treatment of the QAOA algorithm and applying it at utility scale


## Background 
The **Quantum Approximate Optimization Algorithm (QAOA)** is a hybrid quantum-classical iterative method for solving combinatorial optimization problems. In this tutorial, you will use QAOA to solve the **maximum-cut (max-cut)** problem — an NP-hard optimization problem with applications in clustering, network science, and statistical physics. Given a graph of nodes connected by edges, the goal is to partition the nodes into two sets such that the number of edges crossing the partition is maximized.
Combinatorial optimization also appears in sustainability challenges, such as planning delivery routes to reduce fuel use and allocating resources to limit material waste. The max-cut example introduces the process of expressing an objective as a cost Hamiltonian and exploring candidate solutions with QAOA. Applying this approach to a sustainability problem requires an objective and constraints that represent the intended application.
### From classical optimization to quantum circuits 
Max-cut can be expressed as a classical binary optimization problem. Each node is assigned a binary variable x_i \in \\{0, 1\\} indicating which set it belongs to. The objective is to maximize the number of edges where the endpoints are in different sets:
+−2.\max_{x \in \\{0,1\\}^n} \sum_{(i,j)} x_i + x_j - 2x_ix_j.
This is equivalently a **Quadratic Unconstrained Binary Optimization (QUBO)** problem of the form \min_x\, x^T Q x. Through a standard variable substitution (x_i \to (1 - Z_i)/2), the QUBO can be rewritten as a **cost Hamiltonian** whose ground state encodes the optimal solution. In general, this Hamiltonian has both quadratic and linear terms:
=+.H_C = \sum_{ij} Q_{ij} \, Z_i Z_j + \sum_i b_i \, Z_i.
In this tutorial, you do not carry out this substitution by hand. The [Optimization Mapper Qiskit addon](https://qiskit.github.io/qiskit-addon-opt-mapper/) (`qiskit-addon-opt-mapper`) provides a `Maxcut` application class that models the problem directly from a graph, a converter that turns the model into a QUBO, and a translator that maps the QUBO to . For the unweighted max-cut problem considered here, the linear coefficients vanish () and each edge contributes a \tfrac{1}{2} Z_i Z_j term, together with a constant offset that relates \langle H_C \rangle to the cut value. The same pipeline handles weighted graphs and other QUBO-expressible problems without changing the rest of the workflow.
### How QAOA works 
QAOA prepares candidate solutions by applying alternating layers of two operators to an initial superposition state H^{\otimes n}|0\rangle: the **cost operator** e^{-i\gamma_k H_C} and a **mixer operator** e^{-i\beta_k H_m}. The angles and are optimized in a classical feedback loop; the quantum computer evaluates the cost function, and a classical optimizer updates the parameters until convergence. This iterative loop runs within a Quantum Compute **session** , which keeps the quantum device reserved across iterations for lower latency.
For a deeper treatment of QAOA theory, including the full QUBO-to-Hamiltonian derivation, see the [QAOA course module](https://quantum.cloud.ibm.com/learning/en/courses/utility-scale-quantum-computing/variational-quantum-algorithms#1-introduction).
In this tutorial you will first solve max-cut on a small five-node graph, then scale the same workflow to a 100-node utility-scale problem on real hardware. _Note on plan access:_ This tutorial uses Quantum Compute [sessions](https://quantum.cloud.ibm.com/docs/en/guides/execution-modes#session-mode), which are only available on the Premium Plan. If you are on the Open Plan, you cannot run this tutorial as written; instead, you will need to swap `Session` for [job mode](https://quantum.cloud.ibm.com/docs/en/guides/execution-modes#job-mode) (as in, submit each iteration as an independent job rather than wrapping the optimization loop in with `Session(...)`). The workflow still runs, but each iteration incurs the full queue latency rather than reusing a reserved device. See [Overview of available plans](https://quantum.cloud.ibm.com/docs/en/guides/plans-overview) for more information.
## Requirements 
Before starting this tutorial, be sure you have the following installed:
  * Qiskit SDK v2.0 or later, with [visualization](https://quantum.cloud.ibm.com/docs/en/api/qiskit/visualization) support
  * Qiskit Runtime v0.22 or later (`pip install qiskit-ibm-runtime`)
  * Optimization Mapper Qiskit addon (`pip install qiskit-addon-opt-mapper`)


In addition, you will need access to an instance on [IBM Quantum® Platform](https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup).
## Setup 

```
import matplotlib.pyplot as plt
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict


from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler

from qiskit_addon_opt_mapper.applications import Maxcut
from qiskit_addon_opt_mapper.converters import OptimizationProblemToQubo
from qiskit_addon_opt_mapper.translators import to_ising
```

## Small-scale example 
This section walks through each step of the QAOA workflow on a small five-node max-cut instance. Despite the "small-scale" label, this example still runs on real IBM Quantum hardware — the code selects a backend with 127 or more qubits and executes the circuit there.
Initialize your problem by creating a graph with nodes.

```


graph = rx.PyGraph()
graph.add_nodes_from(np.arange(0, n_small, 1))








graph.add_edges_from(edge_list)
draw_graph(graph, node_size=600, with_labels=True)
```

Output:
### Step 1: Map classical inputs to a quantum problem 
Map the classical graph into quantum **circuits** and **operators**. As described in the [Background](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm#background), max-cut is a QUBO problem whose cost Hamiltonian has a ground state that encodes the optimal cut, and QAOA uses a parametrized ansatz circuit to prepare candidate ground states of . Use the Optimization Mapper addon to go from the graph to in three steps: model the problem, convert it to a QUBO, and translate the QUBO to an Ising Hamiltonian.
#### Model the problem 
The `Maxcut` application class accepts the rustworkx graph directly and returns an `OptimizationProblem` whose objective is the cut value to maximize. Each edge contributes x_i + x_j - 2 x_i x_j to the objective, exactly as in the [Background](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm#background).

```
maxcut = Maxcut(graph)
maxcut_problem = maxcut.to_optimization_problem()
print(maxcut_problem.prettyprint())
```

Output:

```
Problem name: Max-cut

Maximize
  -2*x_0*x_1 - 2*x_0*x_2 - 2*x_0*x_4 - 2*x_1*x_2 - 2*x_2*x_3 - 2*x_3*x_4 + 3*x_0
  + 2*x_1 + 3*x_2 + 2*x_3 + 2*x_4

Subject to
  No constraints

  Binary variables (5)
    x_0 x_1 x_2 x_3 x_4


```

#### Build the cost Hamiltonian 
Convert the problem to a QUBO with `OptimizationProblemToQubo`, then translate the QUBO to an Ising Hamiltonian with `to_ising`. The converter turns the maximization into a minimization of the negated cut value, and the translator applies the x_i \to (1 - Z_i)/2 substitution. It returns as a `SparsePauliOp` together with a scalar `offset` such that QUBO value=⟨⟩+offset\text{QUBO value} = \langle H_C \rangle + \text{offset}. Because the QUBO minimizes the negated cut value, the cut value of any bitstring is −(⟨⟩+offset)-(\langle H_C \rangle + \text{offset}).

```
qubo = OptimizationProblemToQubo().convert(maxcut_problem)
cost_hamiltonian, offset = to_ising(qubo)
print("Cost Function Hamiltonian:", cost_hamiltonian)
print("Offset:", offset)
```

Output:

```
Cost Function Hamiltonian: SparsePauliOp(['IIIZZ', 'IIZIZ', 'ZIIIZ', 'IIZZI', 'IZZII', 'ZZIII'],
              coeffs=[0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])
Offset: -3.0

```

#### Build the QAOA ansatz circuit 
Use `QAOAAnsatz` to construct the parametrized QAOA circuit from the cost Hamiltonian. Here we use `reps=2` (two QAOA layers, four parameters: \beta_0, \beta_1, \gamma_0, \gamma_1).

```
circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
circuit.measure_all()

circuit.draw("mpl")
```

Output:

```
circuit.parameters
```

Output:

```
ParameterView([ParameterVectorElement(β[0]), ParameterVectorElement(β[1]), ParameterVectorElement(γ[0]), ParameterVectorElement(γ[1])])

```

### Step 2: Optimize problem for quantum hardware execution 
Transpile the abstract circuit into hardware-native instructions. This step handles qubit mapping, gate decomposition, routing, and error suppression. See the transpilation [documentation](https://quantum.cloud.ibm.com/docs/en/guides/transpile) for more information.

```
service = QiskitRuntimeService()
backend = service.least_busy(
    operational=True, simulator=False, min_num_qubits=127

print(backend)

# Create pass manager for transpilation. Level 3 is the most aggressive
# preset: slower to transpile, but produces shorter circuits that are
# more robust to hardware noise.
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

candidate_circuit = pm.run(circuit)
candidate_circuit.draw("mpl", fold=False, idle_wires=False)
```

Output:

```
<IBMBackend('ibm_pittsburgh')>

```

### Step 3: Execute using Qiskit primitives 
The QAOA optimization loop runs inside a Quantum Compute [session](https://quantum.cloud.ibm.com/docs/en/guides/execution-modes) to keep the device reserved across iterations. An Estimator evaluates \langle H_C \rangle at each step, and a classical optimizer (COBYLA) updates the parameters until convergence.
Define initial parameters and run the optimization loop:

```
# QAOA doesn't prescribe principled default angles — any bounded choice
# works as a warm start for problems this small. beta and gamma are
# periodic (beta in [0, pi] and gamma in [0, 2*pi] modulo the underlying
# Pauli-rotation periods), and pi/2 and pi are just midpoints of those
# ranges. For harder problems you would typically warm start from known
# good angles or transfer parameters from smaller instances.
initial_gamma = np.pi
initial_beta = np.pi /
init_params = [initial_beta, initial_beta, initial_gamma, initial_gamma]
```


```
def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

= (ansatz, isa_hamiltonian, params)
= estimator.run([pub])

    results = job.result()[0]
= results.data.evs

    objective_func_vals.append(cost)


```


```
objective_func_vals =# Global variable
with Session(backend=backend) as session:
    # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `session=`
    estimator = Estimator(mode=session)
    estimator.options.default_shots = 1000

    # Set simple error suppression/mitigation options
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"
    estimator.options.environment.job_tags ="TUT_QAOA"]

    result = minimize(
        cost_func_estimator,
        init_params,
=(candidate_circuit, cost_hamiltonian, estimator),
        method="COBYLA",


    print(result)
```

Output:

```
 message: Return from COBYLA because the trust region radius reaches its lower bound.
 success: True
  status: 0
     fun: -1.0164225323839062
       x: [ 1.404e+00  1.174e+00  4.036e+00  2.969e+00]
    nfev: 30
   maxcv: 0.0

```

The optimizer was able to reduce the cost and find better parameters for the circuit.
A smoothly decreasing curve that plateaus is the signature of convergence. A noisy, non-monotonic curve usually indicates that something upstream needs attention; common causes are too few shots per evaluation (high estimator variance), poor initial parameters, or a circuit whose depth is dominated by hardware noise. COBYLA is derivative-free and fairly robust to moderate noise, but when the noise swamps the actual cost improvements per step, its linear-approximation model can no longer tell real descent from random jitter and the optimizer wanders.

```
plt.figure(figsize=(126))
plt.plot(objective_func_vals)
plt.xlabel("Iteration")
plt.ylabel("Cost")

```

Output:
Assign the optimized parameters and sample the final distribution using the Sampler primitive.

```
optimized_circuit = candidate_circuit.assign_parameters(result.x)
optimized_circuit.draw("mpl", fold=False, idle_wires=False)
```

Output:

```
# If using qiskit-ibm-runtime<0.24.0, change `mode=` to `backend=`
sampler = Sampler(mode=backend)
sampler.options.default_shots = 10000

# Set simple error suppression/mitigation options
sampler.options.dynamical_decoupling.enable = True
sampler.options.dynamical_decoupling.sequence_type = "XY4"
sampler.options.twirling.enable_gates = True
sampler.options.twirling.num_randomizations = "auto"

sampler.options.environment.job_tags ="TUT_QAOA"]

pub = (optimized_circuit,)
job = sampler.run([pub], shots=int(1e4))
counts_int = job.result()[0].data.meas.get_int_counts()
counts_bin = job.result()[0].data.meas.get_counts()
shots = sum(counts_int.values())
final_distribution_int =key: val / shots for key, val in counts_int.items()}
final_distribution_bin =key: val / shots for key, val in counts_bin.items()}
print(final_distribution_int)
```

Output:

```
{22: 0.1003, 15: 0.008, 9: 0.0934, 4: 0.02, 31: 0.004, 17: 0.0288, 18: 0.0327, 14: 0.0305, 10: 0.0358, 28: 0.0128, 5: 0.0868, 20: 0.09, 11: 0.0908, 13: 0.0321, 19: 0.0298, 27: 0.0186, 8: 0.0077, 30: 0.0211, 23: 0.0098, 26: 0.0839, 12: 0.0254, 21: 0.0315, 16: 0.0082, 25: 0.01, 6: 0.0131, 1: 0.0205, 2: 0.0145, 29: 0.0137, 7: 0.0059, 3: 0.0101, 24: 0.0078, 0: 0.0024}

```

### Step 4: Post-process and return result in desired classical format 
Extract the most likely bitstring from the sampled distribution. This represents the best cut found by QAOA. The `interpret` method of the `Maxcut` class decodes the bitstring into the two node sets of the partition.

```
# auxiliary functions to sample most likely bitstring
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    returnint(digit) for digit in result]


keys = list(final_distribution_int.keys())
values = list(final_distribution_int.values())
most_likely = keys[np.argmax(np.abs(values))]
most_likely_bitstring = to_bitstring(most_likely, len(graph))
most_likely_bitstring.reverse()

print("Result bitstring:", most_likely_bitstring)

partition = maxcut.interpret(np.array(most_likely_bitstring))
print("Partition:", partition)
```

Output:

```
Result bitstring: [0, 1, 1, 0, 1]
Partition: [[0, 3], [1, 2, 4]]

```


```
plt.rcParams.update({"font.size"10})
final_bits = final_distribution_bin
values = np.abs(list(final_bits.values()))
top_4_values = sorted(values, reverse=True)[:4]
positions = []
for value in top_4_values:
    positions.append(np.where(values == value)[0])
fig = plt.figure(figsize=(116))
ax = fig.add_subplot(111)
plt.xticks(rotation=45)
plt.title("Result Distribution")
plt.xlabel("Bitstrings (reversed)")
plt.ylabel("Probability")
ax.bar(list(final_bits.keys()), list(final_bits.values()), color="tab:grey")
forin positions:
.get_children()[int(p[0])].set_color("tab:purple")

```

Output:
#### Visualize best cut 
From the optimal bitstring, you can then visualize this cut on the original graph.

```
# auxiliary function to plot graphs
def plot_result(G,):
    colors ="tab:grey" if== else "tab:purple" forin x]
, _default_axes = rx.spring_layout(G), plt.axes(frameon=True)
.visualization.mpl_draw(
        G, node_color=colors, node_size=100, alpha=0.8, pos=pos



plot_result(graph, most_likely_bitstring)
```

Output:
Now, calculate the value of the cut. The objective of the `OptimizationProblem` is the cut value itself, so evaluate it directly on the bitstring. You can also compare it with the cut value implied by the optimizer's final cost, −(⟨⟩+offset)-(\langle H_C \rangle + \text{offset}), which is an average over the whole sampled distribution rather than the value of a single bitstring.

```
cut_value = maxcut_problem.objective.evaluate(np.array(most_likely_bitstring))
print("The value of the cut is:", cut_value)
print("Cut value implied by the optimized cost:"-(result.fun + offset))
```

Output:

```
The value of the cut is: 5.0
Cut value implied by the optimized cost: 4.016422532383906

```

For a graph this small, the true optimum is easy to brute-force, so you can double-check the results by comparing the QAOA result against the exact answer.

```
# Classical baseline: enumerate all 2**n_small bitstrings and take the best cut.
def brute_force_max_cut(problem tuple[int, list[int]]:
= problem.get_num_vars()

    best_x: list[int] []
in range(2**n):
= np.array([(i  forin range(n)])
= problem.objective.evaluate(x)
 cut  best_cut:
            best_cut = int(cut)
=.tolist()
    return best_cut, best_x


classical_best, classical_x = brute_force_max_cut(maxcut_problem)
print(f"Classical optimum (brute force): {classical_best}")
print(f"QAOA cut value:                  {cut_value}")
```

Output:

```
Classical optimum (brute force): 5
QAOA cut value:                  5.0

```

## Large-scale hardware example 
You have access to many devices with over 100 qubits on IBM Quantum Platform. Select one on which to solve max-cut on a 100-node weighted graph. This is a "utility-scale" problem. The workflow follows the same steps as above, applied to a much larger graph.
### End-to-end workflow at utility scale 
All four steps are shown below, applied to the 100-node graph. The structure is the same as the small-scale walkthrough: map, transpile, execute, post-process — but with a larger problem and split across the four cells below for clarity.

```
# Precomputed parity lookup table: _PARITY[b] = +1 if popcount(b) is even, else -1.
# We use this to vectorize expectation-value evaluation across all Pauli terms.
_PARITY = np.array(
-1 if bin(i).count("1"% else forin range(256)],
    dtype=np.complex128,



def evaluate_sparse_pauli(state: int, observable: SparsePauliOp)  complex:
    """Expectation value of a SparsePauliOp on a single computational-basis state.

    For a Z-only observable (which QAOA cost Hamiltonians are, after the
    QUBO-to-Hamiltonian mapping), the eigenvalue of each Pauli term on a
    computational-basis state is simply (-1)**popcount(z_mask AND state),
    i.e., the parity of the bitwise-AND of the term's Z-support and the
    measured bitstring.

    This routine packs the Z-support of every Pauli term into bytes, ANDs
    them against the measured state in a single vectorized op, and looks up
    the parity in _PARITY. For a 100-qubit / ~hundreds-of-terms Hamiltonian
    over 10_000 samples, this is dramatically faster than calling
    SparsePauliOp.expectation_value per sample.

    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes = np.frombuffer(
        state.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8

    reduced = np.bitwise_xor.reduce(packed_uint8  state_bytes, axis=1)
    return np.sum(observable.coeffs * _PARITY[reduced])


def best_solution(samples, hamiltonian):
    """Return the sampled bitstring (as int) with the lowest Hamiltonian cost."""
    min_cost = float("inf")
    min_sol = None
 bit_str in samples.keys():
        candidate_sol = int(bit_str)
= evaluate_sparse_pauli(candidate_sol, hamiltonian).real
 fval  min_cost:
            min_cost = fval
            min_sol = candidate_sol
    return min_sol


def _plot_cdf(objective_values: dict, ax, color):
    x_vals = sorted(objective_values.keys(), reverse=True)
    y_vals = np.cumsum([objective_values[x] forin x_vals])
.plot(x_vals, y_vals, color=color)


def plot_cdf(dist, ax, title):
    _plot_cdf(dist, ax, "C1")
.vlines(min(list(dist.keys())), 01"C1", linestyle="--")
.set_title(title)
.set_xlabel("Objective function value")
.set_ylabel("Cumulative distribution function")
.grid(alpha=0.3)


def samples_to_objective_values(samples, hamiltonian):
    """Convert the samples to values of the objective function."""
    objective_values = defaultdict(float)
 bit_str, prob in samples.items():
        candidate_sol = int(bit_str)
= evaluate_sparse_pauli(candidate_sol, hamiltonian).real
        objective_values[fval] += prob
    return objective_values
```

**Step 1** : Build the graph from the backend's coupling map, then map it to the cost Hamiltonian with the Optimization Mapper addon and build the ansatz.

```
# Step 1: build the 100-node graph, cost Hamiltonian, and QAOA ansatz.

graph_100 = rx.PyGraph()
graph_100.add_nodes_from(np.arange(0, n_large, 1))

for,in backend.coupling_map:
    # The coupling map lists both directions of each coupler, so sort
    # the endpoints to avoid adding the same edge twice.
 n_large and n_large:
.add((min(u, v), max(u, v), 1.0))
graph_100.add_edges_from(sorted(elist))

maxcut_100 = Maxcut(graph_100)
maxcut_problem_100 = maxcut_100.to_optimization_problem()
qubo_100 = OptimizationProblemToQubo().convert(maxcut_problem_100)
cost_hamiltonian_100, offset_100 = to_ising(qubo_100)

circuit_100 = QAOAAnsatz(cost_operator=cost_hamiltonian_100, reps=1)
circuit_100.measure_all()
```

**Step 2** : Transpile for the selected hardware backend.

```
# Step 2: transpile for hardware.
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
candidate_circuit_100 = pm.run(circuit_100)
```

**Step 3** : Run the QAOA optimization loop inside a session, then sample.

```
# Step 3: run the QAOA optimization loop on the device, then sample the
# final distribution with the optimized parameters.
initial_gamma = np.pi
initial_beta = np.pi /
init_params = [initial_beta, initial_gamma]

objective_func_vals =# Global variable
with Session(backend=backend) as session:
    estimator = Estimator(mode=session)
    estimator.options.default_shots = 1000

    # Set simple error suppression/mitigation options
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"
    estimator.options.environment.job_tags ="TUT_QAOA"]

    result = minimize(
        cost_func_estimator,
        init_params,
=(candidate_circuit_100, cost_hamiltonian_100, estimator),
        method="COBYLA",

    print(result)

# Assign optimal parameters and sample the final distribution.
optimized_circuit_100 = candidate_circuit_100.assign_parameters(result.x)

sampler = Sampler(mode=backend)
sampler.options.default_shots = 10000

# Set simple error suppression/mitigation options
sampler.options.dynamical_decoupling.enable = True
sampler.options.dynamical_decoupling.sequence_type = "XY4"
sampler.options.twirling.enable_gates = True
sampler.options.twirling.num_randomizations = "auto"

# Add a unique tag to the job execution
sampler.options.environment.job_tags ="TUT_QAOA"]

pub = (optimized_circuit_100,)
job = sampler.run([pub], shots=int(1e4))

counts_int = job.result()[0].data.meas.get_int_counts()
shots = sum(counts_int.values())
final_distribution_100_int =
: val / shots for key, val in counts_int.items()

```

Output:

```
 message: Return from COBYLA because the trust region radius reaches its lower bound.
 success: True
  status: 0
     fun: -4.214640355142735
       x: [ 2.586e+00  4.161e+00]
    nfev: 26
   maxcv: 0.0

```

**Step 4** : Post-process the sampled distribution to extract the best cut.

```
# Step 4: find the best-cost sample and evaluate its cut value.
best_sol_100 = best_solution(final_distribution_100_int, cost_hamiltonian_100)
best_sol_bitstring_100 = to_bitstring(int(best_sol_100), len(graph_100))
best_sol_bitstring_100.reverse()

print("Result bitstring:", best_sol_bitstring_100)

cut_value_100 = maxcut_problem_100.objective.evaluate(
array(best_sol_bitstring_100)

print("The value of the cut is:", cut_value_100)
```

Output:

```
Result bitstring: [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
The value of the cut is: 78.0

```

Check that the cost minimized in the optimization loop has converged, and visualize results.

```
# Plot convergence
plt.figure(figsize=(126))
plt.plot(objective_func_vals)
plt.xlabel("Iteration")
plt.ylabel("Cost")


# Visualize the cut
plot_result(graph_100, best_sol_bitstring_100)

# Plot cumulative distribution function
result_dist = samples_to_objective_values(
    final_distribution_100_int, cost_hamiltonian_100

fig,= plt.subplots(11, figsize=(86))
plot_cdf(result_dist, ax, backend.name)
```

Output:
## Next steps 
If you found this work interesting, you might be interested in the following material:
Recommendations
  * [Advanced techniques for QAOA](https://quantum.cloud.ibm.com/docs/en/tutorials/advanced-techniques-for-qaoa) — explores advanced strategies for improving QAOA performance
  * [Multi-objective optimization challenge](https://github.com/qiskit-community/qdc-challenges-2025/blob/main/challenges/Track_B/qmoo/qmoo_qdc25.ipynb) — put your skills to the test with this community challenge on multi-objective quantum optimization
  * [Transpilation documentation](https://quantum.cloud.ibm.com/docs/en/guides/transpile) for fine-tuning circuit optimization
  * [Error suppression and mitigation](https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-and-suppression-techniques) for improving hardware results


Was this page helpful?
Report a bug, typo, or request content on [GitHub](https://github.com/Qiskit/documentation/issues/new/choose).
AllDocumentationTutorialsAPI referencesLearning
No results found
Please try a different search query or [browse all documentation](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm).
