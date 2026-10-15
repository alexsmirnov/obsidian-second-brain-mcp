{{Short description|Optimization algorithms using quantum computing}}
{{Use American English|date=January 2019}}

'''Quantum optimization algorithms''' are [[quantum algorithms]] that are used to solve optimization problems.<ref>{{cite journal|last1=Moll|first1=Nikolaj|last2=Barkoutsos|first2=Panagiotis|last3=Bishop|first3=Lev S.|last4=Chow|first4=Jerry M.|last5=Cross|first5=Andrew|last6=Egger|first6=Daniel J.|last7=Filipp|first7=Stefan|last8=Fuhrer|first8=Andreas|last9=Gambetta|first9=Jay M.|last10=Ganzhorn|first10=Marc|last11=Kandala|first11=Abhinav|last12=Mezzacapo|first12=Antonio|last13=Müller|first13=Peter|last14=Riess|first14=Walter|last15=Salis|first15=Gian|last16=Smolin|first16=John|last17=Tavernelli|first17=Ivano|last18=Temme|first18=Kristan|title=Quantum optimization using variational algorithms on near-term quantum devices|journal=Quantum Science and Technology|date=2018|volume=3|issue=3|page=r 030503|doi=10.1088/2058-9565/aab822|arxiv=1710.01022|bibcode=2018QS&T....3c0503M|s2cid=56376912}}</ref> [[Mathematical optimization]] deals with finding the best solution to a problem (according to some criteria) from a set of possible solutions. Mostly, the optimization problem is formulated as a minimization of a target functional. For combinatorial optimization problems no proven exponential speed up has ever been found. Most provable results are similar to [[Grover's algorithm]] and have the potential for a modest polynomial speed up.  

==Quantum data fitting==
[[curve fitting|Data fitting]] is a process of constructing a [[mathematical function]] that best fits a set of data points. The fit's quality is measured by some criteria, usually the distance between the function and the data points.

===Quantum least squares fitting===
One of the most common types of data fitting is solving the [[least squares]] problem, minimizing the sum of the squares of differences between the data points and the fitted function.

The algorithm is given <math> N </math> input data points <math> (x_1, y_1), (x_2, y_2), ... , (x_N, y_N)</math> and <math> M </math> [[continuous function]]s <math> f_{1}, f_{2}, ... , f_{M} </math>. The algorithm finds and gives as output a continuous function <math> f_{\vec{\lambda}} </math> that is a [[linear combination]] of <math> f_j</math>:

:<math>
f_{\vec{\lambda}}(x) = \sum_{j=1}^M f_{j}(x)\lambda_{j}
</math>
 
In other words, the algorithm finds the [[complex number|complex]] coefficients <math>\lambda_j </math>, and thus the vector <math> \vec{\lambda} = (\lambda_1, \lambda_2, ..., \lambda_M) </math>.

The algorithm is aimed at minimizing the error, which is given by:
:<math>
E=\sum_{i=1}^N 
\left\vert f_{\vec{\lambda}}(x_i)-y_i \right\vert^2 = \sum_{i=1}^N \left\vert \sum_{j=1}^M
f_{j}(x_i)\lambda_{j}-y_i \right\vert^2 = \left\vert F\vec{\lambda}-\vec{y} \right\vert^2
</math> 
where <math> F </math> is defined to be the following matrix:

:<math>{F} = \begin{pmatrix}
f_{1}(x_1) & \cdots & f_{M}(x_1) \\
f_{1}(x_2) & \cdots & f_{M}(x_2) \\
\vdots & \ddots & \vdots \\
f_{1}(x_N) & \cdots & f_{M}(x_N) \\
\end{pmatrix}</math>

The quantum least-squares fitting algorithm<ref>{{cite journal|last1=Wiebe|first1=Nathan|last2=Braun|first2=Daniel|last3=Lloyd|first3=Seth|title=Quantum Algorithm for Data Fitting|journal=Physical Review Letters|date=2 August 2012|volume=109|issue=5|article-number=050505|arxiv=1204.5242|doi=10.1103/PhysRevLett.109.050505|pmid=23006156|bibcode=2012PhRvL.109e0505W|s2cid=118439810 }}</ref> makes use of a version of Harrow, Hassidim, and Lloyd's [[quantum algorithm for linear systems of equations]] (HHL), and outputs the coefficients <math> \lambda_j </math> and the fit quality estimation <math> E </math>. It consists of three subroutines: an algorithm for performing a pseudo-[[matrix inversion|inverse]] operation, one routine for the fit quality estimation, and an algorithm for learning the fit parameters.

Because the quantum algorithm is mainly based on the HHL algorithm, it suggests an exponential improvement<ref>{{cite journal|last1=Montanaro|first1=Ashley|title=Quantum algorithms: an overview|journal=[[npj Quantum Information]] |date=12 January 2016|volume=2|issue=1 |article-number=15023|arxiv=1511.04206|doi=10.1038/npjqi.2015.23|bibcode=2016npjQI...215023M|s2cid=2992738}}</ref> in the case where <math> F</math> is [[sparse matrix|sparse]] and the [[condition number]] (namely, the ratio between the largest and the smallest [[eigenvalues]]) of both <math> F F^\dagger </math> and <math> F^\dagger F </math> is small.

==Quantum semidefinite programming==
[[Semidefinite programming]] (SDP) is an optimization subfield dealing with the optimization of a linear [[objective function]] (a user-specified function to be minimized or maximized), over the intersection of the [[cone]] of [[Positive semidefinite matrix|positive semidefinite matrices]] with an [[affine space]]. The objective function is an [[inner product]] of a matrix <math> C </math> (given as an input) with the variable <math> X </math>. Denote by <math>\mathbb{S}^n</math> the space of all <math>n \times n</math> symmetric matrices. The variable <math> X </math> must lie in the (closed convex) cone of positive semidefinite symmetric matrices <math>\mathbb{S}^{n}_+</math>. The inner product of two matrices is defined as:

<math>
  \langle A,B\rangle_{\mathbb{S}^n} = {\rm tr}(A^T B) = \sum_{i=1,j=1}^n
  A_{ij}B_{ij}.
</math>

The problem may have additional constraints (given as inputs), also usually formulated as inner products. Each constraint forces the inner product of the matrices <math> A_k </math> (given as an input) with the optimization variable <math> X </math> to be smaller than a specified value <math> b_k </math> (given as an input). Finally, the SDP problem can be written as:
:<math>
\begin{array}{rl}
{\displaystyle\min_{X \in \mathbb{S}^n}} & \langle C, X \rangle_{\mathbb{S}^n} \\
\text{subject to} & \langle A_k, X \rangle_{\mathbb{S}^n} \leq b_k, \quad k = 1,\ldots,m \\
& X \succeq 0
\end{array}
</math>

The best classical algorithm is not known to unconditionally run in [[polynomial time]]. The corresponding feasibility problem is known to either lie outside of the union of the complexity classes NP and co-NP, or in the intersection of NP and co-NP.<ref>{{Cite journal|doi = 10.1007/BF02614433|title = An exact duality theory for semidefinite programming and its complexity implications|year = 1997|last1 = Ramana|first1 = Motakuri V.|journal = Mathematical Programming|volume = 77|pages = 129–162|s2cid = 12886462}}</ref>

===The quantum algorithm===
The algorithm inputs are <math> A_1 ... A_m, C , b_1 ... b_m</math> and parameters regarding the solution's [[Trace class|trace]], precision and optimal value (the objective function's value at the optimal point).

The quantum algorithm<ref>{{cite arXiv|last1=Brandao|first1=Fernando G. S. L.|last2=Svore|first2=Krysta|author2-link= Krysta Svore |title=Quantum Speed-ups for Semidefinite Programming|eprint=1609.05537|class=quant-ph|year=2016}}</ref> consists of several iterations. In each iteration, it solves a [[Mathematical optimization#Feasibility problem|feasibility problem]], namely, finds any solution satisfying the following conditions (giving a threshold <math>t</math>):

:<math>
\begin{array}{lr}
\langle C, X \rangle_{\mathbb{S}^n} \leq t \\
\langle A_k, X \rangle_{\mathbb{S}^n} \leq b_k, \quad k = 1,\ldots,m \\
X \succeq 0
\end{array}
</math>

In each iteration, a different threshold <math>t</math> is chosen, and the algorithm outputs either a solution <math>X</math> such that <math> \langle C, X\rangle_{\mathbb{S}^n} \leq t</math> (and the other constraints are satisfied, too) or an indication that no such solution exists. The algorithm performs a [[binary search]] to find the minimal threshold <math>t</math> for which a solution <math>X</math> still exists: this gives the minimal solution to the SDP problem.

Quantum SDP algorithms provide a generic quadratic improvement in certain oracle or query models.<ref>{{cite arXiv
 | last1 = Brandão | first1 = Fernando G. S. L.
 | last2 = Svore | first2 = Krysta M.
 | title = Quantum Speed-ups for Semidefinite Programming
 | eprint = 1609.05537
 | class = quant-ph
 | year = 2016
}}</ref> For low-rank SDPs, algorithms can have polylogarithmic dependence on the matrix dimension n when the input matrices are supplied in a fully quantum input model, such as quantum states, and when only a succinct or sampled representation of the solution is required.<ref>{{cite arXiv
 | last1 = Brandão | first1 = Fernando G. S. L.
 | last2 = Kalev | first2 = Amir
 | last3 = Li | first3 = Tongyang
 | last4 = Lin | first4 = Cedric Yen-Yu
 | last5 = Svore | first5 = Krysta M.
 | last6 = Wu | first6 = Xiaodi
 | title = Quantum SDP Solvers: Large Speed-ups, Optimality, and Applications to Quantum Learning
 | eprint = 1710.02581
 | class = quant-ph
 | year = 2017
}}</ref> However, comparable low-rank and sampling-access assumptions also permit ''quantum-inspired'' classical algorithms with polylogarithmic dependence on n; therefore, this does not establish a general exponential quantum advantage for low-rank SDPs given explicitly as classical input.<ref>{{cite arXiv
 | last1 = Chia | first1 = Nai-Hui
 | last2 = Li | first2 = Tongyang
 | last3 = Lin | first3 = Han-Hsuan
 | last4 = Wang | first4 = Chunhao
 | title = Quantum-inspired sublinear algorithm for solving low-rank semidefinite programming
 | eprint = 1901.03254
 | class = quant-ph
 | year = 2019
}}</ref>


==Quantum combinatorial optimization==
The [[combinatorial optimization]] problem is aimed at finding an optimal object from a [[finite set]] of objects. The problem can be phrased as a maximization of an [[objective function]] which is a sum of [[Boolean function]]s. Each Boolean function <math>\,C_\alpha \colon \lbrace {0,1 \rbrace}^n \rightarrow \lbrace {0,1} \rbrace </math> gets as input the <math>n</math>-bit string <math>z = z_1 z_2 \ldots z_n</math> and gives as output one bit (0 or 1). The combinatorial optimization problem of <math>n</math> bits and <math>m</math> clauses is finding an <math>n</math>-bit string <math>z</math> that maximizes the function
:<math>
C(z) = \sum_{\alpha=1}^m C_\alpha(z)
</math>

[[Approximation algorithm|Approximate optimization]] is a way of finding an approximate solution to an optimization problem, which is often [[NP-hard]]. The approximated solution of the combinatorial optimization problem is a string <math> z </math> that is close to maximizing <math> C(z) </math>.

===Quantum approximate optimization algorithm===
For combinatorial optimization, the '''quantum approximate optimization algorithm''' (QAOA)<ref>{{cite arXiv|last1=Farhi|first1=Edward|last2=Goldstone|first2=Jeffrey|last3=Gutmann|first3=Sam|title=A Quantum Approximate Optimization Algorithm|eprint=1411.4028|class=quant-ph|year=2014}}</ref> briefly had a better approximation ratio than any known [[polynomial time]] classical algorithm (for a certain problem),<ref>{{cite arXiv|last1=Farhi|first1=Edward|last2=Goldstone|first2=Jeffrey|last3=Gutmann|first3=Sam|title=A Quantum Approximate Optimization Algorithm Applied to a Bounded Occurrence Constraint Problem|eprint=1412.6062|class=quant-ph|year=2014}}</ref> until a more effective classical algorithm was proposed.<ref>{{cite arXiv|last1=Barak|first1=Boaz|last2=Moitra|first2=Ankur|last3=O'Donnell|first3=Ryan| author3-link = Ryan O'Donnell (computer scientist) |last4=Raghavendra|first4=Prasad|last5=Regev|first5=Oded|last6=Steurer|first6=David|last7=Trevisan|first7=Luca|last8=Vijayaraghavan|first8=Aravindan|last9=Witmer|first9=David|last10=Wright|first10=John|title=Beating the random assignment on constraint satisfaction problems of bounded degree|eprint=1505.03424|class=cs.CC|year=2015}}</ref> The relative speed-up of the quantum algorithm is an open research question.

QAOA consists of the following steps:

# Defining a ''cost Hamiltonian'' <math>H_C</math> such that its ground state encodes the solution to the optimization problem.
# Defining a ''mixer Hamiltonian'' <math>H_M</math>.
# Defining the oracles <math>U_C(\gamma)= \exp(-\imath \gamma H_C)</math> and <math>U_M(\alpha)= \exp(-\imath \alpha H_M)</math>, with parameters <math>\gamma</math> and α.
# Repeated application of the oracles <math>U_C</math> and <math>U_M</math>, in the order: <math>U(\boldsymbol\gamma, \boldsymbol\alpha) = \coprod_{i=1}^N (U_C(\gamma_i) U_M(\alpha_i))</math>
# Preparing an initial state, that is a superposition of all possible states and apply <math>U(\boldsymbol\gamma, \boldsymbol\alpha)</math> to the state.
# Using classical methods to optimize the parameters <math>\boldsymbol\gamma, \boldsymbol\alpha</math> and measure the output state of the optimized circuit to obtain the approximate optimal solution to the cost Hamiltonian. An optimal solution will be one that maximizes the expectation value of the cost Hamiltonian <math>H_C</math>.
[[File:QAOAcircuit.png|thumb|457x457px|Sample QAOA ansatz for a three qubit circuit]]
The layout of the algorithm, viz, the use of cost and mixer Hamiltonians are inspired from the [[Quantum Adiabatic Theorem|Quantum Adiabatic theorem]], which states that starting in a ground state of a time-dependent Hamiltonian, if the Hamiltonian evolves slowly enough, the final state will be a ground state of the final Hamiltonian. Moreover, the adiabatic theorem can be generalized to any other eigenstate as long as there is no overlap (degeneracy) between different eigenstates across the evolution. Identifying the initial Hamiltonian with <math>H_M</math> and the final Hamiltonian with <math>H_C</math>, whose ground states encode the solution to the optimization problem of interest, one can approximate the optimization problem as the adiabatic evolution of the Hamiltonian from an initial to the final one, whose ground (eigen)state gives the optimal solution. In general, QAOA relies on the use of [[unitary operators]] dependent on <math> 2p </math> [[angle]]s (parameters), where <math> p>1 </math> is an input integer, which can be identified the number of layers of the oracle <math>U(\boldsymbol\gamma, \boldsymbol\alpha)</math>. These operators are iteratively applied on a state that is an equal-weighted [[quantum superposition]] of all the possible states in the computational basis. In each iteration, the state is measured in the computational basis and the Boolean function <math> C(z) </math> is estimated. The angles are then updated classically to increase <math> C(z) </math>. After this procedure is repeated a sufficient number of times, the value of <math> C(z) </math> is almost optimal, and the state being measured is close to being optimal as well. A sample circuit that implements QAOA on a quantum computer is given in figure. This procedure is highlighted using the following example of finding the [[minimum vertex cover]] of a graph.<ref>{{Cite journal |last=Ceroni |first=Jack |date=2020-11-18 |title=Intro to QAOA |url=https://pennylane.ai/qml/demos/tutorial_qaoa_intro |journal=PennyLane Demos |language=en}}</ref>

=== QAOA for finding the minimum vertex cover of a graph ===
The goal here is to find a [[Vertex cover|minimum vertex cover]] of a graph: a collection of vertices such that each edge in the graph contains at least one of the vertices in the cover. Hence, these vertices "cover" all the edges. We wish to find a vertex cover that has the smallest possible number of vertices. Vertex covers can be represented by a bit string where each bit denotes whether the corresponding vertex is present in the cover. For example, the bit string 0101 represents a cover consisting of the second and fourth vertex in a graph with four vertices.
[[File:GraphforQAOA.png|thumb|220x220px|Sample graph to illustrate the minimum vertex cover problem.]]
Consider the graph given in the figure. It has four vertices and there are two minimum vertex cover for this graph: vertices 0 and 2, and the vertices 1 and 2. These can be respectively represented by the bit strings 1010 and 0110. The goal of the algorithm is to sample these bit strings with high probability. In this case, the cost Hamiltonian has two ground states, |1010⟩ and |0110⟩, coinciding with the solutions of the problem. The mixer Hamiltonian is the simple, non-commuting sum of [[Pauli matrices|Pauli-X]] operations on each node of the graph and they are given by:

<math>H_C = -0.25 Z_3 + 0.5 Z_0 + 0.5 Z_1 + 1.25 Z_2 + 0.75 (Z_0 Z_1 + Z_0 Z_2 + Z_2 Z_3 + Z_1 Z_2)</math>

<math>H_M = X_0 + X_1 + X_2 + X_3</math>
[[File:QAOAoutput.png|thumb|<nowiki>Output of QAOA implementation in Qiskit for minimum vertex cover problem. Note that the bit string |1010> is flipped as |0101> as Qiskit uses reverse ordering of bits.</nowiki>]]
[[File:QAOAcircuitforGraph.png|thumb|Qiskit implementation of QAOA for minimum vertex cover problem.]]
Implementing QAOA algorithm for this four qubit circuit with two layers of the ansatz in qiskit (see figure) and optimizing leads to a probability distribution for the states given in the figure. This shows that the states |0110⟩ and |1010⟩ have the highest probabilities of being measured, just as expected.

=== Generalization of QAOA to constrained combinatorial optimization ===
In principle the optimal value of <math> C(z) </math> can be reached up to arbitrary precision, this is guaranteed by the adiabatic theorem<ref>{{cite arXiv|last1=Farhi|first1=Edward|last2=Goldstone|first2=Jeffrey|last3=Gutmann|first3=Sam|title=A Quantum Approximate Optimization Algorithm|eprint=1411.4028|class=quant-ph|year=2014}}</ref><ref>{{Cite journal|last1=Binkowski|first1=Lennart |last2=Koßmann|first2=Gereon |last3=Ziegler|first3=Timo |last4=Schwonnek|first4=René |year=2024|title=Elementary proof of QAOA convergence|journal=New Journal of Physics|volume=26|issue=7|page=073001|doi=10.1088/1367-2630/ad59bb|arxiv=2302.04968 |bibcode=2024NJPh...26g3001B }}</ref> or alternatively by the universality of the QAOA unitaries.<ref>{{Cite journal|last1=Morales|first1=M. E. |last2=Biamonte|first2=J. D.|last3=Zimborás|first3=Z. |date=2019-09-20|title=On the universality of the quantum approximate optimization algorithm|journal=Quantum Information Processing|volume=19|issue=9 |page=291|doi=10.1007/s11128-020-02748-9|arxiv=1909.03123 }}</ref> However, it is an open question whether this can be done in a feasible way.
For example, it was shown that QAOA exhibits a strong dependence on the ratio of a problem's [[Constraint (mathematics)|constraint]] to [[Variable (mathematics)|variables]] (problem density) placing a limiting restriction on the algorithm's capacity to minimize a corresponding [[Loss function|objective function]].<ref name=":0">{{Cite journal|last1=Akshay|first1=V.|last2=Philathong|first2=H.|last3=Morales|first3=M. E. S.|last4=Biamonte|first4=J. D.|date=2020-03-05|title=Reachability Deficits in Quantum Approximate Optimization|journal=Physical Review Letters|volume=124|issue=9|article-number=090504|doi=10.1103/PhysRevLett.124.090504|pmid=32202873|arxiv=1906.11259|bibcode=2020PhRvL.124i0504A|s2cid=195699685}}</ref>

It was soon recognized that a generalization of the QAOA process is essentially an alternating application of a continuous-time quantum walk on an underlying graph followed by a quality-dependent phase shift applied to each solution state. This generalized QAOA was termed as QWOA (Quantum Walk-based Optimisation Algorithm).<ref>{{Cite journal|last1=Marsh|first1=S.|last2=Wang|first2=J. B.|date=2020-06-08|title=Combinatorial optimization via highly efficient quantum walks|journal=Physical Review Research|volume=2|issue=2|article-number=023302|doi=10.1103/PhysRevResearch.2.023302|arxiv=1912.07353 |bibcode=2020PhRvR...2b3302M|s2cid=216080740}}</ref>

In the paper ''How many qubits are needed for quantum computational supremacy'' submitted to arXiv,<ref>{{Cite journal|last1=Dalzell|first1=Alexander M.|last2=Harrow|first2=Aram W.|last3=Koh|first3=Dax Enshan|last4=La Placa|first4=Rolando L.|date=2020-05-11|title=How many qubits are needed for quantum computational supremacy?|journal=Quantum|volume=4|article-number=264|doi=10.22331/q-2020-05-11-264|arxiv=1805.05224|issn=2521-327X|doi-access=free|bibcode=2020Quant...4..264D }}</ref> the authors conclude that a QAOA circuit with 420 [[qubits]] and 500 [[Constraint (mathematics)|constraints]] would require at least one century to be simulated using a classical simulation algorithm running on [[State of the art|state-of-the-art]] [[supercomputers]] so that would be [[Necessity and sufficiency#Sufficiency|sufficient]] for [[Quantum supremacy|quantum computational supremacy]].

A rigorous comparison of QAOA with classical algorithms can give estimates on depth <math> p </math> and number of qubits required for quantum advantage. A study of QAOA and [[Maximum cut|MaxCut]] algorithm shows that <math>p>11</math> is required for scalable advantage.<ref name="Lykov Wurtz Poole Saffman p. ">{{cite journal | last1=Lykov | first1=Danylo | last2=Wurtz | first2=Jonathan | last3=Poole | first3=Cody |author4-link=Mark Saffman | last4=Saffman | first4=Mark | last5=Noel | first5=Tom | last6=Alexeev | first6=Yuri | title=Sampling frequency thresholds for the quantum advantage of the quantum approximate optimization algorithm | journal=npj Quantum Information | year=2023 | volume=9 | issue=1 | article-number=73 | doi=10.1038/s41534-023-00718-4  | doi-access=free| arxiv=2206.03579 | bibcode=2023npjQI...9...73L }}</ref>

=== Variations of QAOA ===
Several variations to the basic structure of QAOA have been proposed,<ref>{{Cite journal |last1=Blekos |first1=Kostas |last2=Brand |first2=Dean |last3=Ceschini |first3=Andrea |last4=Chou |first4=Chiao-Hui |last5=Li |first5=Rui-Hao |last6=Pandya |first6=Komal |last7=Summer |first7=Alessandro |date=June 2024 |title=A Review on Quantum Approximate Optimization Algorithm and its Variants |journal=Physics Reports |volume=1068 |pages=1–66 |doi=10.1016/j.physrep.2024.03.002|arxiv=2306.09198 |bibcode=2024PhR..1068....1B }}</ref> which include variations to the ansatz of the basic algorithm. The choice of ansatz typically depends on the problem type, such as combinatorial problems represented as graphs, or problems strongly influenced by hardware design. However, ansatz design must balance specificity and generality to avoid overfitting and maintain applicability to a wide range of problems. For this reason, designing optimal ansatze for QAOA is an extensively researched and widely investigated topic. Some of the proposed variants are:

# Multi-angle QAOA<ref>{{Cite journal |last1=Herrman |first1=Rebekah |last2=Lotshaw |first2=Phillip C. |last3=Ostrowski |first3=James |last4=Humble |first4=Travis S. |last5=Siopsis |first5=George |date=2022-04-26 |title=Multi-angle quantum approximate optimization algorithm |journal=Scientific Reports |language=en |volume=12 |issue=1 |page=6781 |doi=10.1038/s41598-022-10555-8  |doi-access=free|issn=2045-2322 |pmc=9043219 |pmid=35474081|arxiv=2109.11455 |bibcode=2022NatSR..12.6781H }}</ref>
# Expressive QAOA (XQAOA)<ref>{{Cite journal |last1=Vijendran |first1=V |last2=Das |first2=Aritra |last3=Koh |first3=Dax Enshan |last4=Assad |first4=Syed M |last5=Lam |first5=Ping Koy |date=2024-04-01 |title=An expressive ansatz for low-depth quantum approximate optimisation |url=https://iopscience.iop.org/article/10.1088/2058-9565/ad200a |journal=Quantum Science and Technology |volume=9 |issue=2 |page=025010 |doi=10.1088/2058-9565/ad200a |issn=2058-9565|arxiv=2302.04479 |bibcode=2024QS&T....9b5010V }}</ref>
# QAOA+<ref>{{Cite book |last1=Chalupnik |first1=Michelle |last2=Melo |first2=Hans |last3=Alexeev |first3=Yuri |last4=Galda |first4=Alexey |chapter=Augmenting QAOA Ansatz with Multiparameter Problem-Independent Layer |date=September 2022 |title=2022 IEEE International Conference on Quantum Computing and Engineering (QCE) |publisher=IEEE |pages=97–103 |doi=10.1109/QCE53715.2022.00028 |arxiv=2205.01192 |isbn=978-1-6654-9113-6}}</ref>
# Digitised counteradiabatic QAOA<ref>{{Cite journal |last1=Chandarana |first1=P. |last2=Hegade |first2=N. N. |last3=Paul |first3=K. |last4=Albarrán-Arriagada |first4=F. |last5=Solano |first5=E. |last6=del Campo |first6=A. |last7=Chen |first7=Xi |date=2022-02-22 |title=Digitized-counterdiabatic quantum approximate optimization algorithm |url=https://link.aps.org/doi/10.1103/PhysRevResearch.4.013141 |journal=Physical Review Research |language=en |volume=4 |issue=1 |article-number=013141 |doi=10.1103/PhysRevResearch.4.013141 |arxiv=2107.02789 |bibcode=2022PhRvR...4a3141C |issn=2643-1564}}</ref>
# Quantum alternating operator ansatz<ref>{{Cite journal |last1=Hadfield |first1=Stuart |last2=Wang |first2=Zhihui |last3=O'Gorman |first3=Bryan |last4=Rieffel |first4=Eleanor |last5=Venturelli |first5=Davide |last6=Biswas |first6=Rupak |date=2019-02-12 |title=From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz |journal=Algorithms |language=en |volume=12 |issue=2 |page=34 |doi=10.3390/a12020034 |doi-access=free |issn=1999-4893|arxiv=1709.03489 }}</ref>,which allows for constrains on the optimization problem etc.

Another variation of QAOA focuses on techniques for parameter optimization, which aims at selecting the optimal set of initial parameters for a given problem and avoiding barren plateaus, which represent parameters leading to eigenstates which correspond to plateaus in the energy landscape of the cost Hamiltonian. 

Finally, there has been significant research interest in leveraging specific hardware to enhance the performance of QAOA across various platforms, such as trapped ion, neutral atoms, superconducting qubits, and photonic quantum computers. The goals of these approaches include overcoming hardware connectivity limitations and mitigating noise-related issues to broaden the applicability of QAOA to a wide range of combinatorial optimization problems.

==QAOA algorithm Qiskit implementation==
[[File:QAOA quantum circuit.png|thumb|QAOA quantum circuit]]

The quantum circuit shown here is from a simple example of how the QAOA algorithm can be implemented in Python<ref>{{cite web |url=https://learning.quantum.ibm.com/tutorial/quantum-approximate-optimization-algorithm |title=Solve utility-scale quantum optimization problems |access-date=2025-02-24}}</ref> using [[Qiskit]], an open-source quantum computing software development framework by IBM.

== See also ==
*[[Adiabatic quantum computation]]
*[[Quantum annealing]]
{{clear}}

== References ==
{{reflist}}

==External links==
* [https://short.classiq.io/qaoa_knapsack Implementation of the QAOA algorithm for the knapsack problem with Classiq]{{Dead link|date=February 2026 |bot=InternetArchiveBot }}

{{Quantum information}}

[[Category:Quantum algorithms]]