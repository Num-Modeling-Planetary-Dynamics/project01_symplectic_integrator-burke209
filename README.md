| EAPS 591 - Numerical Modeling of Planetary Orbits | Fall 2022 | Prof. David Minton |
| ----------------------------- | --------- | ------------------ |
# Project 1 - Write your own symplectic integrator
###### *Due: Friday, Oct. 14, 2022 by 5:00pm*

Write a symplectic n-body integrator. You may use either Jacobi coordinates or democractic heliocentric coordinates, but be sure to use the correct form of the Hamiltonian depending on which method you use. Only implement the basic symplectic integrator, so do not worry about close encounters. Simulate the orbit of Pluto and Neptune for $10^5$ years. Produce plots of the total change in the system energy, $\Delta\mathcal{E}/\mathcal{E}_0$ and the resonance angle $\phi=3\lambda_P-2\lambda_N-\varpi_P$ vs. time, where $\lambda=M+\varpi$ is the *mean anomaly* and submit them, along with a brief description of the results to Brightspace. All code and data should be submitted via GitHub.

- The integrator should be based on the *Leapfrog* algorithm, where each step is performed by alternating half steps that act on the position and velocity vectors separately (aka *drift-kick-drift*). See below for detailed description for how two of these methods are iimplemented. 

- The *drift* step should make use of a solution to Kepler's equation. Use 
the $f$ and $g$ functions to convert eccentric anomaly, $E$, from Kepler's equation to cartesian position and velocity vectors, $\mathbf{x}$ and $\mathbf{v}$ (see Murray and Dermott sec. 2.4).

- To perform the kick step, you need to make use of one of the two Hamiltonians (see the two papers in the reference folder). The difference between the two is in the choice of coordinate system.

- Place your source code in the `src` folder of your repository. Output all data to the `data` folder of the repository and place all figures you generate in `plots` (*Note: contents of the `plots` folder will be ignored by GitHub*)

- G

#### The Democratic Heliocentric Method

This method was developed for the Symplectic Massive Body Algorithm (SyMBA), which can handle close encounters between massive bodies by reducing the step size of the integration of any bodies that are so close that the interaction part of the Hamiltonian is large relative to the Kepler part for a given step size. However, the fixed step size version works well as an alternative to Wisdom and Holman's Mixed Variable Symplectic map (see below). Its advantage is that the coordinate system is easier to deal with, though because of the additional Hamiltonian it requires slighty more computational work per step than the MVS method.


The Hamiltonian is given by:
$ \mathcal{H} =\mathcal{H}_{Kepler} + \mathcal{H}_{Interaction} +\mathcal{H}_{Sun} $

Defining the central body as index $0$, and the smaller bodies (the planets) as bodies $1...n$, each of the component Hamiltonians can be written as:

$$
\begin{aligned}
\mathcal{H}_{Kepler} &= \sum_{i=i}^{n}\left(\frac{\left|\mathbf{P}_i\right|_i^2}{2m_i}-\frac{\mathcal{G}m_0m_i}{\mathbf{Q}_i}\right)\\
\mathcal{H}_{Interaction} &= -\sum_{i=1}^{n-1}\sum_{j=i+1}^{n}\frac{\mathcal{G}m_im_j}{\left|\mathbf{Q}_i-\mathbf{Q}_j\right|}\\
\mathcal{H}_{Sun} &= \frac{1}{2m_0}\left|\sum_{i=1}^n\mathbf{P}_i\right|^2
\end{aligned}
$$

The coordinate and momentum variables for this Hamiltonian are defined from the standard coordinate variable $\mathbf{q}=\mathbf{r}$ and momenum variable $\mathbf{p}=m\mathbf{v}$ as:

$$
\begin{aligned}
\mathbf{P}_i &= \begin{cases}
   \mathbf{p}_i- \frac{m_i}{m_{tot}}\sum_{j=0}^n\mathbf{p}_j,&\qquad \text{if}\ i\neq0\\
   \sum_{j=0}^n\mathbf{p}_j,&\qquad \text{if}\ i=0
   \end{cases}\\
\mathbf{Q}_i &= \begin{cases}
   \mathbf{q}_i-\mathbf{q}_0 &\qquad \text{if}\ i\neq0\\
   \frac{1}{m_{tot}}\sum_{j=0}^nm_jq_j,&\qquad \text{if}\ i=0
   \end{cases}   
\end{aligned}
$$

Thus the position vectors ($\mathbf{Q}$) of the planets ($i\neq0$) are taken to be heliocentric (that is, relative to the central body) and the velocity vectors ($\mathbf{P}$) of the planets are taken to be barycentric (relative to the center of mass of the whole system). 

The actual algorithm can be written formally as a kick-drift-kick leapfrog integration over one time step of length $\Delta t$ as:

$$E_{Sun}\left(\frac{\Delta t}{2}\right)E_{int}\left(\frac{\Delta t}{2}\right)E_{Kep}\left(\Delta t\right)E_{int}\left(\frac{\Delta t}{2}\right)E_{Sun}\left(\frac{\Delta t}{2}\right)$$

where $E_{int}$ is the interaction step (the kick) and $E_{Kep}$ is the Keplerian step (the planet drift) and $E_{Sun}$ updates the barycenter of the Sun (the solar drift or linear drift). 

To start the simulation, you will need to supply the initial conditions and compute the barycentric velocity values. For each step, the algorithim implementation looks something like this:

1. Drift the Sun 
2. Compute the interaction accelerations for each $i^{th}$ planet, $\left(\dot{\mathbf{v}}_{b,i}\right)_{int}$ and kick the barycentric velocity by $\Delta\mathbf{v}_{b,i}= \left(\dot{\mathbf{v}}_{b,i}\right)_{int}\frac{\Delta t}{2}$
3. Drift the bodies along a Keplerian arc. To do this, solve Kepler's equation for $\Delta E$ in one step $\Delta t$. Use the $f$ and $g$ functions to advance each body's *heliocentric* position and *barycentric* velocity vector.
4. Repeat the kick step as in 2.
5. Repeat the solar drift as in step 1.


Reference
: Duncan, M.J., Levison, H.F., Lee, M.H., 1998. A Multiple Time Step Symplectic Algorithm for Integrating Close Encounters. AJ 116, 2067–2077. https://doi.org/10.1086/300541

  

#### The Mixed Variable Symplectic Map

The Wisdom-Holman mixed variable symplectic map (sometimes called the MVS method) is described in Wisdom & Holman (1991). This method is used when there is a dominant central body (such as the Sun) and the smaller bodies (such as the planets) are distant enough from each other that their gravitational interaction always remains small relative to the Sun. 

The Hamiltonian of the n-body system is separated into two parts:

$$ \mathcal{H} =\mathcal{H}_{Kepler} + \mathcal{H}_{Interaction} $$

In order to achieve this separation, the MVS method makes use of Jacobi coordinates. The Jacobi coordinates are determined algorithmically. See section 9.5 of the Murray and Dermott book to see how to define the coordinate system.

Each component of the Hamiltonian is defined in Jacobi coordinates as:

$$
\begin{aligned}
\mathcal{H}_{Kepler} &= \sum_{j=1}^{N-1}\left(\frac{\tilde{p}_j^2}{2\tilde{m}_j}-\frac{\mathcal{G}\tilde{M}_j\tilde{m}_j}{\tilde{r_j}}\right)\\
\mathcal{H}_{Interaction} &= \sum_{j=1}^{N-1}\frac{\mathcal{G}m_0m_j}{\tilde{r_j}} - \sum_{j=0}^{N=2}\sum_{k=j+1}^{N-1}\frac{\mathcal{G}m_jm_k}{r_{jk}}
\end{aligned}
$$

The actual algorithm can be written formally as a kick-drift-kick leapfrog integration over one time step of length $\Delta t$ as:

$$E_{int}\left(\frac{\Delta t}{2}\right)E_{Kep}\left(\Delta t\right)E_{int}\left(\frac{\Delta t}{2}\right)$$

where $E_{int}$ is the interaction step (the kick) and $E_{Kep}$ is the Keplerian step (the drift).

To implement this, you must decide on a step size $\tau$. A general rule of thumb for these kinds of integrators is that your step size should be no larger $\sim1/30$ of the period of the innermost body in the system. This means that if you were to include Mercury, you would need a step size of about 3 days. For a simulation containing only Neptune and Pluto, this is about 5.5 years. 

To start the simulation, you will need to supply the initial conditions and compute the Jacobi position and velocity values. For each step, the algorithim implementation looks something like this:

1. Compute the interaction accelerations for each $i^{th}$ planet, $\left(\dot{\tilde{\mathbf{v}}}_i\right)_{int}$ and kick the velocity by $\Delta\tilde{\mathbf{v}}_i= \left(\dot{\tilde{\mathbf{v}}}_i\right)_{int}\frac{\Delta t}{2}$
2. Drift the bodies along a Keplerian arc. To do this, solve Kepler's equation for $\Delta E$ in one step $\Delta t$. Use the $f$ and $g$ functions to advance each body's *Jacobi* position and velocity vector.
3. Repeat the kick step as in 1.



Reference
: Wisdom, J., Holman, M., 1991. Symplectic maps for the n-body problem. AJ 102, 1528–1538. https://doi.org/10.1086/115978




