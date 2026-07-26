# PCG Method

This section covers the Preconditioned Conjugate Gradient (PCG) method. It can be utilized as an alternative solver or baseline comparison for symmetric positive-definite systems arising from diffusion-reaction equations on metric graphs. The method follows (Arioli, M., Benzi, M., *A Finite Element Method for Quantuum Graphs*, 2018).

The following functions are used to compute the Schur complement and execute the modified PCG algorithm.

```@docs
sysof_equations
HEEInv
Schur_Mult
PCG_mod
```

The method is exemplarily applied in the following function:

```@docs
PCG_example
```