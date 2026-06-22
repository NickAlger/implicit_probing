# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""DOLFINx (modern FEniCS) implementation of the ``ImplicitProblem`` interface.

``FenicsImplicitProblem`` is **frozen** at a user-supplied expansion point ``(theta, u)``: the user
solves the nonlinear state equation ``R(theta, u) = 0`` themselves (by whatever means, with the *real*
Dirichlet BCs) and hands the solved Functions to this class, which then provides derivative probes at
that point. The class never does a nonlinear solve. It assembles the linearized state operator
``A = d_u R`` once (with *homogenized* Dirichlet BCs), factorizes it (or uses user-supplied solvers),
and turns each ``PartialTerm`` into a single UFL form.

The whole hook is one uniform recipe (``_term_form``): take the base form (``R`` or ``Q``, each a
1-form in its output test function), optionally **replace** that test function with the pairing
(``OMEGA`` -> the functional ``omega``; an adjoint vector -> ``v̂``), **nest** one ``ufl.derivative``
per supplied direction, and (for reverse objects) introduce the **open slot** as one more
``ufl.derivative`` with no explicit direction. Forms within a request share a test-function space, so
they are summed and assembled **once** (the FEniCS performance win behind ``assemble_partial_sum``).

Mixed function spaces are used deliberately (e.g. theta in CG2, u in CG3, the observation test
function in CG1) so that any accidental conflation of the parameter, state, and observation spaces
fails loudly rather than silently.

This module imports ``dolfinx`` and is therefore an OPTIONAL part of implicit_probing (the core
package needs only numpy). It requires a conda DOLFINx environment.
"""
import typing as typ

import ufl
from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc as petsc_fem

from implicit_probing.driver import OMEGA

__all__ = ['FenicsImplicitProblem']


def _lu_solver(A: PETSc.Mat) -> PETSc.KSP:
    """A reusable LU solver for A: factorized once on first solve, reused for all subsequent solves."""
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    return ksp


class FenicsImplicitProblem:
    """``ImplicitProblem`` for a DOLFINx map ``q(theta) = Q(theta, u(theta))``, frozen at ``(theta, u)``.

    Parameters
    ----------
    R_form : ufl.Form
        The state residual as a 1-form linear in a test function in the state space ``V_u``.
    Q_form : ufl.Form
        The output/observation as a 1-form linear in a test function in the observation space ``V_q``.
    theta, u : dolfinx.fem.Function
        The frozen expansion point; ``u`` must already solve ``R(theta, u) = 0`` (with the real BCs).
    bcs : list[dolfinx.fem.DirichletBC] | None
        The **homogenized** (zero-valued) Dirichlet BCs of the state space, applied to ``A`` and to
        every incremental right-hand side.
    forward_solver, adjoint_solver : callable | None
        Optional custom solvers, each mapping a RHS ``PETSc.Vec`` to the solution ``PETSc.Vec``. If
        omitted, a single reused LU factorization of ``A`` is used (adjoint via its transpose solve).
    """

    def __init__(self, R_form, Q_form, theta, u, bcs=None,
                 forward_solver=None, adjoint_solver=None):
        self.R_form = R_form
        self.Q_form = Q_form
        self.theta = theta
        self.u = u
        self.bcs = list(bcs) if bcs is not None else []
        self.v_R = R_form.arguments()[0]        # output test function, in the state space V_u
        self.v_Q = Q_form.arguments()[0]        # observation test function, in V_q
        self.V_u = u.function_space
        self.V_theta = theta.function_space
        self.V_q = self.v_Q.ufl_function_space()  # observation space (from the Q-form test function)

        # Linearized state operator A = d_u R at (theta, u), with homogenized Dirichlet BCs (identity
        # rows on constrained dofs). Homogeneous BCs => the BC columns are multiplied by zero
        # incrementals, so no lifting is needed; we just zero the BC dofs of each incremental RHS.
        a = fem.form(ufl.derivative(R_form, u))
        self.A = petsc_fem.assemble_matrix(a, bcs=self.bcs)
        self.A.assemble()

        self._forward_solver = forward_solver
        self._adjoint_solver = adjoint_solver
        self._ksp = _lu_solver(self.A) if (forward_solver is None or adjoint_solver is None) else None

    # --- ImplicitProblem interface ---

    def solve_operator(self, b):
        """Solve A x = b (homogenized BCs); return the incremental state as a Function in V_u."""
        return self._wrap(self._solve(b, transpose=False))

    def solve_operator_adjoint(self, c):
        """Solve A* x = c; return the incremental adjoint as a Function in V_u."""
        return self._wrap(self._solve(c, transpose=True))

    def assemble_partial_sum(self, terms, omega):
        """Assemble sum_i terms[i] as one combined UFL form, then one PETSc vector.

        OMEGA pairings are resolved to ``omega`` (a Function in the observation space ``V_q``).
        """
        combined = None
        for t in terms:
            form = self._term_form(t, omega)
            if form.empty():
                continue                         # this partial vanishes for these forms (e.g. d_theta Q)
            form = t.coefficient * form
            combined = form if combined is None else combined + form
        if combined is None:                     # every term vanished -> zero vector in the target space
            return fem.Function(self._target_space(terms[0])).x.petsc_vec.copy()

        vec = petsc_fem.assemble_vector(fem.form(combined))
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        if self._is_state_rhs(terms[0]):         # b_beta / c_beta -> homogenized BCs; probe outputs -> not
            petsc_fem.set_bc(vec, self.bcs)
            vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return vec

    # --- internals ---

    def _term_form(self, t, omega):
        """Turn one PartialTerm into its UFL form (the uniform recipe; see the module docstring)."""
        if t.function == 'R':
            form, out_arg = self.R_form, self.v_R
        else:
            form, out_arg = self.Q_form, self.v_Q
        # pairing: replace the output test function with omega (resolved here) or an adjoint vector
        pairing = omega if t.pairing is OMEGA else t.pairing
        if pairing is not None:
            form = ufl.replace(form, {out_arg: pairing})
        # filled directions
        for d in t.theta_dirs:
            form = ufl.derivative(form, self.theta, d)
        for w in t.u_vecs:
            form = ufl.derivative(form, self.u, w)
        # open slot: one more derivative with no direction -> a fresh test function in that space
        if t.open_slot == 'u':
            form = ufl.derivative(form, self.u)
        elif t.open_slot == 'theta':
            form = ufl.derivative(form, self.theta)
        return form

    def _is_state_rhs(self, t):
        """True for state-solve right-hand sides (b_beta, c_beta), which take homogenized BCs."""
        return t.open_slot == 'u' or (t.open_slot is None and t.function == 'R')

    def _target_space(self, t):
        if t.open_slot == 'theta':
            return self.V_theta
        if t.open_slot == 'u':
            return self.V_u
        return self.V_u if t.function == 'R' else self.V_q

    def _solve(self, b, transpose):
        custom = self._adjoint_solver if transpose else self._forward_solver
        if custom is not None:
            return custom(b)
        x = b.duplicate()
        if transpose:
            self._ksp.solveTranspose(b, x)
        else:
            self._ksp.solve(b, x)
        return x

    def _wrap(self, x_vec):
        """Wrap a solution PETSc vector as a Function in V_u (so it can enter later UFL derivatives)."""
        f = fem.Function(self.V_u)
        x_vec.copy(f.x.petsc_vec)
        f.x.scatter_forward()
        return f
