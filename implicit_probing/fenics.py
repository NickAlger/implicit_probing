# Authors: Blake Christierson and Nick Alger
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""DOLFINx (modern FEniCS) implementation of the ``ImplicitProblem`` interface.

``FenicsImplicitProblem`` is **frozen** at a user-supplied expansion point ``(theta, u)``: the user
solves the nonlinear state equation ``R(theta, u) = 0`` themselves (by whatever means, with the *real*
Dirichlet BCs) and hands the solved Functions to this class, which then provides derivative probes at
that point. The class never does a nonlinear solve. It assembles the linearized state operator
``A = d_u R`` once (with *homogenized* Dirichlet BCs), factorizes it (or uses user-supplied solvers),
and turns each ``PartialTerm`` into a single UFL form. The default solver is a direct LU
factorization, MUMPS where PETSc has it (pivoting, MPI) and PETSc's native LU otherwise
(``direct_lu``); ``ksp_factory`` swaps in any other PETSc solver, direct or iterative.

The whole hook is one uniform recipe (``_term_form``): take the base form (``R`` or ``Q``, each a
1-form in its output test function), optionally **replace** that test function with the pairing
(``OMEGA`` -> the functional ``omega``; an adjoint vector -> ``v̂``), **nest** one ``ufl.derivative``
per supplied direction (repeated by its multiplicity -- ``ufl.derivative`` is one direction at a time,
so this hook does not exploit the repetition), and (for reverse objects) introduce the **open slot** as one more
``ufl.derivative`` with no explicit direction. Forms within a request share a test-function space, so
they are summed and assembled **once** (the FEniCS performance win behind ``assemble_partial_sum``).

Mixed function spaces are used deliberately (e.g. theta in CG2, u in CG3, the observation test
function in CG1) so that any accidental conflation of the parameter, state, and observation spaces
fails loudly rather than silently.

**Batched probes** (``dev/batched_probes_design.md``): a Python ``list`` of B Functions in place of a
direction (or of ``omega``) is a *batch* -- B independent probes at the frozen point in ONE ``probe``
call. The lattice is walked once; per node the B right-hand sides are assembled into one dense
PETSc matrix and solved together (``KSP.matSolve`` / ``matSolveTranspose`` on the hook's own KSP --
a direct solver traverses its factorization once for all B, an iterative one loops internally);
and the combined UFL form of a request is built ONCE with *slot* coefficient Functions, then
re-assembled per member after copying that member's dofs into the slots (the ffcx kernel and the
form construction are shared; the integration over the mesh is still done B times). Single
Functions in the same call are shared by every member. Results come back as lists of Vecs wherever
they depend on a batched input (``forward[mu]`` iff a batched direction has ``mu_k >= 1``;
``reverse[mu]`` iff ``omega`` is batched or such a direction exists), and as single Vecs otherwise
-- ``forward[(0, ..)] = q(theta0)`` is always a single Vec, and so is the gradient ``reverse[(0, ..)]``
under direction-only batching. The batched solve runs on the hook's solver: the default direct LU
(MUMPS where available), or whatever ``ksp_factory`` builds; user-supplied ``forward_solver`` /
``adjoint_solver`` callables keep their ``Vec -> Vec`` contract and are applied per member.

This module imports ``dolfinx`` and is therefore an OPTIONAL part of implicit_probing (the core
package needs only numpy). It requires a conda DOLFINx environment.
"""
import dataclasses
import typing as typ

import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc as petsc_fem

from implicit_probing.batching import infer_batch_size
from implicit_probing.driver import OMEGA

__all__ = ['FenicsImplicitProblem', 'direct_lu']


def direct_lu(package: typ.Optional[str] = None):
    """A ``ksp_factory`` for a direct LU solver: ``A -> PETSc.KSP`` (type ``preonly`` + ``lu``), the
    factorization computed once on the first solve and reused for every forward and transpose solve.

    ``package`` names PETSc's factor package (``'mumps'``, ``'superlu_dist'``, ``'umfpack'``, ...).
    ``None`` -- the hook's default -- means **MUMPS if this PETSc build has it, else PETSc's native
    LU**. MUMPS pivots (PETSc's native LU does not, and silently returns NaN on an indefinite
    saddle-point operator) and runs under MPI (native LU is sequential only); on small serial
    problems native LU is the faster of the two per solve.

    Terminology: a PETSc ``KSP`` is the linear-solver *object*, for direct and iterative solves
    alike -- a direct solve is the ``preonly`` KSP whose preconditioner is the factorization. So
    ``ksp_factory`` is not asking for a Krylov method; iterative solvers are one thing it can build.
    """
    if package is None:
        package = 'mumps' if PETSc.Sys.hasExternalPackage('mumps') else 'petsc'

    def factory(A: PETSc.Mat) -> PETSc.KSP:
        ksp = PETSc.KSP().create(A.getComm())
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType(package)
        return ksp

    factory.package = package                    # introspectable: which package the default resolved to
    return factory


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
        omitted, the solver from ``ksp_factory`` is used (adjoint via its transpose solve). In a
        batched probe a custom solver is applied per member (it keeps its single-vector contract and
        does not get the multi-right-hand-side solve).
    ksp_factory : callable | None
        ``PETSc.Mat -> PETSc.KSP``: builds the linear solver for the assembled operator ``A`` (called
        once). A PETSc ``KSP`` is the solver object for direct AND iterative solves. **Default:
        ``direct_lu()`` -- a direct LU factorization, MUMPS if this PETSc build has it, else PETSc's
        native LU** (sequential, unpivoted), computed once and reused. Pass ``direct_lu('superlu_dist')``
        for another factor package, or your own factory for a Krylov solver + preconditioner or a
        KSP configured through the PETSc options database. Batched probes solve all B right-hand
        sides through this solver at once (``KSP.matSolve``). Ignored when both custom solvers are
        given.
    """

    def __init__(self, R_form, Q_form, theta, u, bcs=None,
                 forward_solver=None, adjoint_solver=None, ksp_factory=None):
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
        factory = ksp_factory if ksp_factory is not None else direct_lu()
        self._ksp = factory(self.A) if (forward_solver is None or adjoint_solver is None) else None

    # --- ImplicitProblem interface ---

    def solve_operator(self, b):
        """Solve A x = b (homogenized BCs); return the incremental state as a Function in V_u.
        A batch (a list of B right-hand sides) is one multi-right-hand-side solve -> a list of B."""
        return self._solve_any(b, transpose=False)

    def solve_operator_adjoint(self, c):
        """Solve A* x = c; return the incremental adjoint as a Function in V_u (a list for a batch)."""
        return self._solve_any(c, transpose=True)

    def assemble_partial_sum(self, terms, omega):
        """Assemble sum_i terms[i] as one combined UFL form, then one PETSc vector.

        OMEGA pairings are resolved to ``omega`` (a Function in the observation space ``V_q``). A
        batched request (a list among the vectors it uses; see the module docstring) returns a list
        of B vectors: the combined form is built once with slot coefficients and assembled per member.
        """
        B = infer_batch_size(terms, omega, _list_batch_size)
        if B is None:
            return self._assemble_single(terms, omega)
        return self._assemble_batched(terms, omega, B)

    def _assemble_single(self, terms, omega):
        """One (unbatched) request: the original path, unchanged."""
        combined = None
        for t in terms:
            form = self._term_form(t, omega)
            if form.empty():
                continue                         # this partial vanishes for these forms (e.g. d_theta Q)
            form = t.coefficient * form
            combined = form if combined is None else combined + form
        if combined is None:                     # every term vanished -> zero vector in the target space
            return self._zero_vector(terms[0])
        return self._assemble(fem.form(combined), self._is_state_rhs(terms[0]))

    def _assemble_batched(self, terms, omega, B):
        """A batched request: build the combined form ONCE with a slot Function standing in for every
        batched list (one slot per distinct list; single Functions enter directly -- broadcast), then
        per member copy that member's dofs into the slots and assemble. Returns a list of B Vecs."""
        slots = {}                               # id(list) -> (slot Function, the list)

        def slot_for(vec, space):
            if not isinstance(vec, list):
                return vec                       # single Function: shared by every member
            key = id(vec)
            if key not in slots:
                slots[key] = (fem.Function(space), vec)
            return slots[key][0]

        combined = None
        for t in terms:
            t_slotted = dataclasses.replace(
                t,
                theta_dirs=tuple((slot_for(d, self.V_theta), m) for d, m in t.theta_dirs),
                u_vecs=tuple((slot_for(v, self.V_u), m) for v, m in t.u_vecs),
                pairing=(t.pairing if (t.pairing is None or t.pairing is OMEGA)
                         else slot_for(t.pairing, self.V_u)))
            # omega enters (and gets its slot) only through terms that pair with it -- the driver hands
            # it to every request, and a request that never uses it must not depend on its batch size
            omega_t = slot_for(omega, self.V_q) if (t.pairing is OMEGA and omega is not None) else None
            form = self._term_form(t_slotted, omega_t)
            if form.empty():
                continue
            form = t.coefficient * form
            combined = form if combined is None else combined + form
        if combined is None:
            return [self._zero_vector(terms[0]) for _ in range(B)]

        compiled = fem.form(combined)            # compiled once; coefficients are the slots
        is_rhs = self._is_state_rhs(terms[0])
        out = []
        for j in range(B):
            for slot, members in slots.values():
                slot.x.array[:] = members[j].x.array          # owned + ghost dofs, same space/layout
            out.append(self._assemble(compiled, is_rhs))
        return out

    def _assemble(self, compiled_form, is_state_rhs):
        """Assemble one compiled form into a fresh ghosted Vec (the ghost-reduce / BC recipe)."""
        vec = petsc_fem.assemble_vector(compiled_form)
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        if is_state_rhs:                         # b_beta / c_beta -> homogenized BCs; probe outputs -> not
            petsc_fem.set_bc(vec, self.bcs)
            vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return vec

    def _zero_vector(self, term):
        return fem.Function(self._target_space(term)).x.petsc_vec.copy()

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
        # filled directions: (vector, multiplicity) pairs; ufl.derivative is per-direction, so nest
        # one derivative per copy (this hook gains nothing from the multiplicity -- it just unrolls it).
        for d, mult in t.theta_dirs:
            for _ in range(mult):
                form = ufl.derivative(form, self.theta, d)
        for w, mult in t.u_vecs:
            for _ in range(mult):
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

    def _solve_any(self, b, transpose):
        """Dispatch: a single Vec -> a Function; a list of B Vecs -> a list of B Functions."""
        if isinstance(b, list):
            custom = self._adjoint_solver if transpose else self._forward_solver
            if custom is not None:               # loop fallback: the callable keeps its Vec -> Vec contract
                return [self._wrap(custom(b_j)) for b_j in b]
            return self._solve_batch(b, transpose)
        return self._wrap(self._solve(b, transpose))

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

    def _solve_batch(self, bs, transpose):
        """All B right-hand sides at once: stack their owned blocks into a dense matrix with A's row
        layout, ``KSP.matSolve`` (``matSolveTranspose``) on the hook's KSP, and wrap the solution
        columns as Functions. Dense RHS matrices are never handed out as driver vectors; the RHS
        Vecs were assembled (and BC-constrained) on ghosted vectors by ``assemble_partial_sum``."""
        B = len(bs)
        rows = self.A.getSizes()[0]              # (local, global) row layout of A
        Bm = PETSc.Mat().createDense(size=(rows, (PETSc.DECIDE, B)), comm=self.A.getComm())
        Bm.setUp()
        arr = Bm.getDenseArray()                 # the local (owned rows) x B block, writable view
        for j, b in enumerate(bs):
            arr[:, j] = b.array                  # ``Vec.array`` of a ghosted Vec is its owned part
        Bm.assemble()
        X = Bm.duplicate()
        X.assemble()
        if transpose:
            self._ksp.matSolveTranspose(Bm, X)
        else:
            self._ksp.matSolve(Bm, X)
        cols = np.array(X.getDenseArray(), copy=True)
        out = []
        for j in range(B):
            f = fem.Function(self.V_u)
            f.x.petsc_vec.array[:] = cols[:, j]  # owned dofs, then fill the ghosts
            f.x.scatter_forward()
            out.append(f)
        Bm.destroy()
        X.destroy()
        return out

    def _wrap(self, x_vec):
        """Wrap a solution PETSc vector as a Function in V_u (so it can enter later UFL derivatives)."""
        f = fem.Function(self.V_u)
        x_vec.copy(f.x.petsc_vec)
        f.x.scatter_forward()
        return f


def _list_batch_size(vec):
    """The FEniCSx hook's notion of a batched vector: a Python ``list`` of B Functions/Vecs. Any other
    sequence type is refused rather than guessed at."""
    if isinstance(vec, list):
        return len(vec)
    if isinstance(vec, (tuple, set, frozenset)):
        raise TypeError('a batch of FEniCSx vectors must be a list (a tuple/set is ambiguous here)')
    return None
