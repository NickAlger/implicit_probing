# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
import unittest

from implicit_probing.backend.multiset import Multiset, subset_lattice
from implicit_probing.backend import symbolic as sym
from implicit_probing.backend.symbolic import Term, ID, OMEGA, adjoint


def ms(*xs):
    """Shorthand: ms(1, 1) == Multiset([1, 1])."""
    return Multiset(xs)


class TestSeeds(unittest.TestCase):
    def test_forward_seed(self):
        self.assertEqual(sym.seed_forward_q(), {Term(ID, 'Q', ms(), ms()): 1})

    def test_residual_seed(self):
        self.assertEqual(sym.seed_residual_r(), {Term(ID, 'R', ms(), ms()): 1})

    def test_reverse_seed_shared_by_g_and_radj(self):
        # Lagrangian L = omega(Q) + R(v):  the omega-paired Q term and the vhat_empty-paired R term.
        self.assertEqual(sym.seed_reverse(), {
            Term(OMEGA, 'Q', ms(), ms()): 1,
            Term(adjoint(ms()), 'R', ms(), ms()): 1,
        })


class TestForwardExpansions(unittest.TestCase):
    def test_first_derivative_of_q(self):
        # Dq theta_1 = d_theta Q theta_1 + d_u Q uhat_{1}
        D = sym.differentiate_over_lattice(sym.seed_forward_q(), ms(1))
        self.assertEqual(D[ms(1)], {
            Term(ID, 'Q', ms(1), ms()): 1,
            Term(ID, 'Q', ms(), Multiset([ms(1)])): 1,
        })

    def test_second_derivative_symmetric_matches_paper(self):
        # t4s.pdf Section 4.2 worked example:
        #   D^2 q theta_1^2 = d2_thetatheta Q theta_1^2 + 2 d2_thetau Q theta_1 uhat_{1}
        #                     + d2_uu Q uhat_{1}^2 + d_u Q uhat_{1,1}
        D = sym.differentiate_over_lattice(sym.seed_forward_q(), ms(1, 1))
        u1 = ms(1)        # index of uhat_{1}
        u11 = ms(1, 1)    # index of uhat_{1,1}
        self.assertEqual(D[ms(1, 1)], {
            Term(ID, 'Q', ms(1, 1), ms()): 1,
            Term(ID, 'Q', ms(1), Multiset([u1])): 2,
            Term(ID, 'Q', ms(), Multiset([u1, u1])): 1,
            Term(ID, 'Q', ms(), Multiset([u11])): 1,
        })

    def test_lattice_keys_match_subset_lattice(self):
        for alpha in [ms(1), ms(1, 1), ms(1, 2), ms(1, 1, 2), ms(1, 2, 3), ms(1, 1, 1)]:
            with self.subTest(alpha=alpha):
                D = sym.differentiate_over_lattice(sym.seed_forward_q(), alpha)
                self.assertEqual(set(D.keys()), set(subset_lattice(alpha)))

    def test_coefficients_are_positive_integers(self):
        for alpha in [ms(1, 1, 1), ms(1, 2, 3), ms(1, 1, 2)]:
            D = sym.differentiate_over_lattice(sym.seed_forward_q(), alpha)
            for beta, expansion in D.items():
                for term, coeff in expansion.items():
                    self.assertIsInstance(coeff, int)
                    self.assertGreater(coeff, 0)

    def test_base_node_is_the_seed(self):
        D = sym.differentiate_over_lattice(sym.seed_forward_q(), ms(1, 2))
        self.assertEqual(D[ms()], sym.seed_forward_q())


class TestDifferentiateTerm(unittest.TestCase):
    def test_three_forward_branches(self):
        # differentiate d_u Q uhat_{1}  (i.e. (ID, Q, {}, {{1}})) in direction 2
        out = sym.differentiate_term(Term(ID, 'Q', ms(), Multiset([ms(1)])), 2)
        self.assertEqual(out, {
            Term(ID, 'Q', ms(2), Multiset([ms(1)])): 1,        # (19a) raise theta order
            Term(ID, 'Q', ms(), Multiset([ms(1), ms(2)])): 1,  # (19b) new uhat_{2}
            Term(ID, 'Q', ms(), Multiset([ms(1, 2)])): 1,      # (19c) raise uhat_{1} -> uhat_{1,2}
        })

    def test_multiplicity_factor_in_step_19c(self):
        # differentiate d2_uu Q uhat_{1}^2  (Gamma = {{1}, {1}}) in direction 2:
        # the product rule on uhat_{1}^2 gives coefficient 2 for the raised term.
        out = sym.differentiate_term(Term(ID, 'Q', ms(), Multiset([ms(1), ms(1)])), 2)
        self.assertEqual(out[Term(ID, 'Q', ms(), Multiset([ms(1), ms(1, 2)]))], 2)


class TestReverseExpansions(unittest.TestCase):
    def test_first_derivative_raises_the_adjoint(self):
        # Differentiating the reverse seed once should:
        #   - expand the omega-paired Q term into its two forward branches, and
        #   - expand the vhat_empty-paired R term into its two forward branches PLUS the raised
        #     adjoint vhat_{1} (eq 20).
        D = sym.differentiate_over_lattice(sym.seed_reverse(), ms(1))
        self.assertEqual(D[ms(1)], {
            # from omega(Q):
            Term(OMEGA, 'Q', ms(1), ms()): 1,
            Term(OMEGA, 'Q', ms(), Multiset([ms(1)])): 1,
            # from R(vhat_empty):
            Term(adjoint(ms()), 'R', ms(1), ms()): 1,
            Term(adjoint(ms()), 'R', ms(), Multiset([ms(1)])): 1,
            Term(adjoint(ms(1)), 'R', ms(), ms()): 1,   # raised outer adjoint: vhat_{1}
        })

    def test_adjoint_orders_present_over_lattice(self):
        # Over alpha = {1,1}, an incremental adjoint vhat_{1,1} (order-2 outer pairing) must appear.
        D = sym.differentiate_over_lattice(sym.seed_reverse(), ms(1, 1))
        top = D[ms(1, 1)]
        self.assertTrue(any(
            term.pairing.is_adjoint and term.pairing.delta == ms(1, 1)
            for term in top
        ))


class TestResidualExpansion(unittest.TestCase):
    def test_isolated_operator_term_present(self):
        # The incremental state equation 0 = D^|beta| R Theta^beta contains exactly the term
        # d_u R uhat_beta  =  (ID, R, {}, {beta}), which the driver will isolate as A uhat_beta.
        for alpha in [ms(1), ms(1, 1), ms(1, 2)]:
            D = sym.differentiate_over_lattice(sym.seed_residual_r(), alpha)
            self.assertIn(Term(ID, 'R', ms(), Multiset([alpha])), D[alpha])


if __name__ == '__main__':
    unittest.main()
