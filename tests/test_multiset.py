# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
import math
import unittest

from implicit_probing.multiset import Multiset, subset_lattice


class TestMultiset(unittest.TestCase):
    def test_construction_and_counts(self):
        m = Multiset([1, 1, 2])
        self.assertEqual(m.count(1), 2)
        self.assertEqual(m.count(2), 1)
        self.assertEqual(m.count(3), 0)
        self.assertEqual(len(m), 3)
        self.assertEqual(m.cardinality, 3)
        self.assertEqual(len(Multiset()), 0)

    def test_from_counts(self):
        self.assertEqual(Multiset.from_counts({1: 2, 2: 1}), Multiset([1, 1, 2]))
        # non-positive counts are dropped
        self.assertEqual(Multiset.from_counts({1: 0, 2: 1, 3: -4}), Multiset([2]))

    def test_equality_and_hash_order_independent(self):
        self.assertEqual(Multiset([1, 2, 1]), Multiset([1, 1, 2]))
        self.assertNotEqual(Multiset([1, 1]), Multiset([1, 2]))
        self.assertEqual(hash(Multiset([2, 1])), hash(Multiset([1, 2])))
        # usable as a dict key
        d = {Multiset([1, 1]): 'a'}
        self.assertEqual(d[Multiset([1, 1])], 'a')

    def test_distinct_and_items_deterministic(self):
        m = Multiset([2, 1, 1])
        self.assertEqual(m.distinct(), (1, 2))
        self.assertEqual(m.items(), ((1, 2), (2, 1)))

    def test_contains(self):
        m = Multiset([1, 1])
        self.assertIn(1, m)
        self.assertNotIn(2, m)

    def test_add_remove_return_new(self):
        m = Multiset([1, 1])
        self.assertEqual(m.add(1), Multiset([1, 1, 1]))
        self.assertEqual(m.add(2), Multiset([1, 1, 2]))
        self.assertEqual(m.add(2, 3), Multiset([1, 1, 2, 2, 2]))
        self.assertEqual(m.remove(1), Multiset([1]))
        self.assertEqual(m.remove(1, 2), Multiset())
        # original is untouched (immutability)
        self.assertEqual(m, Multiset([1, 1]))

    def test_remove_too_many_raises(self):
        with self.assertRaises(ValueError):
            Multiset([1]).remove(2)
        with self.assertRaises(ValueError):
            Multiset([1]).remove(1, 2)

    def test_sum_and_difference(self):
        self.assertEqual(Multiset([1, 1]) + Multiset([1, 2]), Multiset([1, 1, 1, 2]))
        self.assertEqual(Multiset([1, 1, 2]) - Multiset([1, 3]), Multiset([1, 2]))  # clamped: 3 absent
        self.assertEqual(Multiset([1]) - Multiset([1, 1]), Multiset())              # clamped at zero

    def test_submultiset(self):
        alpha = Multiset([1, 1, 2])
        self.assertTrue(Multiset().issubmultiset(alpha))
        self.assertTrue(Multiset([1]).issubmultiset(alpha))
        self.assertTrue(Multiset([1, 1]).issubmultiset(alpha))
        self.assertTrue(alpha.issubmultiset(alpha))                 # non-strict
        self.assertFalse(Multiset([1, 1, 1]).issubmultiset(alpha))  # too many 1s
        self.assertFalse(Multiset([3]).issubmultiset(alpha))
        self.assertTrue(Multiset([1]) <= Multiset([1, 1]))

    def test_nested_multisets(self):
        inner = Multiset([1])
        gamma = Multiset([inner, inner])  # the bag {{1}, {1}}
        self.assertEqual(gamma.count(Multiset([1])), 2)
        self.assertEqual(len(gamma), 2)
        raised = gamma.remove(Multiset([1])).add(Multiset([1, 1]))
        self.assertEqual(raised, Multiset([Multiset([1]), Multiset([1, 1])]))

    def test_any_element(self):
        self.assertEqual(Multiset([5, 5]).any_element(), 5)
        self.assertIn(Multiset([7, 9]).any_element(), {7, 9})
        with self.assertRaises(ValueError):
            Multiset().any_element()

    def test_subset_lattice_size_is_product(self):
        # |lattice| == prod(count_i + 1)
        cases = [
            (Multiset(), 1),
            (Multiset([1]), 2),
            (Multiset([1, 1]), 3),
            (Multiset([1, 2]), 4),
            (Multiset([1, 1, 1]), 4),
            (Multiset([1, 1, 2]), 6),
            (Multiset([1, 2, 3]), 8),
            (Multiset([1, 1, 2, 2]), 9),
        ]
        for alpha, expected in cases:
            with self.subTest(alpha=alpha):
                lattice = subset_lattice(alpha)
                self.assertEqual(len(lattice), expected)
                predicted = math.prod(c + 1 for _, c in alpha.items())
                self.assertEqual(len(lattice), predicted)

    def test_subset_lattice_contents_and_order(self):
        alpha = Multiset([1, 1, 2])
        lattice = subset_lattice(alpha)
        # no duplicates; contains the empty multiset and alpha itself
        self.assertEqual(len(set(lattice)), len(lattice))
        self.assertIn(Multiset(), lattice)
        self.assertIn(alpha, lattice)
        # every node is a sub-multiset of alpha
        for beta in lattice:
            self.assertTrue(beta.issubmultiset(alpha))
        # nondecreasing cardinality
        sizes = [len(beta) for beta in lattice]
        self.assertEqual(sizes, sorted(sizes))
        # exact membership
        expected = {
            Multiset(), Multiset([1]), Multiset([2]),
            Multiset([1, 1]), Multiset([1, 2]), Multiset([1, 1, 2]),
        }
        self.assertEqual(set(lattice), expected)

    def test_subset_lattice_symmetric_is_a_chain(self):
        # fully symmetric alpha = {1,1,1}: the lattice is the chain {}, {1}, {1,1}, {1,1,1}
        lattice = subset_lattice(Multiset([1, 1, 1]))
        self.assertEqual(
            list(lattice),
            [Multiset(), Multiset([1]), Multiset([1, 1]), Multiset([1, 1, 1])],
        )


if __name__ == '__main__':
    unittest.main()
