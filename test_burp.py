import burp
import unittest


from discodop.tree import ParentedTree


class BurpTestCase(unittest.TestCase):

    def test_burp(self):
        cases = (
            # Identical trees
            (
                '(A 0)',
                '(A 0)',
                0.0,
            ),
            # Insert B
            (
                '(A (B 0))',
                '(A 0)',
                1.0,
            ),
            # Delete A, relabel C to D, insert E
            (
                '(A (B (C (F 0))))',
                '(B (D (E (F 0))))',
                3.0,
            ),
            # Same with an additional leaf
            (
                '(A (B (C (F 0) (G 1))))',
                '(B (D (E (F 0) (G 1))))',
                3.0,
            ),
            # Insert H
            (
                '(A (B (C (F 0) (G 1) (H (I 2)))))',
                '(B (D (E (F 0) (G 1) (I 2))))',
                4.0,
            ),
            # Move J down
            (
                '(A (B (C (F 0) (G 1) (H (I 2))) (J 3)))',
                '(B (D (E (F 0) (G 1) (I 2) (J 3))))',
                5.0,
            ),
            # Move J down, then delete it
            (
                '(A (B (C (F 0) (G 1) (H (I 2))) (J (K 3) (L 4))))',
                '(B (D (E (F 0) (G 1) (I 2) (K 3) (L 4))))',
                6.0,
            ),
            # Same but J is moved down from A instead of B
            (
                '(A (B (C (F 0) (G 1) (H (I 2)))) (J (K 3) (L 4)))',
                '(B (D (E (F 0) (G 1) (I 2) (K 3) (L 4))))',
                6.0,
            ),
            # Same but there is no J node, instead K and L are moved down
            # separately
            (
                '(A (B (C (F 0) (G 1) (H (I 2))) (K 3)) (L 4))',
                '(B (D (E (F 0) (G 1) (I 2) (K 3) (L 4))))',
                6.0,
            ),
            # The following test cases demonstrate the difference between
            # moving down and moving in. In the first case, D is moved down
            # within the A B C chain. In the second case, D is moved into the
            # B C chain.
            (
                '(A (B (C 0)) (D 1))',
                '(A (B (C 0) (D 1)))',
                1.0,
            ),
            (
                '(A (B (C 0)) (D 1) (E 2))',
                '(A (B (C 0) (D 1)) (E 2))',
                1.0,
            ),
            # The following two test cases demonstrate asymmetry: grouping two
            # siblings (that have other siblings) under a new node is harder
            # (insert, move) than ungrouping them (delete).
            (
                '(A (K 0) (L 1) (M 2))',
                '(A (J (K 0) (L 1)) (M 2))',
                2.0,
            ),
            (
                '(A (J (K 0) (L 1)) (M 2))',
                '(A (K 0) (L 1) (M 2))',
                1.0,
            ),
            # On the other hand, if there are no other siblings, an insert
            # action suffices.
            (
                '(A (K 0) (L 1))',
                '(A (J (K 0) (L 1)))',
                1.0,
            ),
        )
        for tree1, tree2, distance in cases:
            tree1 = ParentedTree(tree1)
            tree2 = ParentedTree(tree2)
            self.assertEqual(burp.burp(tree1, tree2), distance)
