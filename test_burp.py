import burp
import unittest


from discodop.tree import ParentedTree


class BurpTestCase(unittest.TestCase):

    def test_burp(self):
        cases = (
            (
                '(A 0)',
                '(A 0)',
                0.0,
            ),
            (
                '(A (B 0))',
                '(A 0)',
                1.0,
            ),
            (
                '(A (B (C (F 0))))',
                '(B (D (E (F 0))))',
                3.0,
            ),
            (
                '(A (B (C (F 0) (G 1))))',
                '(B (D (E (F 0) (G 1))))',
                3.0,
            ),
            (
                '(A (B (C (F 0) (G 1) (H (I 2)))))',
                '(B (D (E (F 0) (G 1) (I 2))))',
                4.0,
            ),
            (
                '(A (B (C (F 0) (G 1) (H (I 2))) (J 3)))',
                '(B (D (E (F 0) (G 1) (I 2) (J 3))))',
                5.0,
            ),
            (
                '(A (B (C (F 0) (G 1) (H (I 2))) (J (K 3) (L 4))))',
                '(B (D (E (F 0) (G 1) (I 2) (K 3) (L 4))))',
                6.0,
            ),
        )
        for tree1, tree2, distance in cases:
            tree1 = ParentedTree(tree1)
            tree2 = ParentedTree(tree2)
            self.assertEqual(burp.burp(tree1, tree2), distance)
