import burp
import unittest


from discodop.tree import ParentedTree


class BurpTestCase(unittest.TestCase):

    def test_burp(self):
        cases = (
            ('(A (B 0))', '(A 0)', 1.0),
        )
        for tree1, tree2, distance in cases:
            tree1 = ParentedTree(tree1)
            tree2 = ParentedTree(tree2)
            self.assertEqual(burp.burp(tree1, tree2), distance)
