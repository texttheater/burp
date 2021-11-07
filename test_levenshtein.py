import levenshtein
import unittest


DEL = levenshtein.Op.DEL
INS = levenshtein.Op.INS
SUB = levenshtein.Op.SUB
MATCH = levenshtein.Op.MATCH


class LevenshteinTestCase(unittest.TestCase):

    def test_levenshtein(self):
        cases = (
            ('', 'a', 1.0, [INS]),
            ('a', 'aa', 1.0, [MATCH, INS]),
            ('a', 'aaa', 2.0, [MATCH, INS, INS]),
            ('', '', 0.0, []),
            ('a', 'b', 1.0, [SUB]),
            ('aaa', 'aba', 1.0, [MATCH, SUB, MATCH]),
            ('aaa', 'ab', 2.0, [MATCH, SUB, DEL]),
            ('a', 'a', 0.0, [MATCH]),
            ('ab', 'ab', 0.0, [MATCH, MATCH]),
            ('a', '', 1.0, [DEL]),
            ('aa', 'a', 1.0, [MATCH, DEL]),
            ('aaa', 'a', 2.0, [MATCH, DEL, DEL]),
            ('kitten', 'sitting', 3.0, [SUB, MATCH, MATCH, MATCH, SUB, MATCH, INS]),
            ('Orange', 'Apple', 5.0, [SUB, SUB, SUB, SUB, DEL, MATCH]),
            ('ab', 'bc', 2.0, [DEL, MATCH, INS]),
            ('abd', 'bec', 3.0, [DEL, MATCH, SUB, INS]),
        )
        for source, target, distance, script in cases:
            matrix = levenshtein.matrix(source, target)
            got_distance = levenshtein.distance(matrix)
            self.assertEqual(got_distance, distance)
            got_script = levenshtein.script(matrix)
            self.assertEqual(got_script, script)
