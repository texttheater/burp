from enum import Enum
from typing import Any, List, Sequence


DEL_COST = 1.0
INS_COST = 1.0
SUB_COST = 1.0


class Op(Enum):
    DEL = 1
    INS = 2
    SUB = 3
    MATCH = 4


Cost = float
Matrix = List[List[Cost]]
Script = List[Op]
String = Sequence[Any]


def matrix(source: String, target: String) -> Matrix:
    height = len(source) + 1
    width = len(target) + 1
    result: Matrix = [[0 for j in range(width)] for i in range(height)]
    for i in range(height):
        result[i][0] = i * DEL_COST
    for j in range(width):
        result[0][j] = j * INS_COST
    for i in range(1, height):
        for j in range(1, width):
            del_cost = result[i - 1][j] + DEL_COST
            match_sub_cost = result[i - 1][j - 1]
            if source[i - 1] != target[j - 1]:
                match_sub_cost += SUB_COST
            ins_cost = result[i][j - 1] + INS_COST
            result[i][j] = min((del_cost, match_sub_cost, ins_cost))
    return result


def distance(matrix: Matrix) -> Cost:
    return matrix[-1][-1]


def script(matrix: Matrix) -> Script:
    script: Script = []
    backtrace(len(matrix) - 1, len(matrix[0]) - 1, matrix, script)
    return script


def backtrace(i: int, j: int, matrix: Matrix, script: Script) -> None:
    if i > 0 and matrix[i - 1][j] + DEL_COST == matrix[i][j]:
        backtrace(i - 1, j, matrix, script)
        script.append(Op.DEL)
        return
    if j > 0 and matrix[i][j - 1] + INS_COST == matrix[i][j]:
        backtrace(i, j - 1, matrix, script)
        script.append(Op.INS)
        return
    if i > 0 and j > 0 and matrix[i - 1][j - 1] + SUB_COST == matrix[i][j]:
        backtrace(i - 1, j - 1, matrix, script)
        script.append(Op.SUB)
        return
    if i > 0 and j > 0 and matrix[i - 1][j - 1] == matrix[i][j]:
        backtrace(i - 1, j - 1, matrix, script)
        script.append(Op.MATCH)
        return
