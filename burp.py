#!/usr/bin/env python3


import levenshtein
import sys


from discodop.tree import ParentedTree
from discodop.treebank import DiscBracketCorpusReader
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Tuple, Union


Span = FrozenSet[int]
Subtree = Union[ParentedTree, int]


def span(t: Subtree) -> Span:
    if isinstance(t, int):
        return frozenset((t,))
    return frozenset(t.leaves())


def parent(s: Subtree, t: ParentedTree) -> Union[ParentedTree, None]:
    if isinstance(s, ParentedTree):
        return s.parent
    for p in t.subtrees():
        if s in p:
            return p
    raise ValueError(f'{s} not in {t}')


def children(s: Subtree) -> Iterable[Subtree]:
    if isinstance(s, ParentedTree):
        return iter(s)
    return iter(())


def is_preleaf(t: ParentedTree) -> bool:
    assert len(t) > 0
    if any(isinstance(c, int) for c in t):
        assert len(t) == 1
        return True
    return False


def dominates(t: Subtree, s: Subtree) -> bool:
    if isinstance(t, ParentedTree):
        return t == s or any(dominates(c, s) for c in t)
    return t == s


def argmin(xs: Iterable[Any], f: Callable[[Any], float]) -> Tuple[Any, float]:
    record = float('Inf')
    record_holder = None
    for x in xs:
        value = f(x)
        if value < record:
            record = value
            record_holder = x
    return record_holder, record


def chains(tree2: ParentedTree) -> Iterable[Tuple[ParentedTree, ...]]:
    """Returns a post-order traversal of the unary chains in tree2.

    A unary chain is a sequence of all nodes with the same span, ordered from
    highest to lowest.
    """
    if not is_preleaf(tree2):
        for child in tree2:
            yield from chains(child)
    if tree2.parent != None and len(tree2.parent) == 1:
        return
    chain = [tree2]
    while len(tree2) == 1 and not is_preleaf(tree2):
        chain.append(tree2[0])
        tree2 = tree2[0]
    yield tuple(chain)


def xchains(chain2: Tuple[ParentedTree, ...], parts: List[ParentedTree], mapping: Dict[Span, ParentedTree]) -> Iterable[Tuple[Subtree, ...]]:
    """Returns candidate extended chains to transform into chain2.
    
    An extended chain is some path in the original tree that ends with a node
    n such that chain2[-1] has a daughter d with mapping[span(d)] == n."""
    span2 = span(chain2[-1])
    for dtr2 in chain2[-1]:
        dtr2_span = span(dtr2)
        dtr1 = mapping[dtr2_span]
        xchain1 = [dtr1]
        for part in parts:
            if span(dtr1) <= span(part):
                next1 = parent(dtr1, part)
                while True:
                    # A chain is complete when we reach the root of part (then we are
                    # done):
                    if next1 is None:
                        yield tuple(xchain1)
                        break
                    # We also yield intermediate chains that don't go all the way up to
                    # the root of tree1, but only if chain2[0] isn't the root of tree2
                    # (in that case we need to use up all unmapped nodes *now*).
                    if chain2[0].parent is not None:
                        yield tuple(xchain1)
                    xchain1.insert(0, next1)
                    next1 = next1.parent


def edit_cost(xchain1: Tuple[ParentedTree, ...], chain2: Tuple[ParentedTree, ...], parts: List[ParentedTree], mapping: Dict[Span, ParentedTree]) -> float:
    cost = 0.0
    # Find target span
    target_span = span(chain2[-1])
    # Find already-mapped daughters
    dtrs1 = tuple(
        mapping[span(d)]
        for d in chain2[-1]
    )
    # Cost of moving down
    def move_cost(t: ParentedTree) -> float:
        if span(t) <= target_span:
            return 1.0
        return sum(move_cost(d) for d in children(t))
    for n in xchain1[:-2]:
        for d in n:
            if d not in xchain1:
                cost += move_cost(d)
    # Cost of freeing above
    for t in xchain1[:-2]:
        for d in t:
            if not span(d) <= target_span:
                cost += 1.0
    # Cost of chain editing
    labels1 = tuple(t.label for t in xchain1[:-1])
    labels2 = tuple(t.label for t in chain2)
    cost += levenshtein.levenshtein(labels1, labels2)
    # Cost of moving up
    if len(xchain1) > 1:
        for t in xchain1[-2]:
            if not span(t) <= target_span:
                cost += move_cost(t)
    # Cost of freeing below
    if len(xchain1) > 1:
        for t in xchain1[-2]:
            if not span(t) <= target_span:
                cost += 1.0
    # Cost of moving in
    def move_in_cost(t: ParentedTree) -> float:
        if t == xchain1[0]:
            return 0.0
        if span(t) <= target_span and not dominates(t, xchain1[0]):
            return 1.0
        if is_preleaf(t):
            return 0.0
        return sum(move_in_cost(c) for c in t)
    for part in parts:
        if not span(part) <= target_span: # already freed
            cost += move_in_cost(part)
    # Cost of pruning
    def prune_cost(t: Subtree) -> float:
        if t in dtrs1:
            return 0.0
        if span(t) <= target_span and not dominates(t, xchain1[-1]):
            return 1.0 + sum(prune_cost(d) for d in children(t))
        return sum(prune_cost(d) for d in children(t))
    cost += sum(prune_cost(p) for p in parts)
    # Return
    return cost


def edit(xchain1: Tuple[ParentedTree, ...], chain2: Tuple[ParentedTree, ...], parts: List[ParentedTree], mapping: Dict[Span, ParentedTree]) -> None:
    # Find target span
    target_span = span(chain2[-1])
    # Find already-mapped daughters
    dtrs1 = tuple(
        mapping[span(d)]
        for d in chain2[-1]
    )
    # Move down
    def move(t: ParentedTree) -> None:
        if span(t) <= target_span:
            if t.parent is None:
                parts.remove(t)
            else:
                t.detach()
            xchain1[-2].append(t)
            return
        if not is_preleaf(t):
            for d in t:
                move(d)
    for n in xchain1[:-2]:
        for d in n:
            if d not in xchain1:
                move(d)
    # Free above
    for t in xchain1[:-2]:
        for d in t:
            if not span(d) <= target_span:
                parts.append(d.detach())
    # Create new chain
    new_chain1 = [ParentedTree(chain2[-1].label, [])]
    for n in chain2[-2::-1]:
        new_chain1.insert(0, ParentedTree(n.label, [new_chain1[0]]))
    # Replace old with new chain
    if xchain1[0].parent is None:
        parts.append(new_chain1[0])
        parts.remove(xchain1[0])
    else:
        xchain1[0].parent.append(new_chain1[0])
        xchain1[0].parent.remove(xchain1[0])
    # Move daughters to new chain
    if len(xchain1) > 1:
        for d in tuple(xchain1[-2]):
            xchain1[-2].remove(d)
            new_chain1[-1].append(d)
    else:
        new_chain1[-1].append(xchain1[-1])
    # Update xchain1
    xchain1 = tuple(new_chain1) + xchain1[-1:]
    # Move up
    for t in xchain1[-2]:
        if not span(t) <= target_span:
            move(t)
    # Free below
    for t in xchain1[-2]:
        if not span(t) <= target_span:
            parts.append(t.detach())
    # Move in 
    def move_in(t: ParentedTree) -> None:
        if t == new_chain1[0]:
            return
        if span(t) <= target_span and not dominates(t, new_chain1[0]):
            if t.parent is None:
                parts.remove(t)
            else:
                t.detach()
            xchain1[-2].append(t)
            return
        if is_preleaf(t):
            return
        for c in t:
            move_in(c)
    for part in tuple(parts):
        move_in(part)
    # Prune
    def prune(t: ParentedTree) -> None:
        if t in dtrs1 or is_preleaf(t):
            return
        for d in t:
            prune(d)
        if span(t) <= target_span:
            t.prune()
    for d in xchain1[-2]:
        prune(d)
    # Sort children
    xchain1[-2].children.sort(key=lambda c: min(span(c)))
    chain2[-1].children.sort(key=lambda c: min(span(c)))
    # Assertions
    assert new_chain1[0] == chain2[0]
    # Record
    mapping[target_span] = new_chain1[0]


def burp(tree1: ParentedTree, tree2: ParentedTree) -> float:
    span1 = span(tree1)
    span2 = span(tree2)
    assert span1 == span2
    cost = 0.0
    parts = [tree1]
    mapping: Dict[Span, Subtree] = {frozenset((i,)): i for i in span2}
    for chain2 in chains(tree2):
        xchain1, new_cost = argmin(
            xchains(chain2, parts, mapping),
            lambda x: edit_cost(x, chain2, parts, mapping)
        )
        cost += new_cost
        edit(xchain1, chain2, parts, mapping)
    # Assertions
    assert len(parts) == 1
    assert parts[0] == tree2
    # Return
    return cost


def pp_chain(chain: Tuple[Subtree, ...]) -> str:
    return ' '.join(
        s.label if isinstance(s, ParentedTree) else str(s)
        for s in chain
    )


def pp_node(t: Subtree) -> str:
    if isinstance(t, ParentedTree):
        return t.label
    return str(t)


if __name__ == '__main__':
    try:
        _, path1, path2 = sys.argv
    except ValueError:
        print('USAGE: python3 burp.py t1.discbracket t2.discbracket',
                file=sys.stderr)
        sys.exit(1)
    inp1 = DiscBracketCorpusReader(path1)
    inp2 = DiscBracketCorpusReader(path2)
    for t1, t2 in zip(inp1.itertrees(), inp2.itertrees()):
        key1, item1 = t1
        key2, item2 = t2
        assert key1 == key2
        tree1 = item1.tree
        tree2 = item2.tree
        assert sorted(tree1.leaves()) == sorted(tree2.leaves())
        print(burp(tree1, tree2))
