#!/usr/bin/env python3


"""Computes BURP distance for two trees over the same sentence."""


import argparse
import levenshtein
import logging
import os
import re
import sys


from discodop.punctuation import PUNCTUATION
from discodop.tree import DrawTree, ParentedTree
from discodop.treebank import incrementaltreereader
from discodop.treetransforms import removeterminals
from typing import Any, Callable, Dict, FrozenSet, IO, Iterable, List, \
        Sequence, Tuple, Union


Action = str # for now
Script = List[Action]
Sentence = List[str]
Span = FrozenSet[int]
Subtree = Union[ParentedTree, int]


FUNCTION_PATTERN = re.compile(r'-.*')
NO_PARSE_PATTERN = re.compile(r'(?:# NO PARSE FOR: |\?\?# NO PARSE \+ )(?P<sentence>[^\n]+)$')


def ispunct(word, tag):
    return word in PUNCTUATION


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
        if value <= record:
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
            if d not in xchain1 and not span(d) <= target_span:
                cost += 1.0
    # Cost of chain editing
    labels1 = tuple(t.label for t in xchain1[:-1])
    labels2 = tuple(t.label for t in chain2)
    cost += levenshtein.distance(levenshtein.matrix(labels1, labels2))
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


def edit(xchain1: Tuple[ParentedTree, ...], chain2: Tuple[ParentedTree, ...], parts: List[ParentedTree], mapping: Dict[Span, ParentedTree], script: Script, sent: List[str]) -> None:
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
            script.append(f'move {t.label} to {xchain1[-2].label}')
            logging.debug(script[-1])
            if t.parent is None:
                parts.remove(t)
            else:
                t.detach()
            xchain1[-2].append(t)
            return
        if not is_preleaf(t):
            for d in tuple(t):
                move(d)
    for n in xchain1[:-2]:
        for d in tuple(n):
            if d not in xchain1:
                move(d)
    # Free above
    for t in xchain1[:-2]:
        for d in tuple(t):
            if not d in xchain1:
                parts.append(d.detach())
    # Edit chain
    labels1 = tuple(n.label for n in xchain1[:-1])
    labels2 = tuple(n.label for n in chain2)
    lev_script = levenshtein.script(levenshtein.matrix(labels1, labels2))
    chain = list(xchain1)
    i = 0
    for op in lev_script:
        if op == levenshtein.Op.DEL:
            script.append(f'delete {chain[i].label}')
            logging.debug(script[-1])
            if chain[i].parent is None:
                parts.remove(chain[i])
                parts.append(chain[i + 1].detach())
            else:
                chain[i].prune()
            del chain[i]
            logging.debug('Chain: %s', pp_chain(chain))
        elif op == levenshtein.Op.INS:
            script.append(f'insert {labels2[i]}')
            logging.debug(script[-1])
            if i == 0:
                # This case is *not* supported by the RRGparbank annotation
                # interface if chain[i] has siblings or is the root.
                if chain[i].parent is None:
                    parts.remove(chain[i])
                    node = ParentedTree(labels2[i], [chain[i]])
                    parts.append(node)
                else:
                    chain[i].spliceabove(labels2[i])
                chain[i:i] = [chain[i].parent]
            else:
                chain[i - 1].splicebelow(labels2[i])
                chain[i:i] = [chain[i - 1][0]]
            i += 1
            logging.debug('Chain: %s', pp_chain(chain))
        elif op == levenshtein.Op.SUB:
            script.append(f'relabel {chain[i].label} to {labels2[i]}')
            logging.debug(script[-1])
            chain[i].label = labels2[i]
            i += 1
            logging.debug('Chain: %s', pp_chain(chain))
        else:
            i += 1
    assert i == len(chain) - 1
    xchain1 = tuple(chain)
    # Move up
    for t in tuple(xchain1[-2]):
        if not span(t) <= target_span:
            move(t)
    # Free below
    for t in tuple(xchain1[-2]):
        if not span(t) <= target_span:
            parts.append(t.detach())
    # Move in 
    def move_in(t: ParentedTree) -> None:
        if t == xchain1[0]:
            return
        if span(t) <= target_span and not dominates(t, xchain1[0]):
            script.append(f'move {t.label} to {xchain1[-2].label}')
            logging.debug(script[-1])
            if t.parent is None:
                parts.remove(t)
            else:
                t.detach()
            xchain1[-2].append(t)
            return
        if is_preleaf(t):
            return
        for c in tuple(t):
            move_in(c)
    for part in tuple(parts):
        move_in(part)
    # Prune
    def prune(t: ParentedTree) -> None:
        if t in dtrs1 or is_preleaf(t):
            return
        for d in tuple(t):
            prune(d)
        if span(t) <= target_span:
            script.append(f'delete {t.label}')
            logging.debug(script[-1])
            t.prune()
    for d in tuple(xchain1[-2]):
        prune(d)
    # Sort children
    xchain1[-2].children.sort(key=lambda c: min(span(c)))
    chain2[-1].children.sort(key=lambda c: min(span(c)))
    # Assertions
    logging.debug('Source subtree:\n%s', pp_tree(xchain1[0], sent))
    logging.debug('Target subtree:\n%s', pp_tree(chain2[0], sent))
    assert xchain1[0] == chain2[0]
    # Record
    mapping[target_span] = xchain1[0]


def burp(tree1: ParentedTree, tree2: ParentedTree, sent: List[str]) -> Tuple[float, Script]:
    span1 = span(tree1)
    span2 = span(tree2)
    assert span1 == span2
    cost = 0.0
    parts = [tree1]
    logging.info('Source:\n%s', side_by_side(tuple(pp_tree(p, sent) for p in parts)))
    logging.info('Target:\n%s', pp_tree(tree2, sent))
    mapping: Dict[Span, Subtree] = {frozenset((i,)): i for i in span2}
    script: Script = []
    for chain2 in chains(tree2):
        logging.debug('chain2 : %s', pp_chain(chain2))
        xchain1, new_cost = argmin(
            xchains(chain2, parts, mapping),
            lambda x: edit_cost(x, chain2, parts, mapping)
        )
        cost += new_cost
        logging.debug('xchain1: %s', pp_chain(xchain1))
        edit(xchain1, chain2, parts, mapping, script, sent)
        logging.debug('Parts:\n%s', side_by_side(tuple(pp_tree(p, sent) for p in parts), 4))
        logging.debug('Target:\n%s', pp_tree(tree2, sent))
    # Assertions
    assert len(parts) == 1
    assert parts[0] == tree2
    # Return
    return cost, script


def pp_chain(chain: Sequence[Subtree]) -> str:
    """Pretty-print a chain
    """
    return ' '.join(
        s.label if isinstance(s, ParentedTree) else str(s)
        for s in chain
    )


def pp_tree(t: ParentedTree, sent: List[str]) -> str:
    """Pretty-print a tree
    """
    return str(DrawTree(t, sent))


def pp_node(t: Subtree) -> str:
    if isinstance(t, int):
        return str(t)
    return str(t.label)


def side_by_side(blocks: Sequence[str], padding: int=0) -> str:
    blks = tuple(pad(fixed_splitlines(b), padding) for b in blocks)
    widths = tuple(len(b[0]) for b in blks) # TODO grapheme cluster support
    heights = tuple(len(b) for b in blks)
    max_height = max(heights)
    # Pad blks at top
    for block, width, height in zip(blks, widths, heights):
        block[0:0] = [' ' * width] * (max_height - height)
    # Join
    return os.linesep.join(' '.join(p) for p in zip(*blks))


def fixed_splitlines(string: str) -> List[str]:
    """Like str.splitlines, but preserves empty last line."""
    result = string.splitlines()
    if string.endswith(os.linesep):
        result.append('')
    return result


def pad(block: Sequence[str], margin: int=0) -> List[str]:
    width = max(len(l) for l in block) + margin
    return [
        line + ' ' * (width - len(line))
        for line in block
    ] # TODO grapheme cluster support


def read_discbracket(f: IO) -> Iterable[Tuple[ParentedTree, Sentence, bool]]:
    for line in f:
        # skip empty lines
        if not line.rstrip():
            continue
        # recognize "no parse" lines and return dummy tree (TODO make this
        # configurable)
        match = NO_PARSE_PATTERN.match(line)
        if match:
            sent = match.group('sentence').split()
            sent = [w for w in sent if not ispunct(w, 'UH')]
            tree = ParentedTree('SENTENCE', [])
            for i in range(len(sent)):
                child = ParentedTree('UH', [i])
                tree.append(child)
            yield tree, sent, True
            continue
        # parse tree expression
        reader = incrementaltreereader(line)
        tree, sent, rest = next(reader)
        assert not rest
        # remove punctuation (TODO make this configurable)
        removeterminals(tree, sent, ispunct)
        # remove artifical root node (TODO make this configurable)
        if len(tree) == 1 and tree.label == 'ROOT':
            tree = tree[0].detach()
        # remove function tags (TODO make this configurable)
        for subtree in tree.subtrees():
            subtree.label = FUNCTION_PATTERN.sub('', subtree.label)
        # yield
        yield tree, sent, False


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('path1', help='source trees in .discbracket format')
    arg_parser.add_argument('path2', help='target trees in .discbracket format')
    arg_parser.add_argument('-v', '--verbose', action='count', default=0,
            help='Verbosity. Give once for printing trees, twice for debugging.')
    args = arg_parser.parse_args()
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    total_trees = 0
    total_consts = 0
    total_distance = 0.0
    not_parsed = 0
    with open(args.path1) as f1, open(args.path2) as f2:
        for (tree1, sent1, dummy1), (tree2, sent2, dummy2) in \
                zip(read_discbracket(f1), read_discbracket(f2)):
            assert sent1 == sent2
            assert not dummy2
            distance, script = burp(tree1, tree2, sent1)
            print(distance, script) # TODO make output format configurable
            total_trees += 1
            total_consts += sum(1 for _ in tree2.subtrees())
            total_distance += distance
            if dummy1:
                not_parsed += 1
    print(f'total trees:              {total_trees}')
    print(f'dummy trees:              {not_parsed}')
    print(f'total constituents:       {total_consts}')
    print(f'distance per tree:        {total_distance / total_trees}')
    print(f'distance per constituent: {total_distance / total_consts}')
