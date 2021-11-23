burp
====

BURP (“Bottom-up Replugging”) is a tree edit distance measure for trees with
the same leaves, such as two parse trees for the same sentence. BURP indicates
the estimated number of node reattachment, insertion, deletion, and relabeling
operations needed to transform one tree into the other. This repository
contains an implementation under development.

Dependencies
------------

* [disco-dop](https://github.com/andreasvc/disco-dop)

Usage example
-------------

    python3 burp.py example_nofunc_predicted.bracket example_nofunc_gold.bracket
