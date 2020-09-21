# List-Routines Model Comparison: Wave 3.1 Concepts

Below is the list of functions used during Wave 3.1 of our model comparison experiments and simulations. Inputs and outputs are assumed to be `[int]` with 0 to 10 elements of values 0&ndash;9 or 0&ndash;99 (i.e. each concept is `[int] → [int]`). [The DSL used in the simulations][] is small, so I use [a richer DSL][] here to provide more readable names.

For each function, we've generated 5 unique orderings of a single set of 11 machine-generated examples. In the list below, each function links to a representative ordering. The rest can be found [here](./json).

[the DSL used in the simulations]: ./dsl.md
[a richer DSL]: https://github.com/joshrule/list-routines-human-experiments/blob/master/dsl.md

## Functions ([P1](#p1-11-%C2%B7-12-%C2%B7-13-%C2%B7-14) &middot; [P2](#p2))

The functions can be broken into two subsets, P1 and P2, that provide a total of 17 problems to solve.

### P1

P1 contains 15 problems and only uses the numbers 0&ndash;9.

- [`(lambda (if (> 3 (length $0)) empty (singleton (third $0))))`](./json/c002_1.json)
- [`(lambda (singleton (nth (first $0) (drop 1 $0))))`](./json/c005_1.json)
- [`(lambda (slice (first $0) (second $0) (drop 2 $0)))`](./json/c015_1.json)
- [`(lambda (cut_idx (if (== (first $0) (second $0)) 2 3) $0))`](./json/c027_1.json)
- [`(lambda (cut_idx (if (> (first $0) (second $0)) 2 3) $0))`](./json/c028_1.json)
- [`(lambda (droplast 2 $0))`](./json/c030_1.json)
- [`(lambda ((if (== (first $0) (second $0)) drop droplast) 2 $0))`](./json/c031_1.json)
- [`(lambda ((if (> (first $0) (last $0)) drop droplast) 2 $0))`](./json/c032_1.json)
- [`(lambda (if (== (second $0) (third $0)) (swap 1 4 $0) (swap 2 3 $0)))`](./json/c035_1.json)
- [`(lambda (if (> (second $0) (third $0)) (swap 2 3 $0) (swap 1 4 $0)))`](./json/c036_1.json)
- [`(lambda (if (== (length $0) 3) (append $0 3) (if (== (length $0) 9) (append $0 9) $0)))`](./json/c039_1.json)
- [`(lambda (if (is_in $0 3) (append $0 3) (if (is_in $0 9) (append $0 9) $0)))`](./json/c040_1.json)
- [`(lambda (swap 3 1 (replace 4 4 (cut_idx 6 (take 7 $0)))))`](./json/c060_1.json)
- [`(lambda (cons (max $0) (cons (last $0) (cons (length $0) (cons (first $0) (singleton (min $0)))))))`](./json/c076_1.json)
- [`(lambda (singleton (max $0)))`](./json/c078_1.json)

### P2

P1 contains 2 problems and uses the numbers 0&ndash;99.

- [`(lambda (if (== (second $0) (third $0)) (swap 1 4 $0) (swap 2 3 $0)))`](./json/c088_1.json)
- [`(lambda (if (> (second $0) (third $0)) (swap 2 3 $0) (swap 1 4 $0)))`](./json/c089_1.json)
