# List Routines DSL ([Symbols](#symbols) &middot; [Definitions](#definitions)  &middot; [Lambdas](#lambdas) &middot; [Types](#type-system))

This document describes a pair of Domain-specific languages (DSLs) for list manipulation routines following a lisp-like syntax. They are deliberately sparse. The two DSLs differ only in the set of numbers they make available. The smaller DSL contains only the integers 0...9, while the larger DSL contains 0...99.

## Symbols

Below are the symbols in the DSL, their arities, their types, and a brief description.

Notes:
- The DSL supports recursion using `fix` (e.g. `(lambda (fix $0 (lambda (lambda (if (empty? $0) 0 (+ 1 ($1 (tail $0))))))))`). You may instead use explicit recursion (e.g. `c empty = 0; c (cons x y) = (+ 1 (c y));`) if that is more natural for your representation.
- The DSL supports abstraction using `lambda`. If your representation supports another form of abstraction, you may drop `lambda`.
- For applicative representations (e.g. combinatory logic), set all arities to 0 and add an application symbol of type `(t1 -> t2) -> t1 -> t2`.

<table>
  <col>
  <col>
  <col>
  <col>
<thead>
<tr class="header">
<th><strong>Symbol</strong></th>
<th><strong>Arity</strong></th>
<th><strong>Type</strong></th>
<th><strong>Description</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><code>0</code>...<code>9</code></td>
<td>0</td>
<td><code>int</code></td>
<td>integers 0&ndash;9, inclusive.</td>
</tr>
<tr>
<td><code>10</code>...<code>99</code></td>
<td>0</td>
<td><code>int</code></td>
<td>integers 10&ndash;99, inclusive. <em>Problems 81&ndash;100 only.</em></td>
</tr>
<tr>
<td><code>nan</code></td>
<td>0</td>
<td><code>int</code></td>
<td>An out-of-bounds number.</td>
</tr>
<tr>
<td><code>true</code></td>
<td>0</td>
<td><code>bool</code></td>
<td>Boolean literal.</td>
</tr>
<tr>
<td><code>false</code></td>
<td>0</td>
<td><code>bool</code></td>
<td>Boolean literal.</td>
</tr>
<tr>
<td><code>empty</code></td>
<td>0</td>
<td><code>[t1]</code></td>
<td>an empty list.</td>
</tr>
<tr>
<td><code>cons</code></td>
<td>2</td>
<td><code>t1 → [t1] → [t1]</code></td>
<td>Prepends a given item to the beginning of a list.</td>
</tr>
<tr>
<td><code>+</code></td>
<td>2</td>
<td><code>int → int → int</code></td>
<td>Binary addition operator.</td>
</tr>
<tr>
<td><code>-</code></td>
<td>2</td>
<td><code>int → int → int</code></td>
<td>Binary subtraction operator.</td>
</tr>
<tr>
<td><code>&gt;</code></td>
<td>2</td>
<td><code>int → int → bool</code></td>
<td>Binary greater than predicate.</td>
</tr>
<tr>
<td><code>fix</code></td>
<td>2</td>
<td><code>t0 → ((t0 → t1) → t0 → t1) → t1</code></td>
<td>fix-point operator (for recursion)</td>
</tr>
<tr>
<td><code>head</code></td>
<td>1</td>
<td><code>[t1] → t1</code></td>
<td>Returns the first element of a list (i.e. the head).</td>
</tr>
<tr>
<td><code>if</code></td>
<td>3</td>
<td><code>bool → t1 → t1 → t1</code></td>
<td>standard conditional</td>
</tr>
<tr>
<td><code>is_empty</code></td>
<td>1</td>
<td><code>[t1] → bool</code></td>
<td><code>true</code> if the list is empty, else <code>false</code></td>
</tr>
<tr>
<td><code>is_equal</code></td>
<td>2</td>
<td><code>t1 → t1 → bool</code></td>
<td><code>true</code> if the arguments are identical, else <code>false</code></td>
</tr>
<tr>
<td><code>lambda</code></td>
<td>1</td>
<td></td>
<td>a mechanism for creating anonymous functions.</td>
</tr>
<tr>
<td><code>tail</code></td>
<td>1</td>
<td><code>[t1] → [t1]</code></td>
<td>Returns all but the first element of a list (i.e. the tail).</td>
</tr>
</tbody>
</table>

*Table 1.0 - DSL symbols*

## Definitions

Below are definitions for the symbols in the DSL.

- `0`...`99`, `nan`, `true`, `false`, `empty`, and `cons`-cells are constants.
- `lambda` is described [below](#lambdas).
- Use standard definitions for `+`, `-`, and `>`; out-of-bounds operations go to `nan` (e.g. `+ 9 9 = nan`, `- 2 3 = nan`) .
- The remaining symbols follow these rules:
  ```
  fix f x = f (fix f) x;

  head (cons x y) = x;

  if true  x y = x;
  if false x y = y;

  is_empty empty = true;
  is_empty (cons x y) = false;

  is_equal x x = true;
  is_equal x y = false;

  tail (cons x y) = y;
  ```
## Lambdas

`lambda` returns an anonymous function that runs an input expression when called. The $-prefixed integers (e.g. `$0`, `$1`, … `$n`) represent [De Bruijn indices](https://en.wikipedia.org/wiki/De_Bruijn_index); the index N refers to the argument N variable bindings away from the current binding.

| **Example**                     | **Type**  | **Description**                                                 |
| ------------------------------- | ------------------- | --------------------------------------------------------------- |
| `(lambda 5)`                     | `(t1 → int)`         | Returns 5.                                                      |
| `(lambda (+ $0 1))`              | `(int → int)`        | Increments an input by 1.                                 |
| `(lambda (> $0 0))`             | `(int → int)`        | tests whether the input is greater than 0.  |
| `(lambda (lambda $1))`    | `(t1 → t2 → t1)`  | Returns the first input, i.e. the K-combinator. |

*Table 1.1 - Lambda Examples*

## Type System

This DSL uses a [Hindley-Milner type system](https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system).

<table>
<thead>
<tr class="header">
<th><strong>Symbol</strong></th>
<th><strong>Description</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><code>t1,t2,...</code></td>
<td>Universally quantified type variables.</td>
</tr>
<tr>
<td><code>→</code></td>
<td>Arrow type. Left hand side of arrow represents input types, right represents output type. Chaining of arrows represents multiple function arguments, e.g. a function that takes two <code>int</code>s and returns an <code>int</code> would be <code>int → int → int</code>.</td>
</tr>
<tr>
<td><code>int</code></td>
<td>Integer value.</td>
</tr>
<tr>
<td><code>bool</code></td>
<td>Boolean value.</td>
</tr>
<tr>
<td><code>[&lt;type&gt;]</code></td>
<td>List where each value is of type &lt;type&gt;. E.g.:
<ul style="margin-bottom: 0;">
<li><code>[t1]</code> - List of values of type <code>t1</code>.</li>
<li><code>[int]</code> - List of integers.</li>
<li><code>[[int]]</code> - List of lists of integers.</li></ul></td>
</tr>
</tbody>
</table>

*Table 1.2 - Type Definitions*
