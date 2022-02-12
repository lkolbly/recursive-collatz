Recursive Collatz
=================

This is a toy repo "I" built (in actuality, it's mostly copied from the [wgpu hello-compute example](https://github.com/gfx-rs/wgpu/tree/master/wgpu/examples/hello-compute)) with two purposes:
- Primarily, a toy project to get my hands dirty with wgpu, and
- secondly, to investigate a problem which I call the "recursive collatz" problem.

Running
=======

1. [Install Rust](https://www.rust-lang.org/learn/get-started)
2. Run `$ cargo run --release`
3. Profit!

The program will print out the resulting fixed points (see below) in a line like this:
```
Steps: [5, OVERFLOW, 0]
```
which indicates that the two fixed points are 0 and 5. `OVERFLOW` indicates that an internal overflow occurred.

As a disclaimer, since this is mostly just a toy repo, I haven't actually spent the time to debug, test, and fully validate the correctness of this code.

The Problem
===========

So, we're all familiar with the [Collatz Conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture). To resummarize it here, the conjecture is that if you start with any integer greater than zero, and recursively apply the function `collatz_step(x)`:
```
if x % 2 == 0:
    return x / 2
else:
    return x * 3 + 1
```
then you will eventually reach the cycle `1 -> 4 -> 2 -> 1`. Equivalently, the conjecture can be stated that you eventually reach the number `1`.

For example, starting with the number `6`, the cycle is: `6 -> 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1`.

This is all fine and good, and is an active area of research mathematics (I guess). However, we can take this one step further, to what I call the Recursive Collatz Conjecture. In this conjecture, define the function `collatz(N)` as the number of iterations of the above "collatz operation" required before arriving at the number `1`. For example, `collatz(32)` is `5` because the operation requires `5` iterations to get to `1`: `32 -> 16 -> 8 -> 4 -> 2 -> 1`. `collatz(1)` is `0`. In general, if `collatz_step(a) == b` and `collatz(b) == N` then `collatz(a) == N + 1`.

You can, additionally, consider this function as potentially having both fixed points and cycles. A fixed point would be an `N` such that `collatz(N) == N`. A cycle would be a sequence of numbers `N1, N2, ... Nn` such that `collatz(N1) == N2, collatz(N2) == N3, ... collatz(N_(n-1)) == collatz(Nn)`.

The Recursive Collatz Conjecture is then twofold:
* There exists only one fixed point for this function: `5`, with the five-long collatz chain `5 -> 16 -> 8 -> 4 -> 2 -> 1`.
* There exist no cycles.
