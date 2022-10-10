# Parallelogram

Parallelogram is a Rust implementation of the algorithm described in [Text-Translation Alignment](https://aclanthology.org/J93-1006.pdf) (Kay, M. and Röscheisen, M., 1994. Text-translation alignment. Computational linguistics, 19(1), pp.121-142.).

First, implement the `Sentence` trait for whatever type represents the sentences you wish to align. For example, if the sentences of our text are represented by vectors of string slices we could provide the following implementation:

```rust
struct Sentence<'a>(Vec<&'a str>);

impl<'a> parallelogram::Sentence<&'a str> for Sentence<'a> {
    fn words(&self) -> &[&'a str] {
        &self.0
    }
}
```

Next, configure the algorithm and provide the text to be aligned:

```rust
let a: Vec<Sentence> = tale_of_two_cities_en;
let b: Vec<Sentence> = tale_of_two_cities_de;

let config = parallelogram::Config::default();
let output = config.align(&a, &b);

assert_eq!(
    output.a_alignments(0).next().unwrap().0.join(" "),
    "es war die beste zeit es war die schlechteste zeit"
);
assert_eq!(
    output.b_alignments(0).next().unwrap().0.join(" "),
    "it was the best of times it was the worst of times"
);

assert_eq!(
    output.a_alignments(a.len() - 1).next().unwrap().0.join(" "),
    "die ruhe zu der ich eingehe ist viel seliger als ich sie jemals gekannt habe"
);
assert_eq!(
    output.b_alignments(b.len() - 1).next().unwrap().0.join(" "),
    "it is a far far better rest that I go to than I have ever known"
);
```
