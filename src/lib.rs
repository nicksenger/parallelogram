use std::cmp::Reverse;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;

mod alignable_sentence_table;
mod sentence_alignment_table;
mod word_association;
mod word_association_table;
mod word_sentence_index;

use alignable_sentence_table::AlignableSentenceTable;
use sentence_alignment_table::{Score, SentenceAlignmentTable};
use word_association::WordAssociation;
use word_association_table::WordAssociationTable;
use word_sentence_index::WordSentenceIndex;

pub trait Sentence<Word: PartialEq> {
    fn words(&self) -> &[Word];
}

pub struct Output<'a, T, U> {
    a: &'a [T],
    b: &'a [U],
    a_alignments: HashMap<Y, BTreeSet<X>>,
    b_alignments: HashMap<X, BTreeSet<Y>>,
    coverage: Vec<f32>,
}

impl<'a, T, U> Output<'a, T, U> {
    fn new(a: &'a [T], b: &'a [U], sat: SentenceAlignmentTable, coverage: Vec<f32>) -> Self {
        let mut a_alignments: HashMap<Y, BTreeSet<X>> = Default::default();
        let mut b_alignments: HashMap<X, BTreeSet<Y>> = Default::default();

        for Coordinates(x, y) in sat.anchors() {
            a_alignments.entry(y).or_default().insert(x);
            b_alignments.entry(x).or_default().insert(y);
        }

        Self {
            a,
            b,
            a_alignments,
            b_alignments,
            coverage,
        }
    }

    /// Returns an iterator of alignments for the sentence at index `i` of text `a`
    pub fn a_alignments(&self, i: usize) -> impl Iterator<Item = &U> {
        self.a_alignments
            .get(&Y(i))
            .into_iter()
            .flatten()
            .map(|X(j)| &self.b[*j])
        // .iter()
        // .enumerate()
        // .filter_map(|(j, &n)| (n >= self.anchor_threshold).then_some(&self.b[j]))
    }

    /// Returns an iterator of alignments for the sentence at index `i` of text `b`
    pub fn b_alignments(&self, i: usize) -> impl Iterator<Item = &T> {
        self.b_alignments
            .get(&X(i))
            .into_iter()
            .flatten()
            .map(|Y(j)| &self.a[*j])
    }

    /// Returns the coverage (aligned sentences / total sentences) obtained per cycle
    pub fn coverage(&self) -> &[f32] {
        &self.coverage
    }
}

type AssociationMapper<Word> = Box<dyn for<'a> Fn(&'a Word, &'a Word) -> bool>;

pub struct Config<Word> {
    /// Score required for an alignment to be considered an anchor and influence the AST. Defaults to `3`
    pub anchor_threshold: usize,
    /// Maximum number of cycles to perform before termination of the algorithm. Defaults to `20`
    pub max_cycles: usize,
    /// Word frequency required for items to be entered in the WAT. Defaults to `5`
    pub word_frequency_threshold: usize,
    /// The `word_frequency_threshold` will be decreased by this amount each cycle
    /// until the `word_frequency_minimum` is reached. Defaults to `0`
    pub word_frequency_taper: usize,
    /// Floor past which the `word_frequency_threshold` will no longer be decreased by
    /// `word_frequency_taper` on subsequent cycles. Defaults to `0`
    pub word_frequency_minimum: usize,
    /// Word similarity required for items to be entered in the WAT. Defaults to `0.8`
    pub word_similarity_threshold: f32,
    /// The `word_similarity_threshold` will be decreased by this amount each cycle
    /// until the `word_similarity_minimum` is reached. Defaults to `0.05`
    pub word_similarity_taper: f32,
    /// Floor past which the `word_similarity_threshold` will no longer be decreased by
    /// `word_similarity_taper` on subsequent cycles. Defaults to `0.3`
    pub word_similarity_minimum: f32,
    /// Minimum coverage to reach before the alignment is considered finished. The algorithm will
    /// continue processing until either this value or the `max_cycles` is reached. Defaults to `0.95`
    pub min_coverage: f32,
    /// Mapper which may be used to pre-populate the WAT. Associations indicated by the mapper will be
    /// given the highest priority (a similarity score of 1 and maximum frequency). Defaults to `|_, _| false`
    pub association_mapper: AssociationMapper<Word>,
}

impl<Word> Default for Config<Word> {
    fn default() -> Self {
        Self {
            anchor_threshold: 3,
            max_cycles: 20,
            word_frequency_threshold: 5,
            word_frequency_taper: 0,
            word_frequency_minimum: 0,
            word_similarity_threshold: 0.8,
            word_similarity_taper: 0.05,
            word_similarity_minimum: 0.3,
            min_coverage: 0.95,
            association_mapper: Box::new(|_, _| false),
        }
    }
}

impl<Word> Config<Word> {
    pub fn align<'a, T, U>(self, a: &'a [T], b: &'a [U]) -> Output<'a, T, U>
    where
        Word: Eq + PartialOrd + Hash + std::fmt::Debug + 'a,
        T: Sentence<Word>,
        U: Sentence<Word>
    {
        Parallelogram {
            anchor_threshold: Score::from(self.anchor_threshold),
            max_cycles: self.max_cycles,
            word_frequency_threshold: self.word_frequency_threshold,
            word_frequency_minimum: self.word_frequency_minimum,
            word_frequency_taper: self.word_frequency_taper,
            word_similarity_threshold: self.word_similarity_threshold,
            word_similarity_taper: self.word_similarity_taper,
            word_similarity_minimum: self.word_similarity_minimum,
            min_coverage: self.min_coverage,
            a_word_sentence_index: WordSentenceIndex::new(
                a.iter().map(|sentence| sentence.words()),
            ),
            b_word_sentence_index: WordSentenceIndex::new(
                b.iter().map(|sentence| sentence.words()),
            ),
            a,
            b,
            association_mapper: self.association_mapper,
        }
        .align()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct X(usize);

impl From<usize> for X {
    fn from(y: usize) -> Self {
        Self(y)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Y(usize);

impl From<usize> for Y {
    fn from(y: usize) -> Self {
        Self(y)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Coordinates(X, Y);

impl Coordinates {
    const ORIGIN: Self = Self(X(0), Y(0));

    fn x(&self) -> X {
        self.0
    }

    fn y(&self) -> Y {
        self.1
    }
}

struct Parallelogram<'a, Word, T, U> {
    anchor_threshold: Score,
    max_cycles: usize,
    word_frequency_threshold: usize,
    word_frequency_taper: usize,
    word_frequency_minimum: usize,
    word_similarity_threshold: f32,
    word_similarity_taper: f32,
    word_similarity_minimum: f32,
    min_coverage: f32,
    a: &'a [T],
    b: &'a [U],
    a_word_sentence_index: WordSentenceIndex<'a, Word, Y>,
    b_word_sentence_index: WordSentenceIndex<'a, Word, X>,
    association_mapper: AssociationMapper<Word>,
}

impl<'a, Word, T, U> Parallelogram<'a, Word, T, U>
where
    Word: Eq + PartialOrd + Hash + std::fmt::Debug,
    T: Sentence<Word>,
    U: Sentence<Word>
{
    fn align(&self) -> Output<'a, T, U> {
        let mut sat = SentenceAlignmentTable::new(self.a, self.b, self.anchor_threshold);
        let mut cycle_count = 0;
        let mut a_aligned = HashSet::new();
        let mut b_aligned = HashSet::new();
        let mut coverage = 0.0;
        let mut coverage_report = vec![];

        while coverage < self.min_coverage && cycle_count < self.max_cycles {
            let ast = AlignableSentenceTable::from(&sat);

            let wat = self.word_association_table(
                &ast,
                (self.word_similarity_threshold - cycle_count as f32 * self.word_similarity_taper)
                    .max(self.word_similarity_minimum),
                (self.word_frequency_threshold - cycle_count * self.word_frequency_taper)
                    .max(self.word_frequency_minimum),
            );

            for Reverse(association) in wat {
                for Coordinates(x, y) in association.align_sentences(&mut sat) {
                    a_aligned.insert(y);
                    b_aligned.insert(x);
                }
            }

            cycle_count += 1;
            coverage =
                (a_aligned.len() + b_aligned.len()) as f32 / (self.a.len() + self.b.len()) as f32;
            coverage_report.push(coverage);
        }

        Output::new(self.a, self.b, sat, coverage_report)
    }

    fn word_association_table(
        &'a self,
        ast: &'a AlignableSentenceTable,
        similarity_threshold: f32,
        frequency_threshold: usize,
    ) -> WordAssociationTable<'a, Word> {
        let mut visited = HashSet::new();
        let mut wat = BTreeSet::new();

        for Coordinates(x, y) in ast.all() {
            for a_word in self.a[y.0].words() {
                for b_word in self.b[x.0].words() {
                    if !visited.contains(&(a_word, b_word)) {
                        visited.insert((a_word, b_word));
                        let association = WordAssociation::new(
                            ast,
                            &self.a_word_sentence_index,
                            &self.b_word_sentence_index,
                            a_word,
                            b_word,
                            &self.association_mapper,
                        );

                        if association.similarity >= similarity_threshold
                            && association.a_occurrences >= frequency_threshold
                            && association.b_occurrences >= frequency_threshold
                        {
                            wat.insert(Reverse(association));
                        }
                    }
                }
            }
        }

        wat
    }
}
