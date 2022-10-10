use std::cmp::Reverse;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;

mod word_association;

use word_association::WordAssociation;

pub trait Sentence<Word: PartialEq> {
    fn words(&self) -> &[Word];
}

pub struct Output<'a, T> {
    anchor_threshold: usize,
    a: &'a [T],
    b: &'a [T],
    sentence_alignment_table: Vec<Vec<usize>>,
    coverage: Vec<f32>,
}

impl<'a, T> Output<'a, T> {
    /// Returns an iterator of alignments for the sentence at index `i` of text `a`
    pub fn a_alignments(&self, i: usize) -> impl Iterator<Item = &T> {
        self.sentence_alignment_table[i]
            .iter()
            .enumerate()
            .filter_map(|(j, &n)| (n >= self.anchor_threshold).then_some(&self.b[j]))
    }

    /// Returns an iterator of alignments for the sentence at index `i` of text `b`
    pub fn b_alignments(&self, i: usize) -> impl Iterator<Item = &T> {
        self.sentence_alignment_table
            .iter()
            .enumerate()
            .filter_map(move |(j, row)| (row[i] >= self.anchor_threshold).then_some(&self.a[j]))
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
    pub fn align<'a, T>(self, a: &'a [T], b: &'a [T]) -> Output<'a, T>
    where
        Word: Eq + PartialOrd + Hash + std::fmt::Debug + 'a,
        T: Sentence<Word>,
    {
        Parallelogram {
            anchor_threshold: self.anchor_threshold,
            max_cycles: self.max_cycles,
            word_frequency_threshold: self.word_frequency_threshold,
            word_frequency_minimum: self.word_frequency_minimum,
            word_frequency_taper: self.word_frequency_taper,
            word_similarity_threshold: self.word_similarity_threshold,
            word_similarity_taper: self.word_similarity_taper,
            word_similarity_minimum: self.word_similarity_minimum,
            min_coverage: self.min_coverage,
            a_word_sentence_index: Parallelogram::word_sentence_index(a),
            b_word_sentence_index: Parallelogram::word_sentence_index(b),
            a,
            b,
            association_mapper: self.association_mapper,
        }
        .align()
    }
}

struct Parallelogram<'a, Word, T> {
    anchor_threshold: usize,
    max_cycles: usize,
    word_frequency_threshold: usize,
    word_frequency_taper: usize,
    word_frequency_minimum: usize,
    word_similarity_threshold: f32,
    word_similarity_taper: f32,
    word_similarity_minimum: f32,
    min_coverage: f32,
    a: &'a [T], // Y-axis
    b: &'a [T],
    a_word_sentence_index: HashMap<&'a Word, Vec<usize>>,
    b_word_sentence_index: HashMap<&'a Word, Vec<usize>>,
    association_mapper: AssociationMapper<Word>,
}

impl<'a, Word, T> Parallelogram<'a, Word, T>
where
    Word: Eq + PartialOrd + Hash + std::fmt::Debug,
    T: Sentence<Word>,
{
    fn align(&self) -> Output<'a, T> {
        let mut sentence_alignment_table = vec![vec![0; self.b.len()]; self.a.len()];
        let mut cycle_count = 0;
        let mut a_aligned = HashSet::new();
        let mut b_aligned = HashSet::new();
        let mut coverage = 0.0;
        let mut coverage_report = vec![];

        while coverage < self.min_coverage && cycle_count < self.max_cycles {
            let alignable_sentence_table =
                alignable_sentence_table(&sentence_alignment_table, self.anchor_threshold);

            let word_association_table = self.word_association_tables(
                &alignable_sentence_table,
                (self.word_similarity_threshold - cycle_count as f32 * self.word_similarity_taper)
                    .max(self.word_similarity_minimum),
                (self.word_frequency_threshold - cycle_count * self.word_frequency_taper)
                    .max(self.word_frequency_minimum),
            );

            for Reverse(association) in word_association_table {
                for [x, y] in association.align_sentences(&mut sentence_alignment_table) {
                    a_aligned.insert(y);
                    b_aligned.insert(x);
                }
            }

            cycle_count += 1;
            coverage =
                (a_aligned.len() + b_aligned.len()) as f32 / (self.a.len() + self.b.len()) as f32;
            coverage_report.push(coverage);
        }

        Output {
            anchor_threshold: self.anchor_threshold,
            a: self.a,
            b: self.b,
            sentence_alignment_table,
            coverage: coverage_report,
        }
    }

    fn word_association_tables(
        &self,
        alignable_sentence_table: &'a Vec<Vec<usize>>,
        similarity_threshold: f32,
        frequency_threshold: usize,
    ) -> BTreeSet<Reverse<WordAssociation<Word>>> {
        let mut visited = HashSet::new();
        let mut word_association_table = BTreeSet::new();

        for y in 0..alignable_sentence_table.len() {
            for x in 0..alignable_sentence_table[0].len() {
                if alignable_sentence_table[y][x] == 1 {
                    for a_word in self.a[y].words() {
                        for b_word in self.b[x].words() {
                            if !visited.contains(&(a_word, b_word)) {
                                visited.insert((a_word, b_word));
                                let association = WordAssociation::new(
                                    alignable_sentence_table,
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
                                    word_association_table.insert(Reverse(association));
                                }
                            }
                        }
                    }
                }
            }
        }

        word_association_table
    }

    fn word_sentence_index(sentences: &[T]) -> HashMap<&Word, Vec<usize>> {
        let mut map = HashMap::new();

        for (i, sentence) in sentences.iter().enumerate() {
            for word in sentence.words() {
                map.entry(word).or_insert(Vec::new()).push(i);
            }
        }

        map
    }
}

fn alignable_sentence_table(
    sentence_alignment_table: &Vec<Vec<usize>>,
    anchor_threshold: usize,
) -> Vec<Vec<usize>> {
    let next_anchor =
        |sentence_alignment_table: &Vec<Vec<usize>>, from: Option<[usize; 2]>| -> [usize; 2] {
            if let Some([from_x, from_y]) = from {
                sentence_alignment_table
                    .iter()
                    .enumerate()
                    .skip_while(|(y, _)| *y <= from_y)
                    .find_map(|(y, row)| {
                        row.iter()
                            .enumerate()
                            .skip_while(|(x, _)| y == from_y && *x <= from_x)
                            .find_map(|(x, score)| (*score >= anchor_threshold).then_some([x, y]))
                    })
                    .unwrap_or([
                        sentence_alignment_table[0].len() - 1,
                        sentence_alignment_table.len() - 1,
                    ])
            } else {
                [0, 0]
            }
        };

    let mut alignable_sentence_table =
        vec![vec![0; sentence_alignment_table[0].len()]; sentence_alignment_table.len()];

    let [mut start_x, mut start_y] = next_anchor(sentence_alignment_table, None);
    let [mut end_x, mut end_y] = next_anchor(sentence_alignment_table, Some([start_x, start_y]));

    while [start_x, start_y] != [end_x, end_y] {
        let x_distance = (end_x - start_x) as f32;
        let y_distance = (end_y - start_y) as f32;

        if x_distance > y_distance {
            for y in start_y..=end_y {
                let progress = (y - start_y) as f32 / y_distance;
                let scale = (0.5 - progress).abs() / 0.5;
                let n = (x_distance.sqrt() - scale * x_distance.sqrt())
                    .min(x_distance.sqrt())
                    .max(1.0) as usize;

                let diagonal = start_x as f32 + (progress * x_distance);
                let min = (diagonal - n as f32 / 2.0).floor().max(start_x as f32) as usize;
                let max = (diagonal + n as f32 / 2.0).floor().min(end_x as f32) as usize;
                for x in min..=max {
                    alignable_sentence_table[y][x] = 1;
                }
            }
        } else {
            for x in start_x..=end_x {
                let progress = (x - start_x) as f32 / x_distance;
                let scale = (0.5 - progress).abs() / 0.5;
                let n = (y_distance.sqrt() - scale * y_distance.sqrt())
                    .min(y_distance.sqrt())
                    .max(1.0) as usize;

                let diagonal = start_y as f32 + (progress * y_distance);
                let min = (diagonal - n as f32 / 2.0).floor().max(start_y as f32) as usize;
                let max = (diagonal + n as f32 / 2.0).floor().min(end_y as f32) as usize;
                for y in min..=max {
                    alignable_sentence_table[y][x] = 1;
                }
            }
        }

        [start_x, start_y] = [end_x, end_y];
        [end_x, end_y] = next_anchor(sentence_alignment_table, Some([start_x, start_y]));
    }

    alignable_sentence_table
}
