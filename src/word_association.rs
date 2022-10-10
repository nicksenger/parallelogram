use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

pub struct WordAssociation<'a, Word> {
    pub a: &'a Word,
    pub b: &'a Word,
    pub similarity: f32,
    pub a_occurrences: usize,
    pub b_occurrences: usize,
    alignable_sentence_table: &'a Vec<Vec<usize>>,
    a_word_sentence_index: &'a HashMap<&'a Word, Vec<usize>>,
    b_word_sentence_index: &'a HashMap<&'a Word, Vec<usize>>,
}

impl<'a, Word> Clone for WordAssociation<'a, Word> {
    fn clone(&self) -> Self {
        Self {
            a: self.a,
            b: self.b,
            similarity: self.similarity,
            a_occurrences: self.a_occurrences,
            b_occurrences: self.b_occurrences,
            alignable_sentence_table: self.alignable_sentence_table,
            a_word_sentence_index: self.a_word_sentence_index,
            b_word_sentence_index: self.b_word_sentence_index,
        }
    }
}

impl<'a, Word> Copy for WordAssociation<'a, Word> {}

impl<'a, Word> Debug for WordAssociation<'a, Word>
where
    Word: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WordAssociation")
            .field("a", &self.a)
            .field("b", &self.b)
            .field("similarity", &self.similarity)
            .field("b_freq", &self.b_occurrences)
            .finish()
    }
}

impl<'a, Word: Eq + Hash + Debug> WordAssociation<'a, Word> {
    pub fn new(
        alignable_sentence_table: &'a Vec<Vec<usize>>,
        a_word_sentence_index: &'a HashMap<&Word, Vec<usize>>,
        b_word_sentence_index: &'a HashMap<&Word, Vec<usize>>,
        a: &'a Word,
        b: &'a Word,
        association_mapper: impl for<'b> Fn(&'b Word, &'b Word) -> bool,
    ) -> Self {
        let mapped_association = association_mapper(a, b);
        Self {
            a,
            b,
            similarity: if mapped_association {
                1.0
            } else {
                Self::similarity(
                    alignable_sentence_table,
                    a_word_sentence_index,
                    b_word_sentence_index,
                    a,
                    b,
                )
            },
            a_occurrences: if mapped_association {
                usize::MAX
            } else {
                a_word_sentence_index[&a].len()
            },
            b_occurrences: if mapped_association {
                usize::MAX
            } else {
                b_word_sentence_index[&b].len()
            },
            alignable_sentence_table,
            a_word_sentence_index,
            b_word_sentence_index,
        }
    }

    pub fn align_sentences(
        &self,
        sentence_alignment_table: &mut Vec<Vec<usize>>,
    ) -> Vec<[usize; 2]> {
        let mut a_candidates = HashMap::new();
        let mut b_candidates = HashMap::new();
        for &y in &self.a_word_sentence_index[self.a] {
            for &x in &self.b_word_sentence_index[self.b] {
                if self.alignable_sentence_table[y][x] == 1 {
                    a_candidates.entry(y).or_insert_with(HashSet::new).insert(x);
                    b_candidates.entry(x).or_insert_with(HashSet::new).insert(y);
                }
            }
        }

        let matches = b_candidates
            .into_iter()
            .filter(|(x, ys)| {
                ys.len() == 1
                    && a_candidates[ys.iter().next().unwrap()].len() == 1
                    && a_candidates[ys.iter().next().unwrap()]
                        .iter()
                        .next()
                        .unwrap()
                        == x
            })
            .map(|(x, ys)| [x, *ys.iter().next().unwrap()])
            .collect::<Vec<_>>();

        for [x, y] in &matches {
            if sentence_alignment_table[*y][*x] == 0 {
                for y in (y + 1)..sentence_alignment_table.len() {
                    for x in 0..*x {
                        if sentence_alignment_table[y][x] >= 1 {
                            return vec![];
                        }
                    }
                }

                for y in 0..*y {
                    for x in (x + 1)..sentence_alignment_table[0].len() {
                        if sentence_alignment_table[y][x] >= 1 {
                            return vec![];
                        }
                    }
                }
            }
        }

        for [x, y] in &matches {
            sentence_alignment_table[*y][*x] += 1;
        }

        matches
    }

    fn similarity(
        alignable_sentence_table: &Vec<Vec<usize>>,
        a_word_sentence_index: &HashMap<&Word, Vec<usize>>,
        b_word_sentence_index: &HashMap<&Word, Vec<usize>>,
        a: &Word,
        b: &Word,
    ) -> f32 {
        let a_candidates = a_word_sentence_index[&a]
            .iter()
            .map(|&sentence| Candidate {
                sentence,
                alignable_sentence_table,
                y_axis: true,
            })
            .collect::<Vec<_>>();
        let b_candidates = b_word_sentence_index[&b]
            .iter()
            .map(|&sentence| Candidate {
                sentence,
                alignable_sentence_table,
                y_axis: false,
            })
            .collect::<Vec<_>>();

        let output = hirschberg::Config {
            match_score: 1,
            mismatch_score: 0,
            gap_score: 0,
        }
        .compute(&a_candidates, &b_candidates);

        let c = output.score();
        let a_occurrences = a_word_sentence_index[&a].len();
        let b_occurrences = b_word_sentence_index[&b].len();

        (2 * c) as f32 / (a_occurrences + b_occurrences) as f32
    }
}

impl<'a, Word: PartialEq> Eq for WordAssociation<'a, Word> {}

impl<'a, Word: PartialEq> PartialEq for WordAssociation<'a, Word> {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b
    }
}

impl<'a, Word: PartialEq + PartialOrd> PartialOrd for WordAssociation<'a, Word> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else {
            match self.similarity.partial_cmp(&other.similarity) {
                Some(Ordering::Equal) => match (self.a_occurrences + self.b_occurrences)
                    .partial_cmp(&(other.a_occurrences + other.b_occurrences))
                {
                    Some(Ordering::Equal) => self.a.partial_cmp(other.a),
                    ordering => ordering,
                },
                ordering => ordering,
            }
        }
    }
}

impl<'a, Word: PartialEq + PartialOrd> Ord for WordAssociation<'a, Word> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

struct Candidate<'a> {
    sentence: usize,
    alignable_sentence_table: &'a Vec<Vec<usize>>,
    y_axis: bool,
}

impl<'a> PartialEq for Candidate<'a> {
    fn eq(&self, other: &Self) -> bool {
        if self.y_axis {
            self.alignable_sentence_table[self.sentence][other.sentence] != 0
        } else {
            self.alignable_sentence_table[other.sentence][self.sentence] != 0
        }
    }
}
