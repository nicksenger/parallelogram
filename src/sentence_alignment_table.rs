use std::collections::BTreeMap;
use std::ops::{AddAssign, Bound};

use super::{Coordinates, X, Y};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Score(usize);

impl AddAssign for Score {
    fn add_assign(&mut self, other: Self) {
        *self = Self(self.0 + other.0);
    }
}

impl From<usize> for Score {
    fn from(u: usize) -> Self {
        Self(u)
    }
}

impl Score {
    pub const ZERO: Self = Self(0);
}

#[derive(Debug)]
pub struct SentenceAlignmentTable {
    anchor_threshold: Score,
    map: BTreeMap<X, BTreeMap<Y, Score>>,
    end: Coordinates,
}

impl SentenceAlignmentTable {
    pub(crate) fn new<T, U>(a: &[T], b: &[U], anchor_threshold: Score) -> Self {
        Self {
            anchor_threshold,
            map: Default::default(),
            end: Coordinates(X(b.len() - 1), Y(a.len() - 1)),
        }
    }

    pub(crate) fn score(&self, Coordinates(x, y): Coordinates) -> Score {
        self.map
            .get(&x)
            .and_then(|ys| ys.get(&y))
            .copied()
            .unwrap_or(Score(0))
    }

    pub(crate) fn next_anchor(&self, start: Option<Coordinates>) -> Coordinates {
        if let Some(Coordinates(x, y)) = start {
            self.map
                .range((Bound::Excluded(x), Bound::Included(self.end.x())))
                .find_map(|(&x, ys)| {
                    ys.range((Bound::Excluded(y), Bound::Included(self.end.y())))
                        .find_map(|(&y, &score)| {
                            (score >= self.anchor_threshold).then_some(Coordinates(x, y))
                        })
                })
                .unwrap_or(self.end)
        } else {
            Coordinates::ORIGIN
        }
    }

    pub(crate) fn crossover(&self, Coordinates(x, y): Coordinates) -> bool {
        self.map
            .range((Bound::Excluded(x), Bound::Included(self.end.x())))
            .find_map(|(&x, ys)| {
                ys.range((Bound::Included(Y(0)), Bound::Excluded(y)))
                    .map(|(&y, _score)| Coordinates(x, y))
                    .next()
            })
            .is_some()
            || self
                .map
                .range((Bound::Included(X(0)), Bound::Excluded(x)))
                .find_map(|(&x, ys)| {
                    ys.range((Bound::Excluded(y), Bound::Included(self.end.y())))
                        .map(|(&y, _score)| Coordinates(x, y))
                        .next()
                })
                .is_some()
    }

    pub(crate) fn increment(&mut self, Coordinates(x, y): Coordinates) {
        *self.map.entry(x).or_default().entry(y).or_default() += Score(1);
    }

    pub(crate) fn anchors(&self) -> impl Iterator<Item = Coordinates> + '_ {
        self.map.iter().flat_map(move |(&x, ys)| {
            ys.iter().filter_map(move |(&y, &score)| {
                (score >= self.anchor_threshold).then_some(Coordinates(x, y))
            })
        })
    }
}
