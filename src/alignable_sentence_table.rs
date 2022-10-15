use std::collections::{HashMap, HashSet};

use super::sentence_alignment_table::SentenceAlignmentTable;
use super::{Coordinates, X, Y};

#[derive(Default, Debug)]
pub struct AlignableSentenceTable(HashMap<X, HashSet<Y>>);

impl AlignableSentenceTable {
    pub(crate) fn contains(&self, Coordinates(x, y): Coordinates) -> bool {
        self.0.get(&x).map(|set| set.contains(&y)).unwrap_or(false)
    }

    pub(crate) fn insert(&mut self, Coordinates(x, y): Coordinates) {
        self.0.entry(x).or_default().insert(y);
    }

    pub(crate) fn all(&self) -> impl Iterator<Item = Coordinates> + '_ {
        self.0
            .iter()
            .flat_map(|(x, ys)| ys.iter().map(|y| Coordinates(*x, *y)))
    }
}

impl From<&SentenceAlignmentTable> for AlignableSentenceTable {
    fn from(sat: &SentenceAlignmentTable) -> Self {
        let mut ast = Self::default();

        let mut start = sat.next_anchor(None);
        let mut end = sat.next_anchor(Some(start));

        while start != end {
            let x_distance = (end.x().0 - start.x().0) as f32;
            let y_distance = (end.y().0 - start.y().0) as f32;

            if x_distance > y_distance {
                for y in start.y().0..=end.y().0 {
                    let progress = (y - start.y().0) as f32 / y_distance;
                    let scale = (0.5 - progress).abs() / 0.5;
                    let n = (x_distance.sqrt() - scale * x_distance.sqrt())
                        .min(x_distance.sqrt())
                        .max(1.0) as usize;

                    let diagonal = start.x().0 as f32 + (progress * x_distance);
                    let min = (diagonal - n as f32 / 2.0).floor().max(start.x().0 as f32) as usize;
                    let max = (diagonal + n as f32 / 2.0).floor().min(end.x().0 as f32) as usize;
                    for x in min..=max {
                        ast.insert(Coordinates(X(x), Y(y)))
                    }
                }
            } else {
                for x in start.x().0..=end.x().0 {
                    let progress = (x - start.x().0) as f32 / x_distance;
                    let scale = (0.5 - progress).abs() / 0.5;
                    let n = (y_distance.sqrt() - scale * y_distance.sqrt())
                        .min(y_distance.sqrt())
                        .max(1.0) as usize;

                    let diagonal = start.y().0 as f32 + (progress * y_distance);
                    let min = (diagonal - n as f32 / 2.0).floor().max(start.y().0 as f32) as usize;
                    let max = (diagonal + n as f32 / 2.0).floor().min(end.y().0 as f32) as usize;
                    for y in min..=max {
                        ast.insert(Coordinates(X(x), Y(y)))
                    }
                }
            }

            start = end;
            end = sat.next_anchor(Some(start));
        }

        ast
    }
}
