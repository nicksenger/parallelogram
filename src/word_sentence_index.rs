use std::collections::HashMap;
use std::hash::Hash;

pub struct WordSentenceIndex<'a, Word, Axis>(HashMap<&'a Word, Vec<Axis>>);

impl<'a, Word, Axis: From<usize>> WordSentenceIndex<'a, Word, Axis>
where
    Word: Eq + Hash,
    Axis: Clone + Copy + From<usize>,
{
    pub fn new(text: impl Iterator<Item = &'a [Word]>) -> Self {
        let mut map: HashMap<&Word, Vec<Axis>> = HashMap::new();

        for (i, sentence) in text.enumerate() {
            for word in sentence {
                map.entry(word).or_default().push(Axis::from(i));
            }
        }

        Self(map)
    }

    pub fn sentences(&self, word: &Word) -> impl Iterator<Item = Axis> + '_ {
        self.0.get(word).into_iter().flatten().copied()
    }

    pub fn occurrences(&self, word: &Word) -> usize {
        self.0.get(word).map(|v| v.len()).unwrap_or(0)
    }
}
