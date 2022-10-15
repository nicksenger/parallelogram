use std::cmp::Reverse;
use std::collections::BTreeSet;

use super::WordAssociation;

pub type WordAssociationTable<'a, Word> = BTreeSet<Reverse<WordAssociation<'a, Word>>>;
