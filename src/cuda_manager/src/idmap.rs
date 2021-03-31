use std::collections::{hash_map::Iter, HashMap};
#[derive(Debug)]
pub(crate) struct IdMap<T> {
    inner: HashMap<usize, T>,
    cursor: usize,
}

impl<T> IdMap<T> {
    pub fn new() -> Self {
        IdMap {
            inner: HashMap::new(),
            cursor: 0,
        }
    }

    pub fn insert(&mut self, item: T) -> usize {
        self.inner.insert(self.cursor, item);
        let ret = self.cursor;
        self.cursor += 1;
        ret
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        self.inner.remove(&index)
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(&index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.inner.get_mut(&index)
    }

    pub fn iter(&self) -> Iter<'_, usize, T> {
        self.inner.iter()
    }
}

impl<'a, T> IntoIterator for &'a IdMap<T> {
    type Item = (&'a usize, &'a T);
    type IntoIter = Iter<'a, usize, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}
