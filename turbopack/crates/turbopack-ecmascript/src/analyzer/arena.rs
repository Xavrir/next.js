//! A bump-allocation arena for [`JsValue`](super::JsValue)s.
//!
//! A whole module analysis builds and links a large tree of `JsValue`s. This is a significant
//! amount of allocations / deallocations. Instead, we allocate all of the `JsValue`s into a single
//! [`Arena`] that lives for the duration of `analyze_ecmascript_module_internal` and is
//! freed in one shot when that function returns.

use std::{
    fmt,
    ops::{Deref, DerefMut},
};

use bumpalo::Bump;

/// Bump-allocated box used for the scalar children of a `JsValue` (e.g. the operand of `Not`).
pub type BumpBox<'a, T> = bumpalo::boxed::Box<'a, T>;

/// Bump-allocated growable vector used for the list children of a `JsValue` (e.g. `Array` items).
///
/// This is [`ArenaVec`], a thin wrapper over [`bumpalo::collections::Vec`] that adds `Send`/`Sync`.
/// The wrapper is required because the bare bumpalo `Vec` embeds an `&Bump`, which is
/// `!Send`/`!Sync`.
pub type BumpVec<'a, T> = ArenaVec<'a, T>;

#[derive(PartialEq, Eq, Hash)]
pub struct ArenaVec<'a, T>(bumpalo::collections::Vec<'a, T>);

/// We do not perform concurrent allocation; therefore, it is safe
/// to mark an `ArenaVec` as Send/Sync so that the analyzer's
/// async code continues to work.
unsafe impl<T: Send> Send for ArenaVec<'_, T> {}
unsafe impl<T: Sync> Sync for ArenaVec<'_, T> {}

impl<'a, T> Deref for ArenaVec<'a, T> {
    type Target = bumpalo::collections::Vec<'a, T>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for ArenaVec<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, T> IntoIterator for ArenaVec<'a, T> {
    type Item = T;
    type IntoIter = <bumpalo::collections::Vec<'a, T> as IntoIterator>::IntoIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'b, T> IntoIterator for &'b ArenaVec<'_, T> {
    type Item = &'b T;
    type IntoIter = std::slice::Iter<'b, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'b, T> IntoIterator for &'b mut ArenaVec<'_, T> {
    type Item = &'b mut T;
    type IntoIter = std::slice::IterMut<'b, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<T: fmt::Debug> fmt::Debug for ArenaVec<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

/// A bump-allocation arena scoped to a single module analysis.
pub struct Arena(Bump);
unsafe impl Sync for Arena {}

impl Arena {
    pub fn new() -> Self {
        Self(Bump::new())
    }

    /// The underlying bump allocator, for use with the `bumpalo::vec!` macro and `*_in`
    /// constructors.
    #[inline]
    pub fn bump(&self) -> &Bump {
        &self.0
    }

    /// Allocate `value` in the arena and return an owning [`BumpBox`].
    #[inline]
    pub fn alloc<T>(&self, value: T) -> BumpBox<'_, T> {
        BumpBox::new_in(value, &self.0)
    }

    /// Allocate `value` in the arena and return a shared reference to it.
    #[inline]
    pub fn alloc_ref<T>(&self, value: T) -> &T {
        self.0.alloc(value)
    }

    /// A new empty [`BumpVec`] backed by this arena.
    #[inline]
    pub fn vec<T>(&self) -> BumpVec<'_, T> {
        ArenaVec(bumpalo::collections::Vec::new_in(&self.0))
    }

    /// A [`BumpVec`] with capacity for `capacity` elements.
    #[inline]
    pub fn vec_with_capacity<T>(&self, capacity: usize) -> BumpVec<'_, T> {
        ArenaVec(bumpalo::collections::Vec::with_capacity_in(
            capacity, &self.0,
        ))
    }

    /// Collect `iter` into a [`BumpVec`] backed by this arena.
    #[inline]
    pub fn vec_from_iter<T>(&self, iter: impl IntoIterator<Item = T>) -> BumpVec<'_, T> {
        ArenaVec(bumpalo::collections::Vec::from_iter_in(iter, &self.0))
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}
