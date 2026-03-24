// This file is part of midnight-ledger.
// Copyright (C) 2025 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! State tree navigation extracted from the VM's `idx` instruction.
//!
//! Navigates a `StateValue` tree using `AlignedValue` keys. Each key is
//! interpreted based on the current node's type: array index, map key,
//! or merkle tree position.

use base_crypto::fab::{AlignedValue, InvalidBuiltinDecode, Value};
use crate::state::StateValue;
use storage::arena::Sp;
use storage::db::DB;

/// Navigate one level into a `StateValue` using the given key.
///
/// The key is interpreted based on the current variant:
/// - **Array**: key is converted to `u8` index
/// - **Map**: key is used directly for `HashMap::get`
/// - **BoundedMerkleTree**: key is converted to `u64` position
///
/// Returns `Ok(None)` for map keys that don't exist (normal lookup miss).
/// Returns `Err` for structural errors: type conversion failures,
/// out-of-bounds array indices, invalid merkle tree positions, or
/// unsupported variants (Cell, Null).
pub fn idx<D: DB>(
    sv: &StateValue<D>,
    key: &AlignedValue,
) -> Result<Option<StateValue<D>>, IdxError> {
    match sv {
        StateValue::Array(arr) => {
            let index: u8 = (&**AsRef::<Value>::as_ref(key))
                .try_into()
                .map_err(IdxError::Decode)?;
            arr.get(index as usize)
                .cloned()
                .map(Some)
                .ok_or(IdxError::IndexOutOfBounds(index))
        }
        StateValue::Map(map) => Ok(map.get(key).map(|sp| (*sp).clone())),
        StateValue::BoundedMerkleTree(tree) => {
            let pos: u64 = (&**AsRef::<Value>::as_ref(key))
                .try_into()
                .map_err(IdxError::Decode)?;
            if pos >= (1u64 << tree.height() as u64) {
                return Err(IdxError::MissingKey);
            }
            Ok(tree.index(pos).map(|(hash, ())| {
                StateValue::Cell(Sp::new(hash.into()))
            }))
        }
        _ => Err(IdxError::UnsupportedVariant),
    }
}

/// Navigate a full path of keys through a `StateValue` tree.
///
/// Each key in the path is applied via [`idx`]. Stops early if a key is not
/// found (returns `Ok(None)`).
pub fn idx_path<D: DB>(
    sv: &StateValue<D>,
    keys: &[AlignedValue],
) -> Result<Option<StateValue<D>>, IdxError> {
    let mut current = sv.clone();
    for key in keys {
        match idx(&current, key)? {
            Some(next) => current = next,
            None => return Ok(None),
        }
    }
    Ok(Some(current))
}

/// Error from [`idx`] / [`idx_path`] navigation.
#[derive(Clone, Debug, PartialEq)]
pub enum IdxError {
    /// Array index is valid but exceeds the array length.
    IndexOutOfBounds(u8),
    /// The key could not be converted to the expected type for the variant.
    Decode(InvalidBuiltinDecode),
    /// MerkleTree position is outside the tree's range.
    MissingKey,
    /// The `StateValue` variant does not support indexing (Cell, Null).
    UnsupportedVariant,
}

impl std::fmt::Display for IdxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IdxError::IndexOutOfBounds(i) => write!(f, "index {i} out of bounds"),
            IdxError::Decode(e) => write!(f, "{e}"),
            IdxError::MissingKey => write!(f, "key not found"),
            IdxError::UnsupportedVariant => {
                write!(f, "unsupported variant: only array, map, and merkle tree can be indexed")
            }
        }
    }
}

impl std::error::Error for IdxError {}

#[cfg(test)]
mod tests {
    use super::*;
    use storage::db::InMemoryDB;
    use storage::storage::{Array, HashMap};

    fn make_cell<D: DB>(val: u64) -> StateValue<D> {
        StateValue::from(val)
    }

    // -- idx tests --

    #[test]
    fn idx_array() {
        let arr: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![make_cell(42), make_cell(100)]));
        let key: AlignedValue = 1u8.into();
        let result = idx(&arr, &key).unwrap().unwrap();
        assert!(matches!(result, StateValue::Cell(_)));
    }

    #[test]
    fn idx_array_out_of_bounds() {
        let arr: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![make_cell(1)]));
        let key: AlignedValue = 5u8.into();
        assert_eq!(idx(&arr, &key), Err(IdxError::IndexOutOfBounds(5)));
    }

    #[test]
    fn idx_array_invalid_key() {
        let arr: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![make_cell(1)]));
        // Multi-byte value that can't convert to u8
        let key: AlignedValue = 999u64.into();
        assert!(matches!(idx(&arr, &key), Err(IdxError::Decode(_))));
    }

    #[test]
    fn idx_map_found() {
        let map_key: AlignedValue = 42u64.into();
        let mut map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        map = map.insert(map_key.clone(), make_cell(999));
        let sv = StateValue::Map(map);
        assert!(idx(&sv, &map_key).unwrap().is_some());
    }

    #[test]
    fn idx_map_not_found() {
        let map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        let sv = StateValue::Map(map);
        let key: AlignedValue = 1u64.into();
        assert!(idx(&sv, &key).unwrap().is_none());
    }

    #[test]
    fn idx_unsupported_cell() {
        let cell: StateValue<InMemoryDB> = make_cell(1);
        let key: AlignedValue = 0u8.into();
        assert_eq!(idx(&cell, &key), Err(IdxError::UnsupportedVariant));
    }

    #[test]
    fn idx_unsupported_null() {
        let null: StateValue<InMemoryDB> = StateValue::Null;
        let key: AlignedValue = 0u8.into();
        assert_eq!(idx(&null, &key), Err(IdxError::UnsupportedVariant));
    }

    // -- idx_path tests --

    #[test]
    fn idx_path_empty() {
        let root: StateValue<InMemoryDB> = make_cell(42);
        let result = idx_path(&root, &[]).unwrap().unwrap();
        assert!(matches!(result, StateValue::Cell(_)));
    }

    #[test]
    fn idx_path_through_array_and_map() {
        let map_key: AlignedValue = 7u64.into();
        let mut map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        map = map.insert(map_key.clone(), make_cell(42));
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![StateValue::Map(map)]));

        let index_key: AlignedValue = 0u8.into();
        let result = idx_path(&root, &[index_key, map_key]).unwrap().unwrap();
        assert!(matches!(result, StateValue::Cell(_)));
    }

    #[test]
    fn idx_path_stops_on_not_found() {
        let map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![StateValue::Map(map)]));

        let index_key: AlignedValue = 0u8.into();
        let missing_key: AlignedValue = 999u64.into();
        assert!(idx_path(&root, &[index_key, missing_key]).unwrap().is_none());
    }
}
