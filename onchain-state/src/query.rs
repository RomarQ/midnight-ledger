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

//! Targeted queries into a contract's state tree.
//!
//! Navigates the `StateValue` tree following a path of `AlignedValue` keys,
//! mirroring the VM's `idx` instruction. Each key is interpreted based on the
//! current node's type: array index, map key, or merkle tree position.

use base_crypto::fab::{AlignedValue, Value};
use crate::state::StateValue;
use serialize::Deserializable;
use storage::arena::Sp;
use storage::db::DB;

// ---------------------------------------------------------------------------
// Navigation — follows the same model as the VM's `idx` instruction
// ---------------------------------------------------------------------------

/// Navigate one level into a `StateValue` using the given key.
///
/// The key is interpreted based on the current variant:
/// - **Array**: key is converted to `u8` index
/// - **Map**: key is used directly for `HashMap::get`
/// - **BoundedMerkleTree**: key is converted to `u64` position
///
/// Returns `None` for map keys / tree positions that don't exist.
/// Returns `Err` for type mismatches or out-of-bounds indices.
pub fn idx<D: DB>(
    sv: &StateValue<D>,
    key: &AlignedValue,
) -> Result<Option<StateValue<D>>, IdxError> {
    match sv {
        StateValue::Array(arr) => {
            let index: u8 = (&**AsRef::<Value>::as_ref(key))
                .try_into()
                .map_err(|_| IdxError::InvalidKey("cannot convert key to array index".into()))?;
            arr.get(index as usize)
                .cloned()
                .map(Some)
                .ok_or(IdxError::IndexOutOfBounds(index))
        }
        StateValue::Map(map) => Ok(map.get(key).map(|sp| (*sp).clone())),
        StateValue::BoundedMerkleTree(tree) => {
            let pos: u64 = (&**AsRef::<Value>::as_ref(key))
                .try_into()
                .map_err(|_| IdxError::InvalidKey("cannot convert key to tree position".into()))?;
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
#[derive(Clone, Debug)]
pub enum IdxError {
    IndexOutOfBounds(u8),
    InvalidKey(String),
    MissingKey,
    UnsupportedVariant,
}

impl std::fmt::Display for IdxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IdxError::IndexOutOfBounds(i) => write!(f, "index {i} out of bounds"),
            IdxError::InvalidKey(msg) => write!(f, "invalid key: {msg}"),
            IdxError::MissingKey => write!(f, "key not found"),
            IdxError::UnsupportedVariant => {
                write!(f, "unsupported variant: only array, map, and merkle tree can be indexed")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Batch query
// ---------------------------------------------------------------------------

/// A query: a path of serialized `AlignedValue` keys through the state tree.
#[derive(Clone, Debug)]
pub struct StateQuery {
    /// Each element is a serialized `AlignedValue`. Interpreted as array index,
    /// map key, or merkle tree position depending on the node at each level.
    pub path: Vec<Vec<u8>>,
}

/// Result of a single state query.
#[derive(Clone, Debug)]
pub struct StateQueryResult {
    pub query: StateQuery,
    /// Serialized value (tagged format). `None` if not found.
    pub value: Option<Vec<u8>>,
    /// Error message. `None` if successful.
    pub error: Option<String>,
}

/// Resolve a list of queries against a contract's root `StateValue`.
///
/// Each query's path is deserialized into `AlignedValue` keys and navigated
/// via [`idx_path`]. The final value is serialized with `tagged_serialize`.
/// Collection values (Map, Set, MerkleTree) at the end of the path are not
/// serialized (would be O(n)); only leaf values (Cell, Null) are returned.
pub fn query_state<D: DB>(root: &StateValue<D>, queries: &[StateQuery]) -> Vec<StateQueryResult> {
    queries
        .iter()
        .map(|query| {
            let ok = |value| StateQueryResult {
                query: query.clone(),
                value,
                error: None,
            };
            let err = |msg: String| StateQueryResult {
                query: query.clone(),
                value: None,
                error: Some(msg),
            };

            // Deserialize path keys
            let keys: Vec<AlignedValue> = match query
                .path
                .iter()
                .map(|bytes| {
                    let mut reader: &[u8] = bytes.as_slice();
                    <AlignedValue as Deserializable>::deserialize(&mut reader, 0)
                        .map_err(|e| format!("bad key: {e}"))
                })
                .collect::<Result<Vec<_>, _>>()
            {
                Ok(keys) => keys,
                Err(e) => return err(e),
            };

            match idx_path(root, &keys) {
                Ok(Some(sv)) => match sv {
                    StateValue::Map(_) | StateValue::BoundedMerkleTree(_) => {
                        err("path resolves to a collection; provide a deeper path".into())
                    }
                    val => serialize_sv(&val).map_or_else(err, |b| ok(Some(b))),
                },
                Ok(None) => ok(None),
                Err(e) => err(e.to_string()),
            }
        })
        .collect()
}

fn serialize_sv<D: DB>(sv: &StateValue<D>) -> Result<Vec<u8>, String> {
    let size = serialize::tagged_serialized_size(sv);
    let mut buf = Vec::with_capacity(size);
    serialize::tagged_serialize(sv, &mut buf).map_err(|e| format!("serialize: {e}"))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serialize::Serializable;
    use storage::db::InMemoryDB;
    use storage::storage::{Array, HashMap};

    fn make_cell<D: DB>(val: u64) -> StateValue<D> {
        StateValue::from(val)
    }

    fn serialize_key(val: impl Into<AlignedValue>) -> Vec<u8> {
        let av: AlignedValue = val.into();
        let mut bytes = Vec::new();
        av.serialize(&mut bytes).unwrap();
        bytes
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
        assert!(matches!(idx(&arr, &key), Err(IdxError::IndexOutOfBounds(5))));
    }

    #[test]
    fn idx_map_found() {
        let map_key: AlignedValue = 42u64.into();
        let mut map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        map = map.insert(map_key.clone(), make_cell(999));
        let sv = StateValue::Map(map);
        let result = idx(&sv, &map_key).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn idx_map_not_found() {
        let map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        let sv = StateValue::Map(map);
        let key: AlignedValue = 1u64.into();
        assert!(idx(&sv, &key).unwrap().is_none());
    }

    #[test]
    fn idx_unsupported_variant() {
        let cell: StateValue<InMemoryDB> = make_cell(1);
        let key: AlignedValue = 0u8.into();
        assert!(matches!(idx(&cell, &key), Err(IdxError::UnsupportedVariant)));
    }

    // -- idx_path tests --

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

    // -- query_state tests --

    #[test]
    fn query_cell() {
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![make_cell(42)]));
        let results = query_state(&root, &[StateQuery {
            path: vec![serialize_key(0u8)],
        }]);
        assert!(results[0].value.is_some());
        assert!(results[0].error.is_none());
    }

    #[test]
    fn query_map_entry() {
        let map_key: AlignedValue = 42u64.into();
        let mut map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        map = map.insert(map_key, make_cell(999));
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![StateValue::Map(map)]));
        let results = query_state(&root, &[StateQuery {
            path: vec![serialize_key(0u8), serialize_key(42u64)],
        }]);
        assert!(results[0].value.is_some());
    }

    #[test]
    fn query_map_not_found() {
        let map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![StateValue::Map(map)]));
        let results = query_state(&root, &[StateQuery {
            path: vec![serialize_key(0u8), serialize_key(999u64)],
        }]);
        assert!(results[0].value.is_none());
        assert!(results[0].error.is_none());
    }

    #[test]
    fn query_stops_at_collection() {
        let map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![StateValue::Map(map)]));
        // Path stops at the Map without going deeper
        let results = query_state(&root, &[StateQuery {
            path: vec![serialize_key(0u8)],
        }]);
        assert!(results[0].error.as_ref().unwrap().contains("collection"));
    }

    #[test]
    fn query_batch() {
        let map_key: AlignedValue = 1u64.into();
        let mut map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        map = map.insert(map_key, make_cell(10));
        let root: StateValue<InMemoryDB> = StateValue::Array(Array::from(vec![
            StateValue::Map(map),
            make_cell(42),
        ]));
        let results = query_state(&root, &[
            StateQuery { path: vec![serialize_key(1u8)] },            // cell
            StateQuery { path: vec![serialize_key(0u8), serialize_key(1u64)] }, // map entry
        ]);
        assert_eq!(results.len(), 2);
        assert!(results[0].value.is_some());
        assert!(results[1].value.is_some());
    }

    #[test]
    fn query_echoes_query() {
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![make_cell(42)]));
        let path = vec![serialize_key(0u8)];
        let results = query_state(&root, &[StateQuery { path: path.clone() }]);
        assert_eq!(results[0].query.path, path);
    }
}
