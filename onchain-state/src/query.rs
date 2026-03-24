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
//! Provides navigation helpers and a batch query function that resolve
//! specific fields/keys without serializing the full DAG.

use crate::state::StateValue;
use serialize::Deserializable;
use storage::db::DB;

// ---------------------------------------------------------------------------
// Navigation helpers (generic over D: DB)
// ---------------------------------------------------------------------------

/// Error returned by state navigation functions.
#[derive(Clone, Debug)]
pub enum NavError {
    /// The path expected an Array but found a different variant.
    ExpectedArray,
    /// The array index is out of bounds.
    IndexOutOfBounds(u8),
}

impl std::fmt::Display for NavError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NavError::ExpectedArray => write!(f, "expected array"),
            NavError::IndexOutOfBounds(idx) => write!(f, "index {idx} out of bounds"),
        }
    }
}

/// Navigate one level into a `StateValue::Array` by index.
pub fn get_field<D: DB>(sv: &StateValue<D>, index: u8) -> Result<&StateValue<D>, NavError> {
    match sv {
        StateValue::Array(arr) => arr
            .get(index as usize)
            .ok_or(NavError::IndexOutOfBounds(index)),
        _ => Err(NavError::ExpectedArray),
    }
}

/// Navigate multiple levels through nested `StateValue::Array` nodes.
pub fn get_field_path<'a, D: DB>(
    sv: &'a StateValue<D>,
    path: &[u8],
) -> Result<&'a StateValue<D>, NavError> {
    let mut current = sv;
    for &index in path {
        current = get_field(current, index)?;
    }
    Ok(current)
}

// ---------------------------------------------------------------------------
// Batch query
// ---------------------------------------------------------------------------

/// A single query targeting a path and optional key.
#[derive(Clone, Debug)]
pub struct StateQuery {
    /// Path of indices through nested `StateValue::Array` nodes.
    pub path: Vec<u8>,
    /// Optional key bytes for collection lookups. Serialized `AlignedValue`
    /// for Map/Set, position bytes for MerkleTree.
    pub key: Option<Vec<u8>>,
}

/// Result of a single state query.
#[derive(Clone, Debug)]
pub struct StateQueryResult {
    /// The original query.
    pub query: StateQuery,
    /// Serialized value bytes (tagged format). `None` if not found.
    pub value: Option<Vec<u8>>,
    /// Per-query error message. `None` if successful.
    pub error: Option<String>,
}

/// Resolve a list of queries against a contract's root `StateValue`.
///
/// Each query navigates `path` through nested `Array` nodes, then:
/// - **Map/Set + key**: looks up via `HashMap::get` (O(log n)).
/// - **Map/Set without key**: error (serializing the full collection is O(n)).
/// - **Key on non-collection**: error.
/// - **Everything else**: `tagged_serialize`s the value.
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

            let current = match get_field_path(root, &query.path) {
                Ok(sv) => sv,
                Err(e) => return err(e.to_string()),
            };

            match (&query.key, current) {
                (Some(key_bytes), StateValue::Map(map)) => {
                    let mut reader: &[u8] = key_bytes.as_slice();
                    match <base_crypto::fab::AlignedValue as Deserializable>::deserialize(
                        &mut reader,
                        0,
                    ) {
                        Ok(key) => match map.get(&key) {
                            Some(sp) => serialize_sv(&*sp).map_or_else(err, |b| ok(Some(b))),
                            None => ok(None),
                        },
                        Err(e) => err(format!("bad key: {e}")),
                    }
                }
                (None, StateValue::Map(_)) => err("key required for map fields".into()),
                (Some(_), _) => err("key provided but field is not a map".into()),
                (None, val) => serialize_sv(val).map_or_else(err, |b| ok(Some(b))),
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
    use base_crypto::fab::AlignedValue;
    use serialize::Serializable;
    use storage::db::InMemoryDB;
    use storage::storage::{Array, HashMap};

    fn make_cell<D: DB>(val: u64) -> StateValue<D> {
        StateValue::from(val)
    }

    // -- Navigation helper tests --

    #[test]
    fn get_field_returns_element() {
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![make_cell(42), make_cell(100)]));
        let sv = get_field(&root, 1).unwrap();
        assert!(matches!(sv, StateValue::Cell(_)));
    }

    #[test]
    fn get_field_out_of_bounds() {
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![make_cell(1)]));
        assert!(matches!(get_field(&root, 5), Err(NavError::IndexOutOfBounds(5))));
    }

    #[test]
    fn get_field_on_non_array() {
        let root: StateValue<InMemoryDB> = make_cell(1);
        assert!(matches!(get_field(&root, 0), Err(NavError::ExpectedArray)));
    }

    #[test]
    fn get_field_path_navigates_nested() {
        let inner = StateValue::Array(Array::from(vec![make_cell(99)]));
        let root: StateValue<InMemoryDB> = StateValue::Array(Array::from(vec![inner]));
        let sv = get_field_path(&root, &[0, 0]).unwrap();
        assert!(matches!(sv, StateValue::Cell(_)));
    }

    // -- Query tests --

    #[test]
    fn query_cell_value() {
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![make_cell(42)]));
        let results = query_state(&root, &[StateQuery { path: vec![0], key: None }]);
        assert!(results[0].value.is_some());
        assert!(results[0].error.is_none());
    }

    #[test]
    fn query_map_without_key_errors() {
        let map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        let root = StateValue::Array(Array::from(vec![StateValue::Map(map)]));
        let results = query_state(&root, &[StateQuery { path: vec![0], key: None }]);
        assert!(results[0].error.as_ref().unwrap().contains("key required"));
    }

    #[test]
    fn query_map_key_found() {
        let key_val: AlignedValue = 42u64.into();
        let mut key_bytes = Vec::new();
        key_val.serialize(&mut key_bytes).unwrap();
        let mut map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        map = map.insert(key_val, make_cell(999));
        let root = StateValue::Array(Array::from(vec![StateValue::Map(map)]));
        let results = query_state(&root, &[StateQuery { path: vec![0], key: Some(key_bytes) }]);
        assert!(results[0].value.is_some());
    }

    #[test]
    fn query_map_key_not_found() {
        let key_val: AlignedValue = 999u64.into();
        let mut key_bytes = Vec::new();
        key_val.serialize(&mut key_bytes).unwrap();
        let map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        let root = StateValue::Array(Array::from(vec![StateValue::Map(map)]));
        let results = query_state(&root, &[StateQuery { path: vec![0], key: Some(key_bytes) }]);
        assert!(results[0].value.is_none());
        assert!(results[0].error.is_none());
    }

    #[test]
    fn query_index_out_of_bounds() {
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![make_cell(1)]));
        let results = query_state(&root, &[StateQuery { path: vec![99], key: None }]);
        assert!(results[0].error.as_ref().unwrap().contains("out of bounds"));
    }

    #[test]
    fn query_result_echoes_query() {
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![make_cell(42)]));
        let results = query_state(&root, &[StateQuery { path: vec![0], key: None }]);
        assert_eq!(results[0].query.path, vec![0]);
    }

    #[test]
    fn query_batch() {
        let key_val: AlignedValue = 1u64.into();
        let mut key_bytes = Vec::new();
        key_val.serialize(&mut key_bytes).unwrap();
        let mut map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        map = map.insert(key_val, make_cell(10));
        let root: StateValue<InMemoryDB> = StateValue::Array(Array::from(vec![
            StateValue::Map(map),
            make_cell(42),
        ]));
        let results = query_state(
            &root,
            &[
                StateQuery { path: vec![1], key: None },
                StateQuery { path: vec![0], key: Some(key_bytes) },
            ],
        );
        assert_eq!(results.len(), 2);
        assert!(results[0].value.is_some());
        assert!(results[1].value.is_some());
    }
}
