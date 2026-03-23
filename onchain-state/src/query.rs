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
//! Instead of serializing the entire state DAG (O(n) via `serialize_to_node_list`),
//! these functions navigate the `StateValue` tree lazily and return only the
//! requested values.

use crate::state::StateValue;
use serialize::Deserializable;
use storage::db::DB;

/// A single query targeting a field path and optional key.
#[derive(Clone, Debug)]
pub struct StateQuery {
    /// Path of indices through nested `StateValue::Array` nodes.
    pub field_path: Vec<u32>,
    /// Optional key bytes for map lookup. Serialized `AlignedValue`.
    pub key: Option<Vec<u8>>,
}

/// Result of a single state query.
#[derive(Clone, Debug)]
pub struct StateQueryResult {
    /// Whether the requested value exists.
    pub found: bool,
    /// Serialized value bytes (tagged format). `None` if not found or error.
    pub value: Option<Vec<u8>>,
    /// Per-query error message. `None` if successful.
    pub error: Option<String>,
}

/// Resolve a list of queries against a contract's root `StateValue`.
///
/// Each query navigates `field_path` through nested `Array` nodes, then:
/// - **Map + key**: deserializes the key as `AlignedValue`, looks up via
///   `HashMap::get` (O(log n) lazy MPT traversal), and `tagged_serialize`s
///   the entry.
/// - **Map without key**: returns the map size as `u64` little-endian bytes.
/// - **Key on non-map**: returns a per-query error.
/// - **Everything else**: `tagged_serialize`s the value at the resolved path.
pub fn query_state<D: DB>(root: &StateValue<D>, queries: &[StateQuery]) -> Vec<StateQueryResult> {
    let serialize_sv = |sv: &StateValue<D>| -> Result<Vec<u8>, String> {
        let size = serialize::tagged_serialized_size(sv);
        let mut buf = Vec::with_capacity(size);
        serialize::tagged_serialize(sv, &mut buf).map_err(|e| format!("serialize: {e}"))?;
        Ok(buf)
    };

    queries
        .iter()
        .map(|query| {
            let ok =
                |value, found| StateQueryResult { found, value, error: None };
            let err = |msg: String| StateQueryResult {
                found: false,
                value: None,
                error: Some(msg),
            };

            // Navigate field_path through nested Arrays
            let mut current = root;
            for &idx in &query.field_path {
                match current {
                    StateValue::Array(arr) => match arr.get(idx as usize) {
                        Some(child) => current = child,
                        None => return err(format!("index {idx} out of bounds")),
                    },
                    _ => return err("expected array".into()),
                }
            }

            match (&query.key, current) {
                (Some(key_bytes), StateValue::Map(map)) => {
                    let mut reader: &[u8] = key_bytes.as_slice();
                    match <base_crypto::fab::AlignedValue as Deserializable>::deserialize(
                        &mut reader,
                        0,
                    ) {
                        Ok(key) => match map.get(&key) {
                            Some(sp) => {
                                serialize_sv(&*sp).map_or_else(err, |b| ok(Some(b), true))
                            }
                            None => ok(None, false),
                        },
                        Err(e) => err(format!("bad key: {e}")),
                    }
                }
                (None, StateValue::Map(map)) => {
                    ok(Some((map.size() as u64).to_le_bytes().to_vec()), true)
                }
                (Some(_), _) => err("key provided but field is not a map".into()),
                (None, val) => serialize_sv(val).map_or_else(err, |b| ok(Some(b), true)),
            }
        })
        .collect()
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

    #[test]
    fn query_cell_value() {
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![make_cell(42), make_cell(100)]));

        let results = query_state(
            &root,
            &[StateQuery {
                field_path: vec![0],
                key: None,
            }],
        );

        assert_eq!(results.len(), 1);
        assert!(results[0].found);
        assert!(results[0].value.is_some());
        assert!(results[0].error.is_none());
    }

    #[test]
    fn query_map_size() {
        let mut map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        map = map.insert(AlignedValue::from(1u64), make_cell(10));
        map = map.insert(AlignedValue::from(2u64), make_cell(20));

        let root = StateValue::Array(Array::from(vec![StateValue::Map(map)]));

        let results = query_state(
            &root,
            &[StateQuery {
                field_path: vec![0],
                key: None,
            }],
        );

        assert!(results[0].found);
        let size_bytes: [u8; 8] = results[0].value.as_ref().unwrap()[..].try_into().unwrap();
        assert_eq!(u64::from_le_bytes(size_bytes), 2);
    }

    #[test]
    fn query_map_key_found() {
        let key_val: AlignedValue = 42u64.into();
        let mut key_bytes = Vec::new();
        key_val.serialize(&mut key_bytes).unwrap();

        let mut map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        map = map.insert(key_val, make_cell(999));

        let root = StateValue::Array(Array::from(vec![StateValue::Map(map)]));

        let results = query_state(
            &root,
            &[StateQuery {
                field_path: vec![0],
                key: Some(key_bytes),
            }],
        );

        assert!(results[0].found);
        assert!(results[0].value.is_some());
    }

    #[test]
    fn query_map_key_not_found() {
        let key_val: AlignedValue = 999u64.into();
        let mut key_bytes = Vec::new();
        key_val.serialize(&mut key_bytes).unwrap();

        let map = HashMap::<AlignedValue, StateValue<InMemoryDB>, InMemoryDB>::default();
        let root = StateValue::Array(Array::from(vec![StateValue::Map(map)]));

        let results = query_state(
            &root,
            &[StateQuery {
                field_path: vec![0],
                key: Some(key_bytes),
            }],
        );

        assert!(!results[0].found);
        assert!(results[0].value.is_none());
    }

    #[test]
    fn query_index_out_of_bounds() {
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![make_cell(1)]));

        let results = query_state(
            &root,
            &[StateQuery {
                field_path: vec![99],
                key: None,
            }],
        );

        assert!(!results[0].found);
        assert!(results[0].error.as_ref().unwrap().contains("out of bounds"));
    }

    #[test]
    fn query_key_on_non_map() {
        let root: StateValue<InMemoryDB> =
            StateValue::Array(Array::from(vec![make_cell(1)]));

        let results = query_state(
            &root,
            &[StateQuery {
                field_path: vec![0],
                key: Some(vec![0]),
            }],
        );

        assert!(!results[0].found);
        assert!(results[0].error.as_ref().unwrap().contains("not a map"));
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
                StateQuery { field_path: vec![1], key: None },         // cell
                StateQuery { field_path: vec![0], key: None },         // map size
                StateQuery { field_path: vec![0], key: Some(key_bytes) }, // map entry
            ],
        );

        assert_eq!(results.len(), 3);
        assert!(results[0].found); // cell
        assert!(results[1].found); // map size = 1
        assert!(results[2].found); // map entry found
    }
}
