// SPDX-License-Identifier: Apache-2.0 OR MIT
//! JSON helpers for Safetensors headers and HF index files (no duplicate keys).

use super::model_load;
use crate::error::Result;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

struct NoDuplicateValue(Value);

pub(super) fn stringify_metadata(
    prefix: &str,
    metadata: BTreeMap<String, Value>,
) -> BTreeMap<String, String> {
    metadata
        .into_iter()
        .map(|(key, value)| (format!("{prefix}:{key}"), stringify_json_value(&value)))
        .collect()
}

pub(super) fn parse_json_rejecting_duplicate_keys(
    bytes: &[u8],
    path: &Path,
    context: &str,
) -> Result<Value> {
    serde_json::from_slice::<NoDuplicateValue>(bytes)
        .map(|value| value.0)
        .map_err(|e| model_load(path, format!("parse {context} JSON: {e}")))
}

pub(super) fn parse_metadata_object(
    path: &Path,
    value: &Value,
) -> Result<BTreeMap<String, String>> {
    let object = value.as_object().ok_or_else(|| {
        model_load(
            path,
            "Safetensors __metadata__ must be a JSON object of strings".into(),
        )
    })?;
    object
        .iter()
        .map(|(key, value)| {
            let value = value.as_str().ok_or_else(|| {
                model_load(
                    path,
                    format!("Safetensors __metadata__ key '{key}' must be a string"),
                )
            })?;
            Ok((key.clone(), value.to_string()))
        })
        .collect()
}

pub(super) fn stringify_json_value(value: &Value) -> String {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string())
}

impl<'de> Deserialize<'de> for NoDuplicateValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(NoDuplicateValueVisitor)
    }
}

struct NoDuplicateValueVisitor;

impl<'de> Visitor<'de> for NoDuplicateValueVisitor {
    type Value = NoDuplicateValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("valid JSON without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        let number = Number::from_f64(value).ok_or_else(|| E::custom("invalid JSON number"))?;
        Ok(NoDuplicateValue(Value::Number(number)))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateValue(Value::String(value.to_string())))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Null))
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Null))
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element::<NoDuplicateValue>()? {
            values.push(value.0);
        }
        Ok(NoDuplicateValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut seen = BTreeSet::new();
        let mut object = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate JSON key '{key}'")));
            }
            let value = map.next_value::<NoDuplicateValue>()?;
            object.insert(key, value.0);
        }
        Ok(NoDuplicateValue(Value::Object(object)))
    }
}
