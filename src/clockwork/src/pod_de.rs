//! A plain old data(POD) deserializer for both TVM and Clockwork models
//! Sadly, TVM and Clockwork use different POD serializers.
//! For primitive data type, it can be deserialized by simply memory copy.
//! For regular Struct, it can be deserialized by sequential access.
//! For container type like HasMap, String, Vec, it can be deserialized by read the size of the
//! container first, and read the remaining data.
//! **caveat: TVM and Clockwork use C++ String for some binary data, too.(e.g: cuda's fatbin).**
//! Use Vec<u8> instead of String in this case**
//! TVM -> https://github.com/mtrempoltsev/pods
//! Clockwork -> https://github.com/dmlc/dmlc-core/blob/main/include/dmlc/serializer.h

use serde::de::{self, Deserialize, MapAccess, SeqAccess, Visitor};
use std::fmt::{self, Display};

#[derive(Debug)]
pub enum Error {
    Message(String),
    Unimplemnted,
}

impl std::error::Error for Error {}

impl de::Error for Error {
    fn custom<T>(msg: T) -> Self
    where
        T: Display,
    {
        Error::Message(msg.to_string())
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Message(msg) => f.write_str(msg),
            Error::Unimplemnted => f.write_str("This type is currently not implemented"),
        }
    }
}

/// TVM's serializer use u64 as container size.
/// Clockwork's serializer use u32 as container size.
pub(super) enum PodType {
    TVM,
    Clockwork,
}

pub(super) struct Deserializer<'de> {
    bytes: &'de [u8],
    pod_type: PodType,
}

impl<'de> Deserializer<'de> {
    pub fn from_bytes(bytes: &'de [u8], pod_type: PodType) -> Self {
        Deserializer { bytes, pod_type }
    }

    fn parse_primitive<T: Copy>(&mut self) -> T {
        let size = std::mem::size_of::<T>();
        let (data, remainder) = self.bytes.split_at(size);
        self.bytes = remainder;
        // SAFETY
        // data pointer is from a slice, hence a valid address
        unsafe { std::ptr::read(data.as_ptr() as *const T) }
    }

    fn get_container_size(&mut self) -> usize {
        match self.pod_type {
            PodType::TVM => {
                let size_u64: u64 = self.parse_primitive();
                size_u64 as usize
            }
            PodType::Clockwork => {
                let size_u32: u32 = self.parse_primitive();
                size_u32 as usize
            }
        }
    }
}

macro_rules! deserialize_primitive {
    ($de:ident, $vis: ident) => {
        fn $de<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where
            V: Visitor<'de>,
        {
            visitor.$vis(self.parse_primitive())
        }
    };
}

impl<'de> de::Deserializer<'de> for &mut Deserializer<'de> {
    type Error = Error;
    fn deserialize_any<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        Err(Error::Unimplemnted)
    }

    deserialize_primitive!(deserialize_i8, visit_i8);
    deserialize_primitive!(deserialize_i16, visit_i16);
    deserialize_primitive!(deserialize_i32, visit_i32);
    deserialize_primitive!(deserialize_i64, visit_i64);
    deserialize_primitive!(deserialize_u8, visit_u8);
    deserialize_primitive!(deserialize_u16, visit_u16);
    deserialize_primitive!(deserialize_u32, visit_u32);
    deserialize_primitive!(deserialize_u64, visit_u64);
    deserialize_primitive!(deserialize_f32, visit_f32);
    deserialize_primitive!(deserialize_f64, visit_f64);

    fn deserialize_struct<V>(
        mut self,
        _name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_seq(InternalAccess::new(&mut self, fields.len()))
    }

    fn deserialize_map<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let size = self.get_container_size();
        visitor.visit_map(InternalAccess::new(&mut self, size))
    }

    fn deserialize_seq<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let size = self.get_container_size();
        visitor.visit_seq(InternalAccess::new(&mut self, size))
    }

    serde::forward_to_deserialize_any! {
        bool i128 u128 char str string
        bytes byte_buf option unit unit_struct newtype_struct tuple
        tuple_struct enum identifier ignored_any
    }
}

struct InternalAccess<'de, 'a> {
    de: &'a mut Deserializer<'de>,
    remain: usize,
}

impl<'de, 'a> InternalAccess<'de, 'a> {
    fn new(de: &'a mut Deserializer<'de>, size: usize) -> InternalAccess<'de, 'a> {
        InternalAccess { de, remain: size }
    }
}

impl<'de> MapAccess<'de> for InternalAccess<'de, '_> {
    type Error = Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: de::DeserializeSeed<'de>,
    {
        match self.remain {
            0 => Ok(None),
            _ => {
                self.remain -= 1;
                seed.deserialize(&mut *self.de).map(Some)
            }
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        seed.deserialize(&mut *self.de)
    }
}

impl<'de> SeqAccess<'de> for InternalAccess<'de, '_> {
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: de::DeserializeSeed<'de>,
    {
        match self.remain {
            0 => Ok(None),
            _ => {
                self.remain -= 1;
                seed.deserialize(&mut *self.de).map(Some)
            }
        }
    }
}

pub(super) fn from_bytes<'a, T>(bytes: &'a [u8], pod_type: PodType) -> Result<T, Error>
where
    T: Deserialize<'a>,
{
    let mut de = Deserializer::from_bytes(bytes, pod_type);
    let t = T::deserialize(&mut de)?;
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32() {
        let data: &[u8] = &[0xcd, 0xcc, 0x4c, 0x3e]; // 0.2 in f32
        let result: f32 = from_bytes(data, PodType::TVM).unwrap();
        assert!(result.partial_cmp(&0.2).unwrap() == std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_f64() {
        let data: &[u8] = &[0x9a, 0x99, 0x99, 0x99, 0x99, 0x99, 0xc9, 0x3f]; // 0.2 in f64
        let result: f64 = from_bytes(data, PodType::TVM).unwrap();
        assert!(result.partial_cmp(&0.2).unwrap() == std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_i8() {
        let data: &[u8] = &[0xf0]; // -16 in i8
        let result: i8 = from_bytes(data, PodType::TVM).unwrap();
        assert_eq!(result, -16);
    }

    #[test]
    fn test_u8() {
        let data: &[u8] = &[0xab];
        let result: u8 = from_bytes(data, PodType::TVM).unwrap();
        assert_eq!(result, 0xab);
    }

    #[test]
    fn test_i32() {
        let data: &[u8] = &[0x12, 0x34, 0x56, 0x78];
        let result: i32 = from_bytes(data, PodType::TVM).unwrap();
        dbg!(result);
        assert_eq!(result, 0x78563412);
    }

    #[derive(Debug, serde::Deserialize)]
    struct Foo {
        foo: u8,
        bar: Bar,
    }

    #[derive(Debug, serde::Deserialize)]
    struct Bar {
        bar: u32,
    }

    #[test]
    fn test_struct() {
        let data: &[u8] = &[0x12, 0x34, 0x56, 0x78, 0xab];
        let result: Foo = from_bytes(data, PodType::TVM).unwrap();
        assert_eq!(result.foo, 0x12);
        assert_eq!(result.bar.bar, 0xab785634);
    }

    #[test]
    fn test_vec_tvm() {
        let data: &[u8] = &[0x2, 0, 0, 0, 0, 0, 0, 0, 0x12, 0x34, 0x56, 0x78];
        let result: Vec<u16> = from_bytes(data, PodType::TVM).unwrap();
        assert_eq!(result, vec![0x3412, 0x7856]);
    }

    #[test]
    fn test_vec_clockwork() {
        let data: &[u8] = &[0x2, 0, 0, 0, 0x12, 0x34, 0x56, 0x78];
        let result: Vec<u16> = from_bytes(data, PodType::Clockwork).unwrap();
        assert_eq!(result, vec![0x3412, 0x7856]);
    }

    #[test]
    fn test_map() {
        let data: &[u8] = &[0x2, 0, 0, 0, 0, 0, 0, 0, 0x12, 0x34, 0x56, 0x78];
        let result: std::collections::HashMap<u8, u8> = from_bytes(data, PodType::TVM).unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert(0x12, 0x34);
        map.insert(0x56, 0x78);
        assert_eq!(result, map);
    }
}
