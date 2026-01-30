//! Variable-length integer encoding for postings compression
//!
//! Uses VByte encoding (7 bits per byte with continuation bit).

use std::io::{Read, Write, Result};

/// Encode u32 as variable-length integer
pub fn encode_varint(mut value: u32, writer: &mut impl Write) -> Result<usize> {
    let mut bytes_written = 0;
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80; // Set continuation bit
        }
        writer.write_all(&[byte])?;
        bytes_written += 1;
        if value == 0 {
            break;
        }
    }
    Ok(bytes_written)
}

/// Decode variable-length integer
pub fn decode_varint(reader: &mut impl Read) -> Result<u32> {
    let mut value = 0u32;
    let mut shift = 0u32;
    let mut buf = [0u8; 1];

    loop {
        reader.read_exact(&mut buf)?;
        let byte = buf[0];
        value |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 35 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "varint overflow",
            ));
        }
    }
    Ok(value)
}

/// Encode delta-compressed postings list
pub fn encode_postings(doc_ids: &[u32]) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut prev = 0u32;
    for &doc_id in doc_ids {
        let delta = doc_id - prev;
        encode_varint(delta, &mut buf).unwrap();
        prev = doc_id;
    }
    buf
}

/// Decode delta-compressed postings list
pub fn decode_postings(data: &[u8]) -> Vec<u32> {
    let mut reader = std::io::Cursor::new(data);
    let mut doc_ids = Vec::new();
    let mut prev = 0u32;

    while reader.position() < data.len() as u64 {
        match decode_varint(&mut reader) {
            Ok(delta) => {
                prev += delta;
                doc_ids.push(prev);
            }
            Err(_) => break,
        }
    }
    doc_ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_roundtrip() {
        let test_values = vec![0, 1, 127, 128, 16383, 16384, u32::MAX];
        for &value in &test_values {
            let mut buf = Vec::new();
            encode_varint(value, &mut buf).unwrap();
            let decoded = decode_varint(&mut std::io::Cursor::new(&buf)).unwrap();
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn test_postings_compression() {
        let doc_ids = vec![1, 5, 10, 100, 1000, 10000];
        let compressed = encode_postings(&doc_ids);
        let decompressed = decode_postings(&compressed);
        assert_eq!(decompressed, doc_ids);

        // Check compression ratio
        let uncompressed_size = doc_ids.len() * 4; // u32 = 4 bytes
        assert!(compressed.len() < uncompressed_size);
    }

    #[test]
    fn test_empty_postings() {
        let doc_ids: Vec<u32> = vec![];
        let compressed = encode_postings(&doc_ids);
        let decompressed = decode_postings(&compressed);
        assert_eq!(decompressed, doc_ids);
    }
}
