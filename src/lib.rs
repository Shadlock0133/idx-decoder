//! ## About
//! An IDX file format decoding library. (Currently WIP)
//! 
//! The main type is [`IDXDecoder`]. It implements Iterator whose Item
//! correspond to items of file format.
//! 
//! ### Type parameters
//! 
//! [`IDXDecoder`] takes three type parameters.
//! - `R`: Reader from which data is taken. Can be file, network stream etc.
//! - `T`: Type of items produced by Iterator. E.g. U8, I16, F32.
//!   
//!   All possible types can be found in [`types`](types/index.html) module
//! - `D`: Type-level integer of dimensions. Must be less than 256.
//!   
//!   If it's less than 128 use nalgebra's U* types.
//!   For value >=128 use typenum's consts.
//! 
//! ### Dimensions
//! 
//! For one-dimensional decoder returns simply items.
//! 
//! For more dimensions, output is a `Vec` of values containing a single item.
//! 
//! E.g. a 3-dimensional decoder where items are of size 4x4 will return `Vec`s
//! of length 16.
//! 
//! First dimension of decoder corresponds to amount of items left.
//! 
//! ## Caveats
//! 
//! Currently decoder only implements Iterator for 1 and 3 dimensions.
//! It's simply because I didn't implement other.
//! 
//! Crate also assumes that items are stored in big endian way, just like sizes.
//! 
//! If you found a bug or the crate is missing some functionality,
//! add an issue or send a pull request.
//! 
//! ## Example
//! ```ignore
//! let file = std::fs::File::open("data.idx")?;
//! let decode = idx_decoder::IDXDecoder::<_, idx_decoder::types::U8, nalgebra::U1>::new(file)?;
//! for item in decode {
//!     println!("Item: {}", item);
//! }
//! ```
//! 
//! ## Acknowledgement
//! This crate is implemented according to file format
//! found at <http://yann.lecun.com/exdb/mnist/>
//! 
//! [`IDXDecoder`]: struct.IDXDecoder.html

use std::{convert::TryInto, io::{self, Read}, marker::PhantomData};
use nalgebra::{self as na, VectorN, DimName, allocator::Allocator, DefaultAllocator};
// use typenum::{self as tn, type_operators::IsLess};
use failure::Fail;

/// Types used by [`IDXDecoder`](struct.IDXDecoder.html) to specify iterator's output type
pub mod types {
    use std::{io::Read, mem::size_of};

    #[doc(hidden)]
    mod private { pub trait Sealed {} }
    use private::Sealed;

    /// Trait implemented by output types used by IDXDecoder's iterator
    /// 
    /// It can't be implemented outside this crate.
    pub trait Type: Sealed {
        const VALUE: u8;
        type TypeValue;
    }

    // implemented by types that can be read from reader using big endiann
    #[doc(hidden)]
    pub trait BEReadable<R>: Sized {
        fn read_self(r: &mut R) -> Option<Self>;
    }

    macro_rules! new_type_int {
        ( $( $vis:vis $name:ident : $tv:ty = $value:expr,)* ) => {
            $(
                $vis struct $name;
                impl Sealed for $name {}
                impl Type for $name {
                    type TypeValue = $tv;
                    const VALUE: u8 = $value;
                }

                impl<R: Read> BEReadable<R> for $tv {
                    fn read_self(r: &mut R) -> Option<Self> {
                        let mut buf = [0u8; size_of::<Self>()];
                        r.read_exact(&mut buf).ok()?;
                        Some(Self::from_be_bytes(buf))
                    }
                }
            )*
        }
    }

    macro_rules! new_type_f {
        ( $( $vis:vis $name:ident : $uint:ty as $tv:ty = $value:expr,)* ) => {
            $(
                $vis struct $name;
                impl Sealed for $name {}
                impl Type for $name {
                    type TypeValue = $tv;
                    const VALUE: u8 = $value;
                }

                impl<R: Read> BEReadable<R> for $tv {
                    fn read_self(r: &mut R) -> Option<Self> {
                        let mut buf = [0u8; size_of::<Self>()];
                        r.read_exact(&mut buf).ok()?;
                        Some(Self::from_bits(<$uint>::from_be_bytes(buf)))
                    }
                }
            )*
        }
    }

    new_type_int!(
        pub U8: u8 = 0x08,
        pub I8: i8 = 0x09,
        pub I16: i16 = 0x0b,
        pub I32: i32 = 0x0c,
    );
    new_type_f!(
        pub F32: u32 as f32 = 0x0d,
        pub F64: u64 as f64 = 0x0e,
    );
}

use types::*;

/// The decoder. Check [crate level docs](index.html) for more informations
pub struct IDXDecoder<R, T: Type, D: DimName>
where
    DefaultAllocator: Allocator<u32, D>
{
    reader: R,
    output_type: PhantomData<T>,
    dimensions: VectorN<u32, D>,
}

/// Error type return by `IDXDecoder::new`
#[derive(Debug, Fail)]
pub enum IDXError {
    #[fail(display = "Wrong magic, first two bytes should be zero")]
    WrongMagic,
    #[fail(display = "Wrong type, expected {}, got {}", _0, _1)]
    WrongType(u8, u8),
    #[fail(display = "Wrong number of dimensions, expected {}, got {}", _0, _1)]
    WrongDimensions(u8, u8),
    #[fail(display = "{}", _0)]
    IOError(#[cause] io::Error),
}

impl From<io::Error> for IDXError {
    fn from(error: io::Error) -> Self {
        IDXError::IOError(error)
    }
}

impl<R: Read, T: Type, D: DimName> IDXDecoder<R, T, D>
where
    // D: IsLess<tn::consts::U256>,
    DefaultAllocator: Allocator<u32, D>
{
    /// Returns error in case provided types aren't valid 
    pub fn new(mut reader: R) -> Result<Self, IDXError> {
        // Read magic and check if it's valid
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        if buf[0] != 0 || buf[1] != 0 { Err(IDXError::WrongMagic)? }
        if buf[2] != T::VALUE { Err(IDXError::WrongType(T::VALUE, buf[2]))? }
        let dims: u8 = D::dim().try_into().ok()?;
        if buf[3] != dims { Err(IDXError::WrongDimensions(dims, buf[3]))? }

        // Read dimensions
        // To simplify code we treat amount of items as first dimension
        let mut dimensions: VectorN<u32, D> = na::zero();
        for d in dimensions.iter_mut() {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *d = u32::from_be_bytes(buf);
        }
        Ok(IDXDecoder { reader, output_type: PhantomData, dimensions })
    }

    /// Size of return values.
    /// 
    /// First dimension of decoder corresponds to amount of items left.
    pub fn dimensions(&self) -> VectorN<u32, D> {
        self.dimensions.clone()
    }
}

impl<R: Read, T: Type> Iterator for IDXDecoder<R, T, na::U1>
where
    DefaultAllocator: Allocator<u32, na::U1>,
    T::TypeValue: BEReadable<R>,
{
    type Item = T::TypeValue;
    fn next(&mut self) -> Option<Self::Item> {
        if self.dimensions[0] > 0 {
            self.dimensions[0] -= 1;
            T::TypeValue::read_self(&mut self.reader)
        } else {
            None
        }
    }
}

impl<R: Read, T: Type> Iterator for IDXDecoder<R, T, na::U3>
where
    DefaultAllocator: Allocator<u32, na::U3>,
    T::TypeValue: Default + Clone + BEReadable<R>,
{
    type Item = Vec<T::TypeValue>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.dimensions[0] > 0 {
            self.dimensions[0] -= 1;
            let as_usize = |n: u32| -> Option<usize> { n.try_into().ok() };
            let len = as_usize(self.dimensions[1])?.checked_mul(as_usize(self.dimensions[2])?)?;
            let mut items = vec![Default::default(); len];
            for item in items.iter_mut() {
                *item = T::TypeValue::read_self(&mut self.reader)?
            }
            Some(items)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.dimensions[0].try_into().ok())
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn example_1d() {
        const DATA: &[u8] = &[
            // magic, type u8, 1 dim
            0, 0, 8, 1,
            // len as big endiann u32
            0, 0, 0, 3,
            // items
            1, 2, 3];
        let reader = std::io::Cursor::new(DATA);
        let mut decoder = IDXDecoder::<_, U8, nalgebra::U1>::new(reader)
            .expect("Decoder creation error");
        assert_eq!(decoder.next(), Some(1));
        assert_eq!(decoder.next(), Some(2));
        assert_eq!(decoder.next(), Some(3));
        assert_eq!(decoder.next(), None);
    }

    #[test]
    fn example_3d() {
        const DATA: &[u8] = &[
            // magic, type u8, 1 dim
            0, 0, 8, 3,
            // lens as big endiann u32: 3 matrices of 2x2
            0, 0, 0, 3,
            0, 0, 0, 2,
            0, 0, 0, 2,
            // items
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12];
        let reader = std::io::Cursor::new(DATA);
        let mut decoder = IDXDecoder::<_, U8, nalgebra::U3>::new(reader)
            .expect("Decoder creation error");
        assert_eq!(decoder.next(), Some(vec![1, 2, 3, 4]));
        assert_eq!(decoder.next(), Some(vec![5, 6, 7, 8]));
        assert_eq!(decoder.next(), Some(vec![9, 10, 11, 12]));
        assert_eq!(decoder.next(), None);
    }
}
