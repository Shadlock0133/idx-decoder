use std::{convert::TryInto, io::{self, Read}, marker::PhantomData, mem::size_of};
use nalgebra::{self as na, VectorN, DimName, allocator::Allocator, DefaultAllocator};
use failure::Fail;

#[doc(hidden)]
mod private { pub trait Sealed {} }
use private::Sealed;

pub trait Type: Sealed {
    const VALUE: u8;
    type TypeValue;
}

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

pub struct IDXDecoder<R, T: Type, D: DimName>
where
    DefaultAllocator: Allocator<u32, D>
{
    reader: R,
    output_type: PhantomData<T>,
    dimensions: VectorN<u32, D>,
}

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
    DefaultAllocator: Allocator<u32, D>
{
    // Return error in case specified generics don't apply to 
    pub fn new(mut reader: R) -> Result<Self, IDXError> {
        let mut buf = [0u8; 4];
        // Read magic
        reader.read_exact(&mut buf)?;
        if buf[0] != 0 || buf[1] != 0 { Err(IDXError::WrongMagic)? }
        if buf[2] != T::VALUE { Err(IDXError::WrongType(T::VALUE, buf[2]))? }
        if buf[3] != D::dim() as u8 {
            Err(IDXError::WrongDimensions(D::dim() as u8, buf[3]))?
        }
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
    T::TypeValue: Default + BEReadable<R>,
{
    type Item = Vec<T::TypeValue>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.dimensions[0] > 0 {
            self.dimensions[0] -= 1;
            let as_usize = |n: u32| -> Option<usize> { n.try_into().ok() };
            let len = as_usize(self.dimensions[1])?.checked_mul(as_usize(self.dimensions[2])?)?;
            let items = (0..)
                .map(|_| T::TypeValue::read_self(&mut self.reader).ok_or(()))
                .take(len)
                .collect::<Result<Vec<_>, ()>>().ok()?;
            Some(items).filter(|i: &Self::Item| i.len() == len)
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
    fn example() {
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
}