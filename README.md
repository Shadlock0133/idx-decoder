# idx-decoder

### About
An IDX file format decoding library. (Currently WIP)

The main type is [`IDXDecoder`]. It implements Iterator whose Item
correspond to items of file format.

#### Type parameters

[`IDXDecoder`] takes three type parameters.
- `R`: Reader from which data is taken. Can be file, network stream etc.
- `T`: Type of items produced by Iterator. E.g. U8, I16, F32.

  All possible types can be found in [`types`](types/index.html) module
- `D`: Type-level integer of dimensions. Must be less than 256.

  If it's less than 128 use nalgebra's U* types.
  For value >=128 use typenum's consts.

#### Dimensions

For one-dimensional decoder returns simply items.

For more dimensions, output is a `Vec` of values containing a single item.

E.g. a 3-dimensional decoder where items are of size 4x4 will return `Vec`s
of length 16.

First dimension of decoder corresponds to amount of items left.

### Caveats

Currently decoder only implements Iterator for 1 and 3 dimensions.
It's simply because I didn't implement other.

Crate also assumes that items are stored in big endian way, just like sizes.

If you found a bug or the crate is missing some functionality,
add an issue or send a pull request.

### Example
```rust
let file = std::fs::File::open("data.idx")?;
let decode = idx_decoder::IDXDecoder::<_, idx_decoder::types::U8, nalgebra::U1>::new(file)?;
for item in decode {
    println!("Item: {}", item);
}
```

### Acknowledgement
This crate is implemented according to file format
found at <http://yann.lecun.com/exdb/mnist/>

[`IDXDecoder`]: struct.IDXDecoder.html

License: MIT
