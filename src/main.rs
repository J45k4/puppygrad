mod tensor;

fn main() {
    let mut ten = tensor::Mat::constant(&[&[1.0, 2.0, 3.0], &[1.0,1.0,1.0]]).unwrap();
    let ten2 = tensor::Mat::constant(&[&[2.0, 2.0, 2.0], &[1.0, 1.0, 1.0]]).unwrap();
    let ten3 = tensor::Mat::constant(&[&[2.0, 2.0, 2.0], &[1.0, 1.0, 1.0]]).unwrap();

    ten += ten2;
    // ten *= ten2;

    println!("{}", ten);
}
