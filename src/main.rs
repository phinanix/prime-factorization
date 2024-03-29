#![allow(unused)]
use std::env;
use std::fmt::Debug;

use num_bigint::{BigUint, BigInt, ToBigInt, RandBigInt};
use num_traits::{Zero, One, Signed, Pow, ToPrimitive};
use num_integer::{Integer};


fn trial_divide(n: &BigUint, limit: &BigUint) -> Option<BigUint> {
  let mut divisor: BigUint = 3u8.into();
  let two = 2u8;
  if Zero::is_zero(&(n % two)) {
    return Some(two.into()); 
  }
  while &divisor < &limit {
    if Zero::is_zero(&(n % &divisor)) {
      return Some(divisor)
    }
    divisor += two;
  }
  return None
}

fn trial_divide_u32(n: u32, limit: u32) -> Option<u32> {
  match trial_divide(&n.into(), &limit.into()) {
    None => None, 
    Some(big_int) => match big_int.try_into() {
      Err(_) => panic!("answer too large"),
      Ok(ans) => Some(ans)
    }
  }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RhoState(BigInt, BigInt);

enum SeedOrRhoState{
  Seed(BigInt),
  State(RhoState),
}
use SeedOrRhoState::*;

fn pollard_rho(n: BigInt, iters: u64, seed_or_start: SeedOrRhoState) 
  -> Result<BigInt, RhoState> {
    let mut cur_state = match seed_or_start {
      Seed(seed) => RhoState(seed.clone(), seed),
      State(state) => state,
    };
    fn rho_iter(n: &BigInt, x: &BigInt) -> BigInt {
      let one: BigInt = One::one();
      (x * x + one) % n
    }
    for iter in (0..iters) {
      if iter % 1000000 == 0 {
        println!("iter count {}", iter);
      }
      cur_state = RhoState(rho_iter(&n, &cur_state.0), 
        rho_iter(&n, &rho_iter(&n, &cur_state.1)));
      let cur_diff : BigInt = (&cur_state.0 - &cur_state.1).abs();
      let divisor = &cur_diff.gcd(&n);
      if divisor != &One::one() && divisor != &n {
        return Ok(divisor.clone())
      }
    }
    println!("final state {:?}", cur_state);
    return Err(cur_state)
}

fn basic_factorize(mut n: BigInt, iters: u64, mut seed: u64) -> Vec<BigInt> {
  let mut out = vec![];
  while let Some(divisor) = trial_divide(&n.clone().try_into().unwrap(), 
    &1000000u32.into()) {
      let idiv : BigInt = divisor.into();
      n = n / &idiv;
      out.push(idiv);
      println!("{:?}", &out);
      if One::is_one(&n) {break}
  }
  if One::is_one(&n) {return out}

  while let Ok(divisor) = pollard_rho(n.clone(), iters, Seed(seed.into())) {
    seed += 2;
    n = n / &divisor;
    out.push(divisor);
    println!("{:?}", &out);
    if One::is_one(&n) {break} 
  }
  out.push(n);
  return out 
}

fn rho_u32(n: u32, iters: u64, seed: u32) -> Option<u32> {
  match pollard_rho(n.into(), iters, Seed(seed.into())) {
    Err(_) => None,
    Ok(divisor) => match divisor.try_into() {
      Err(_) => panic!("too large2"),
      Ok(ans) => Some(ans),
    }
  }
}


fn extract_2s(n: &BigUint) -> (BigUint, u32) {
  let mut num_twos = 0; 
  let mut out = n.clone();
  while Zero::is_zero(&(&out % 2u8)) {
    num_twos += 1;
    out /= 2u8; 
  }
  return (out, num_twos)
}

fn miller_rabin(n: &BigUint, rounds: u32) -> bool {
  dbg!(&n);
  let mut rng = rand::thread_rng(); 
  let n_m1 = n-1u8;
  let (odd_base, num_twos) = extract_2s(&n_m1);
  for round in (0.. rounds) {
    let base = rng.gen_biguint_range(&2u8.into(), &(&n_m1-1u8));
    let mut abs_val_one_seen = false; 
    let mut base_to_pow = base.modpow(&odd_base, &n);
    if One::is_one(&base_to_pow) || base_to_pow == n_m1 {
      abs_val_one_seen = true;
    }
    for i in (0..num_twos) {

      base_to_pow = base_to_pow.modpow(&2u8.into(), &n);
      if abs_val_one_seen && !One::is_one(&base_to_pow) && base_to_pow != n_m1 {
        return false;
      } else if !abs_val_one_seen && 
          (One::is_one(&base_to_pow) || base_to_pow == n_m1 ){
        abs_val_one_seen = true;
      }        
    }
    if !One::is_one(&base_to_pow) {
      return false;
    }
  }
  return true 
}

fn is_prime(n: &BigUint) -> bool {
  let td_limit = 10000u32.into();
  match trial_divide(n, &td_limit) {
    Some(d) => {if d < *n {return false}}, 
    None => (),
  }
  if n < &td_limit.pow(2u8) {return true}
  return miller_rabin(n, 25);
  // TODO: ppw test
}

//good: a + b 

//bad: a.my_plus(b)

trait FactorGroup : Sized + Clone + Debug {
  type Data;
  fn identity(data: &Self::Data) -> Self; 
  fn invert(&self, data: &Self::Data, n: &BigInt) -> Result<Self, BigInt>; 
  fn compose(&self, rhs: &Self, data: &Self::Data, n: &BigInt) -> Result<Self, BigInt>; 
  fn double(&self, data: &Self::Data, n: &BigInt) -> Result<Self, BigInt> {
    self.compose(self, data, n)
  }
}

//x ^ (100101)_2 = 37_10
//x ^ 10 = x * x 
//x ^ 100 = (x ^ 10) * (x ^ 10)
//x ^ 1001 = (x ^ 100) * (x ^ 100) * x

fn group_pow<G: FactorGroup + Clone>(x: &BigUint, pt : &G, data: &(G::Data), n: &BigInt)
 -> Result<G, BigInt> { 
  // uses repeated exponentiation
  let mut acc = pt.clone(); 
  let x_bits = x.bits(); 
  dbg!(x_bits);
  for bit_index in (0..x_bits-1).rev() {
    // dbg!(&bit_index, x.bit(bit_index));
    // dbg!(&acc);
    acc = acc.double(data, n)?;
    if x.bit(bit_index) {
      acc = acc.compose(&pt, data, n)?;
    }
  }
  Ok(acc)
}

trait EllipticCurve : Sized { 
  type Point : FactorGroup;
  fn create<R: Rng>(rng : &mut R, n: &BigInt) -> (Self, Self::Point);   
}

fn gcd_path(a: &BigInt, b: &BigInt) -> (BigInt, Vec<(BigInt, BigInt, BigInt, BigInt)>) {
  // assumes a < b
  assert!(a < b);
  let mut path = vec![];
  let mut larger = b.clone(); 
  let mut smaller = a.clone();
  while !Zero::is_zero(&smaller){
    let q = &larger / &smaller; 
    let r = &larger - &q * &smaller; 
    path.push((smaller.clone(), larger.clone(), q.clone(), r.clone()));
    larger = smaller;
    smaller = r;
  }
  (larger, path)
}

fn bezout(path: Vec<(BigInt, BigInt, BigInt, BigInt)>) -> (BigInt, BigInt) {
  // given a vector of (a, b, q, r) calculates (x, y) st ax + by = d 
  let mut x = Zero::zero(); 
  let mut y = One::one(); 
  for (a, b, q, r) in path.into_iter().rev() {
    (x, y) = (y - &x * q, x);
  }
  (x, y)
}

fn invert_mod(x: &BigInt, n: &BigInt) -> Result<BigInt, BigInt> {
  /* returns Ok(x^-1) or Err(d) st d|n */
  // dbg!(&x, &n);
  assert!(x > &Zero::zero() && x < n);
  let (gcd, path) = gcd_path(x, n); 
  if !One::is_one(&gcd) {return Err(gcd)};
  let (b_x, b_y) = bezout(path);
  return Ok(b_x);
}

fn pos_mod(x: BigInt, n: &BigInt) -> BigInt {
  let m = x % n; 
  if m >= Zero::zero() {
    return m
  } else {
    return (m + n) % n
  }
}

// y^2 = x^3 + ax + b (mod n)
// b = y^2 - x^3 - ax
#[derive(Debug, Clone)]
struct WeierstrassCurve{a: BigInt, b: BigInt} 
#[derive(Debug, Clone, PartialEq, Eq)]
enum WeierstrassPoint{
  Identity,
  Affine(BigInt, BigInt)
} 

use WeierstrassPoint::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_chacha::ChaCha8Rng;

impl FactorGroup for WeierstrassPoint {
  type Data = WeierstrassCurve;
  fn identity(curve: &WeierstrassCurve) -> Self {
    Identity
  }

  fn invert(&self, data: &Self::Data, n: &BigInt) -> Result<Self, BigInt> {
    match self {
      Identity => Ok(Identity),
      Affine(x, y) => {
        let neg_y = -1 * y;
        Ok(Affine(x.clone(), neg_y))
      }
    }    
  }

  fn compose(&self, rhs: &Self, WeierstrassCurve { a, b }: &Self::Data, n: &BigInt) -> Result<Self, BigInt> {
    /*
    formula 1
    rx = lam^2 - px - qx 
    nu = py - lam * px 
    ry = lam * rx + nu = lam * rx + py - lam * px = lam * (rx - px) + py 
    cases: 
    px != qx => lam = (qy - py) / (qx - px) & formula 1 
    px == qx and py == -qy => 0
    P = Q and py != -qy => lam = (3 * px ^ 2 + A) / 2 * py & formula 1
    */

    fn finish(lam : BigInt, px: &BigInt, py: &BigInt, qx : &BigInt, n: &BigInt) -> WeierstrassPoint{
      let rx = (&lam).pow(2) - px - qx;
      let ry = lam * (&rx - px) + py;
      Affine(pos_mod(rx,n), pos_mod(-1 * ry, n))
    }
    match (self, rhs) {
      (Identity, rhs) => Ok(rhs.clone()), 
      (lhs, Identity) => Ok(lhs.clone()), 
      (Affine(px, py), Affine(qx, qy)) => {
        if px == qx {
          if *py == -1 * qy {return Ok(Identity)} 
          else {
            let lam = (3 * px.pow(2) + a) * invert_mod(&pos_mod(2 * py, n), n)?;
            return Ok(finish(lam, px, py, qx, n));
          }
        } else {
          let lam = (qy - py) * invert_mod(&pos_mod(qx - px, n), n)?; 
          return Ok(finish(lam, px, py, qx, n));
        }
      }
    }
  }

  fn double(&self, data: &Self::Data, n: &BigInt) -> Result<Self, BigInt> {
    //TODO: make faster?
    self.compose(self, data, n)
  }
  
}

impl EllipticCurve for WeierstrassCurve {
  type Point = WeierstrassPoint;

  fn create<R: Rng>(rng : &mut R, n: &BigInt) -> (Self, Self::Point) {
    let a = rng.gen_bigint_range(&(2i8.into()), n);
    let px = rng.gen_bigint_range(&(2i8.into()), n);
    let py = rng.gen_bigint_range(&(2i8.into()), n);
    let b = (&py).pow(2) - (&px).pow(3) - &a * &px;
    (WeierstrassCurve{a, b: pos_mod(b, n)}, Affine(px, py))
  }
}

fn factorial(x: u64) -> BigUint {
  //use splitting in half strategy
  fn factorial_range(lower: u64, upper: u64) -> BigUint {
    //lower and upper are both inclusive
    if upper - lower < 10 {
      let mut out = One::one();
      for n in (lower..=upper) {
        out *= n; 
      }
      out
    } else {
      let mid = (upper + lower) / 2;
      factorial_range(lower, mid) * factorial_range(mid+1, upper)
    }
  }
  factorial_range(1, x)
}

fn lenstra_factorization<G, C> (n: &BigInt, b1: u64, curves: u32, seed: u64)
 -> Option<BigInt> 
 where G: FactorGroup<Data = C>, C: EllipticCurve<Point = G>
{
  dbg!(b1, curves);
  // returns a factor if found
  let mut rng : ChaCha8Rng = SeedableRng::seed_from_u64(seed);
  //TODO factorial is not optimal 
  let b1_pow = factorial(b1);
  for _ in (0.. curves) {
    let (curve, point) : (G::Data, _) = EllipticCurve::create(&mut rng, n);
    //phase 1
    let b1_point = match group_pow(&b1_pow, &point, &curve, n) {
      Err(factor) => return Some(factor), 
      Ok(pt) => pt,
    };
    //todo implement phase 2
  }
  return None
}

fn factorize_with_ecm<G, C> (n: &BigInt, 
  smallest_factor_digits: u32, largest_factor_digits: u32, 
  rng: &mut ChaCha8Rng, verbose: bool) -> Option<BigInt> 
  where G: FactorGroup<Data = C>, C: EllipticCurve<Point = G>
  {
    //drives lenstra_factorization via slowly increasing b1 and curves until a 
    //factor is found
    if verbose {
      println!("begin-ing to factor {} using ECM", n);
    }
    let goal_factor_digits = smallest_factor_digits; 
    for goal_factor_digits in 
      (smallest_factor_digits..=largest_factor_digits).step_by(5) 
    {
      //magic numbers determined from GMP-ECM
      let b1 : f64 = 3.7f64.pow(goal_factor_digits.to_f64().unwrap().pow(0.666));
      let curves : f64 = 1.6f64.pow(goal_factor_digits.to_f64().unwrap().pow(0.75));
      if verbose {
        println!("looking for {} digits with b1={} and curves={}", goal_factor_digits, b1, curves);
      }
      match lenstra_factorization::<G, C>(n, b1.to_u64().unwrap(), curves.to_u32().unwrap(), rng) {
        Some(factor) => return Some(factor),
        None => (),
      }
    }
    return None
}

fn random_prime_of_size(digits : u32, rng: &mut ChaCha8Rng) -> BigUint {
  let ten : BigUint = 10u8.into();
  let lower_bound = ten.pow(digits-1); 
  let mut candidate = lower_bound.clone();
  //
  
  while !is_prime(&candidate) {
    candidate = rng.gen_biguint_range(&lower_bound, &(&lower_bound * 10u8));
  }
  return candidate
}

fn benchmark(prime_digits: Vec<u32>, total_digits: u32, rng: &mut ChaCha8Rng) {
 let prime_pairs = prime_digits.into_iter()
  .map(|d|(random_prime_of_size(d, rng), random_prime_of_size(total_digits - d, rng))); 
  todo!()
}
/*
   todo: 
   2 primality tests
     miller rabin (check)
     lucas
   sqrt testing
   large prime power testing
   ecm factorization 
    ellipic curve multiply
    other stuff
    phase 2 

  make tests not slow
 */
fn main() {
    let args: Vec<String> = env::args().collect();
    dbg!(&args);
    let seed : u64 = args[1].parse().unwrap();
    let num_str = b"9704908642083262698153783974486202587110431138914168939859964858523789532878233926566020661650899734176571735356880422957189795452451191552634871734668348427870800259380139205278118494937289129593448654877556908494135291096530547109539598669599059827";
    let num = BigUint::parse_bytes(num_str, 10).unwrap();
    println!("{:?}", &num);
    let iter_limit = 10u64.checked_pow(10).unwrap();
    dbg!(&iter_limit);
    let ans = basic_factorize(num.clone().into(), iter_limit, seed);
    let ecm_fac = lenstra_factorization::<WeierstrassPoint, WeierstrassCurve>(&(num.into()), 50, 50, 50);
    println!("ans was: {:?}", ans);
}

mod test {
  use num_integer::gcd;
use rand::random;

  use super::*;

  #[test]
  fn trial_divide_finds_divisor() {
    assert_eq!(trial_divide_u32(41*43, 100), Some(41));
  }

  #[test]
  fn trial_divide_no_divisor_of_prime() {
    assert_eq!(trial_divide_u32(35695349, 1000000), None);
  }

  #[test]
  fn rho_finds_divisor() {
    assert_eq!(rho_u32(41*43, 100, 2), Some(43));
  }

  #[test]
  fn rho_finds_divisor_larger_semiprime() {
    assert_eq!(rho_u32(79273*17393, 10000, 2), Some(17393));
  }

  #[test]
  fn rho_finds_no_divisor_of_prime() {
    assert_eq!(rho_u32(104743, 10000, 2), None)
  }

  #[test]
  fn factorizes_small_number() {
    let mut factors = basic_factorize((7*41*47*199).into(), 1000, 2);
    factors.sort();
    let i32_factors : Vec<i32> = factors.into_iter().map(|i|i.try_into().unwrap()).collect();
    assert_eq!(i32_factors, vec![7,41,47,199])
  }

  #[test]
  fn factorizes_larger_number() {
    let mut factors = basic_factorize((7i64*15485867i64*15650309i64).into(), 100000, 2);
    factors.sort();
    let i32_factors : Vec<i64> = factors.into_iter().map(|i|i.try_into().unwrap()).collect();
    assert_eq!(i32_factors, vec![7, 15485867, 15650309])
  }


  //todo: benchmark that factors a series of larger primes 
  #[test]
  fn lenstra_test() {
    let prime_pairs = [(15485867i64,15650309i64)];  //[(47,199), (17393,79273), (15485867i64,15650309i64)]; 
    fn lenstra_wrapper(p: i64, q: i64) {
      let p : BigInt = p.into();
      let q : BigInt = q.into();
      let n = &p * &q; 
      match lenstra_factorization::<WeierstrassPoint, WeierstrassCurve>
        (&n, 500, 30, 1234567) 
      {
        None => panic!("no factor of {} found", n),
        Some(fac) => {
          assert!(Zero::is_zero(&(&n % &fac)));
          let other_fac = &n / &fac; 
          if fac < other_fac {
            assert_eq!((fac, other_fac), (p, q))
          } else {
            assert_eq!((other_fac, fac), (p, q))
          }
        }
      }
    }

    for (p, q) in prime_pairs {
      lenstra_wrapper(p, q);
    }
  }

  #[test] 
  fn miller_rabin_test() {
    let primes : [u32; 7] = [41, 277, 30869, 1093889, 1992769, 3450749, 4256233];
    let not_primes : [u32; 7] = [95, 3285, 34341, 66623, 67841, 330501, 974689];
    for p in primes {
      assert!(miller_rabin(&p.into(), 25), "{}", p);
    }
    for np in not_primes {
      assert!(!miller_rabin(&np.into(), 25), "{}", np);
    }
  }

  #[test]
  fn is_prime_test() {
    let primes : [u32; 8] = [2, 41, 277, 30869, 1093889, 1992769, 3450749, 4256233];
    let not_primes : [u32; 10] = [95, 100, 3285, 34341, 66623, 12222, 67841, 330501, 974689, 56432932];
    for p in primes {
      assert!(is_prime(&p.into()), "{}", p);
    }
    for np in not_primes {
      assert!(!is_prime(&np.into()), "was prime: {}", np);
    }

  }

  #[test]
  fn gcd_test() {
    fn gcd_u64(a: i64, b : i64) -> i64 {
      gcd_path(&a.into(), &b.into()).0.try_into().unwrap()
    }
    assert_eq!(gcd_u64(8, 64), 8);
    assert_eq!(gcd_u64(8, 65), 1);
    assert_eq!(gcd_u64(128, 130), 2);
  }

  #[test] 
  fn bezout_test() {
    let triples = [(8, 64, 8), (8, 65, 1), (128, 130, 2), (65, 138, 1)]
      .map(|(a, b, d)|(a.into(), b.into(), d.into()));
    for (a, b, d) in triples {
      let (gcd, path) = gcd_path(&a, &b);
      assert_eq!(d, gcd); 
      let (x, y) = bezout(path);
      //dbg!((&x, &y, &a, &b, &d));
      assert_eq!(x * a + y * b, d);
    }
  }

  #[test]
  fn factorial_test() {
    fn slow_factorial(x: u64) -> BigUint {
      let mut acc = One::one();
      for i in (1..=x) {
        acc *= i
      }
      acc
    }
    let vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 50, 100, 103];
    for val in vals {
      assert_eq!(factorial(val), slow_factorial(val));
    }
  }

  #[test] 
  fn pos_mod_test() {
    let triples = [(-3, 10, 7), (117, 11, 7), (63, 3, 0)]
      .map(|(x, n, ans)|(x.into(), n.into(), ans.into()));
    for (x, n, ans) in triples {
      assert_eq!(pos_mod(x, &n), ans);
    }
  }

  #[test] 
  fn group_pow_test() {
    fn slow_group_pow<G: FactorGroup + Clone>(x: &BigUint, pt : &G, data: &(G::Data), n: &BigInt)
     -> Result<G, BigInt> { 
      let mut acc = G::identity(data);
      for i in (0..x.try_into().unwrap()) {
        acc = acc.compose(pt, data, n)?;
      }
      Ok(acc)
     }
     //todo property test
     let n = (79273*17393).into();
     let (curve, point) = WeierstrassCurve::create(&mut rand::thread_rng(),&n);
     let vals : [u32; 13] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 50, 100, 103];
     for val in vals {
       assert_eq!(group_pow(&val.into(), &point, &curve, &n),
        slow_group_pow(&val.into(), &point, &curve, &n))
     }

  }

  #[test]
  fn group_associative_test() {
    let n = (41*59).into();
    let (curve, point) = WeierstrassCurve::create(&mut rand::thread_rng(),&n);
    dbg!(&curve, &point);
    let two_point = point.compose(&point, &curve, &n).unwrap();
    let three_point = two_point.compose(&point, &curve, &n).unwrap();
    // four_point = (((x * x) * x) * x)
    let four_point = three_point.compose(&point, &curve, &n).unwrap();
    // two_two_point = ((x * x) * (x * x))
    let two_two_point = two_point.compose(&two_point, &curve, &n).unwrap();
    assert_eq!(four_point, two_two_point);
  }

  #[test]
  fn weier_associative_repro() {
    /*[src/main.rs:501] &curve = WeierstrassCurve {
    a: 2078,
    b: 648,
    }
    [src/main.rs:501] &point = Affine(
        2131,
        1105,
    ) */
    let n = (41*59).into();
    let curve = WeierstrassCurve{a: 2078.into(), b: 648.into()};
    let point = Affine(2131.into(),1105.into());
    let two_point = point.compose(&point, &curve, &n).unwrap();
    let three_point = two_point.compose(&point, &curve, &n).unwrap();
    let four_point = three_point.compose(&point, &curve, &n).unwrap();
    let two_two_point = two_point.compose(&two_point, &curve, &n).unwrap();
    assert_eq!(four_point, two_two_point);

  }
  //todo: test Weierstrass satisfies group laws

  #[test]
  fn test_random_prime() {
    let p = random_prime_of_size(2, 1234);
    assert_eq!(p, 59u8.into())
  }

  #[test]
  fn test_factorize_with_ecm() {
    let seed = 1234;
    let mut rng : ChaCha8Rng = SeedableRng::seed_from_u64(seed);
    let p = random_prime_of_size(8, &mut rng); 
    dbg!(&p);
    let q = random_prime_of_size(30, &mut rng);
    let ans = factorize_with_ecm::<WeierstrassPoint, WeierstrassCurve>
      (&((&p*q).to_bigint().unwrap()), 5, 15, &mut rng, false);
    assert_eq!(ans.unwrap(), p.to_bigint().unwrap()); 
  }
}