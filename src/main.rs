#![allow(unused)]
use std::env;

use num_bigint::{BigUint, BigInt, ToBigInt, RandBigInt};
use num_traits::{Zero, One, Signed, Pow};
use num_integer::{Integer};

fn trial_divide(n: &BigUint, limit: BigUint) -> Option<BigUint> {
  let mut divisor: BigUint = 3u8.into();
  let two: BigUint = 2u8.into();

  while &divisor < &limit {
    if n % &divisor == Zero::zero() {
      return Some(divisor)
    }
    divisor += &two;
  }
  return None
}

fn trial_divide_u32(n: u32, limit: u32) -> Option<u32> {
  match trial_divide(&n.into(), limit.into()) {
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
    1000000u32.into()) {
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
  match trial_divide(n, 10000u32.into()) {
    Some(_) => return false, 
    None => (),
  }
  return miller_rabin(n, 25);
  // TODO: ppw test
}


trait FactorGroup : Sized {
  type Data;
  fn identity(data: &Self::Data) -> Self; 
  fn invert(&self, data: &Self::Data, n: &BigInt) -> Result<Self, BigInt>; 
  fn compose(&self, rhs: &Self, data: &Self::Data, n: &BigInt) -> Result<Self, BigInt>; 
  fn double(&self, data: &Self::Data, n: &BigInt) -> Result<Self, BigInt> {
    self.compose(self, data, n)
  }
}

fn group_pow<G: FactorGroup>(x: &BigUint, pt : G, data: &G::Data, n: &BigInt) -> Result<G, BigInt> { 
  // uses repeated exponentiation
  todo!()
}

trait EllipticCurve : Sized { 
  type Point : FactorGroup;
  fn create<R: Rng>(rng : &mut R, n: &BigInt) -> (Self, Self::Point);   
}

fn invert_mod(x: &BigInt, n: &BigInt) -> Result<BigInt, BigInt> {
  /* returns Ok(x^-1) or Err(d) st d|n */
  todo!()
}

// y^2 = x^3 + ax + b (mod n)
#[derive(Debug, Clone)]
struct WeierstrassCurve{a: BigInt, b: BigInt} 
#[derive(Debug, Clone)]
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

    fn finish(lam : BigInt, px: &BigInt, py: &BigInt, qx : &BigInt) -> WeierstrassPoint{
      let rx = (&lam).pow(2) - px - qx;
      let ry = lam * (&rx - px) + py;
      todo!("mod rx and ry by n");
      Affine(rx, ry)
    }
    match (self, rhs) {
      (Identity, rhs) => Ok(rhs.clone()), 
      (lhs, Identity) => Ok(lhs.clone()), 
      (Affine(px, py), Affine(qx, qy)) => {
        if px == qx {
          if *py == -1 * qy {return Ok(Identity)} 
          else {
            let lam = (3 * px.pow(2) + a) * invert_mod(&(2 * py), n)?;
            return Ok(finish(lam, px, py, qx));
          }
        } else {
          let lam = (qy - py) * invert_mod(&(qx - px), n)?; 
          return Ok(finish(lam, px, py, qx));
        }
      }
    }
  }

  fn double(&self, data: &Self::Data, n: &BigInt) -> Result<Self, BigInt> {
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
    (WeierstrassCurve{a, b}, Affine(px, py))
  }
}

fn factorial(x: u64) -> BigUint {
  //use splitting in half strategy
  todo!()
}

fn lenstra_factorization<G: FactorGroup>(n: &BigInt, b1: u64, curves: u32, seed: u64)
 -> Option<BigInt> {
  // returns a factor if found
  let mut rng : ChaCha8Rng = SeedableRng::seed_from_u64(seed);
  let b1_pow = factorial(b1);
  for _ in (0.. curves) {
    let (curve, point) : (G::Data, _) = EllipticCurve::create(&mut rng, n);
    let b1_point = group_pow(&b1_pow, point, &curve: FactorGroup::Data, n);
  }
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
    let ecm_fac = lenstra_factorization::<WeierstrassCurve>(&(num.into()), 50, 50, 50);
    println!("ans was: {:?}", ans);
}

mod test {
    use super::*;

  #[test]
  fn trial_divide_finds_divisor() {
    assert_eq!(trial_divide_u32(41*43, 100), Some(41));
  }

  #[test]
  fn trial_divide_no_divisor_of_prime() {
    assert_eq!(trial_divide_u32(35695349, 10000000), None);
  }

  #[test]
  fn rho_finds_divisor() {
    assert_eq!(rho_u32(41*43, 100, 2), Some(43));
  }

  #[test]
  fn rho_finds_divisor_larger_semiprime() {
    assert_eq!(rho_u32(7927*17393, 10000, 2), Some(7927));
  }

  #[test]
  fn rho_finds_no_divisor_of_prime() {
    assert_eq!(rho_u32(104743, 1000000, 2), None)
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
}