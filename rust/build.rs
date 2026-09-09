use std::{env, fs, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=Cargo.lock");
    let rustc = Command::new(env::var("RUSTC").expect("RUSTC"))
        .arg("--version")
        .output()
        .expect("read compiler version");
    assert!(rustc.status.success());
    println!(
        "cargo:rustc-env=BUILD_RUSTC={}",
        String::from_utf8(rustc.stdout).unwrap().trim()
    );
    println!(
        "cargo:rustc-env=BUILD_PROFILE={}",
        env::var("PROFILE").unwrap()
    );
    println!(
        "cargo:rustc-env=BUILD_TARGET={}",
        env::var("TARGET").unwrap()
    );
    let lock = fs::read_to_string("Cargo.lock").expect("read dependency lock");
    let source = lock
        .lines()
        .find(|line| line.starts_with("source = ") && line.contains("text-processing-rs"))
        .expect("locked upstream revision");
    println!(
        "cargo:rustc-env=BUILD_UPSTREAM={}",
        source.trim_start_matches("source = ").trim_matches('"')
    );
}
