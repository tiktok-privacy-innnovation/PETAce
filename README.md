# PETAce

<!-- start-petace-overview -->

Privacy-Enhancing Technologies via Applied Cryptography Engineering (PETAce) is a framework for privacy-preserving computing. It provides strong privacy guarantee by analytzing and computing cryptographically pseudonymized data without revealing hidden sensitive information. It consists of the following parts:

- The "user interface" layer provides users with high-level programming interfaces for collaborative data analysis (SecureNumpy), joint SQL query (SecureSQL), and privacy-preserving machine learning (SecureML). It also supports large-scale data computing by combining Secure Multi-party Computation and parallel processing systems (such as Spark).

- The "virtual machine" layer is responsible for parsing high-level language into secure multi-party computation (MPC) operators, and performing automatic optimization and scheduling.

- The "protocol" layer includes secure multi-party computation protocols, such as general-purpose two-party secure computation protocols, privacy set intersection, and privacy information retrieval, etc.

- The "primitive" layer consists of standard cryptographic algorithms and protocols, differential privacy mechanisms, and abstract network interfaces, etc.

<figure>
<img src='images/PETAce.png' align="middle"/>
</figure>

PETAce enables fast prototyping of ideas based on privacy-enhancing technologies, and we plan to integrate state-of-the-art research results into the PETAce in future releases. Its core modules are implemented in C++ and are modularized into the following repositories.

- [PETAce-Solo](https://github.com/tiktok-privacy-innovation/PETAce-Solo) implements primitive hashing, encryption, and randomness generation algorithms performed by one party.
    - Hash functions: SHA-256, SHA3-256, and BLAKE2b
    - Psuedo-random number generators based on: SHAKE_128, BLAKE2Xb, and AES_ECB_CTR.
    - Sampling of bytes, 32-bit unsigned integers, and 64-bit unsigned integers from the uniform distribution
    - Prime field elliptic curve group arithmetics including hash-to-curve
    - Hashing tables: Cuckoo hashing and simple hashing
    - Partially homomorphic encryption: the Paillier cryptosystem
- [PETAce-Verse](https://github.com/tiktok-privacy-innovation/PETAce-Verse) includes frequently used cryptographic subprotocols such as oblivious transfer and oblivious shuffling.
    - [Naor-Pinkas OT](https://dl.acm.org/doi/10.5555/365411.365502)
    - [IKNP OT](https://link.springer.com/chapter/10.1007/978-3-540-45146-4_9) with [optimization](https://link.springer.com/article/10.1007/s00145-016-9236-6)
    - [KKRT OT](https://dl.acm.org/doi/abs/10.1145/2976749.2978381)
- [PETAce-Duet](https://github.com/tiktok-privacy-innovation/PETAce-Duet) abstracts general-purpose two-party secure computing operator protocols.
    - Protocols from [ABY](https://www.ndss-symposium.org/wp-content/uploads/2017/09/08_2_1.pdf)
    - Secure comparison protocols from [Cheetah](https://www.usenix.org/system/files/sec22-huang-zhicong.pdf)
    - The secure random shuffling protocol from [Secret-Shared Shuffle](https://link.springer.com/chapter/10.1007/978-3-030-64840-4_12)
    - Protocols that convert arithmetic shares to and from ciphertexts of the Paillier cryptosystem
- [PETAce-SetOps](https://github.com/tiktok-privacy-innovation/PETAce-SetOps) archives several protocols that perform private set operations.
    - An ECDH-PSI protocol based on Elliptic-Curve Diffie-Hellman
    - The [KKRT-PSI](https://dl.acm.org/doi/abs/10.1145/2976749.2978381) protocol based on Oblivious Pseudorandom Functions (OPRF)
    - A private join and compute protocol based on [Circuit-PSI](https://www.researchgate.net/publication/356421123_Circuit-PSI_With_Linear_Complexity_via_Relaxed_Batch_OPPRF)
- [PETAce-Network](https://github.com/tiktok-privacy-innovation/PETAce-Network) provides a preliminary interface of network communication.
    - Network abstract interface
    - Socket network implementation

<!-- end-petace-overview -->

## Requirements

<!-- start-petace-getting-started -->

| System | Toolchain                                             |
|--------|-------------------------------------------------------|
| Linux  | Clang++ (>= 5.0) or GNU G++ (>= 5.5), CMake (>= 3.15) |

| Required dependency                                                            | Tested version | Use                               |
|--------------------------------------------------------------------------------|----------------|-----------------------------------|
| [OpenSSL](https://github.com/openssl/openssl) | 1.1.1          | Cryptographic primitives |
| [GMP](https://gmplib.org)                     | 6.3.0          | Bignumer operations for GMP-based Paillier|
| [Eigen](https://gitlab.com/libeigen/eigen)                                     | 3.4.0          | Matrix and vector templates       |
| [Microsoft SEAL](https://github.com/microsoft/SEAL)                                     | 4.1.0          | Fully homomorphic encryption       |
| [Google Logging](https://github.com/google/glog)                               | 0.4.0          | Logging                              |
| [JSON for Modern C++(JSON)](https://github.com/nlohmann/json)                  | 3.10.1         | PSI parameter configuration          |
| [PETAce-Solo](https://github.com/tiktok-privacy-innovation/PETAce-Solo)       | 0.4.0          | Cryptography primitives           |
| [PETAce-Network](https://github.com/tiktok-privacy-innovation/PETAce-Network) | 0.4.0          | Network communication protocols   |
| [PETAce-Verse](https://github.com/tiktok-privacy-innovation/PETAce-Verse)     | 0.4.0          | Primitive cryptographic protocols |
| [PETAce-SetOps](https://github.com/tiktok-privacy-innovation/PETAce-SetOps)     | 0.4.0          | Set operations MPC protocol |
| [PETAce-Verse](https://github.com/tiktok-privacy-innovation/PETAce-Duet)     | 0.4.0          | MPC protocol |

| Optional dependency                                    | Tested version | Use                    |
|--------------------------------------------------------|----------------|------------------------|
| [Intel Paillier Cryptosystem Library (IPCL)](https://github.com/intel/pailliercryptolib)                                     | 495beaad1f6e70741f2b5cf1279cb919fd66d894| Paillier Encryption       |
| [gRPC](https://github.com/grpc/grpc)                                          | 1.62.1          | Network communication              |
| [CryptoDataEngine](https://github.com/tiktok-privacy-innovation/CryptoDataEngine) | 0.1.0 | Big data parallel support|

## Building PETAce

The commands presented below should be executed in the root directory of PETAce.

First, build [JSON for Modern C++ (JSON)](https://github.com/nlohmann/json) using the following scripts.
Assume that JSON is cloned into the directory `${JSON}`.

```bash
cmake -B ${JSON}/build -S ${JSON}
cmake --build ${JSON}/build -j
```

Then, build PETAce library:

```bash
cmake -S . -B build -Dnlohmann_json_DIR=${JSON}/build -DPETACE_BUILD_PYTHON=ON
cmake --build build
cd build
make wheel
```

Output binaries can be found in `build/python/wheel/` directory.

| Compile Options            | Values        | Default | Description                                         |
|----------------------------|---------------|---------|-----------------------------------------------------|
| `CMAKE_BUILD_TYPE`         | Release/Debug | Release | Debug mode decreases run-time performance.          |

For more compilation options, such as enabling IPCL and network agents, please refer to [PETAce-Solo](https://github.com/tiktok-privacy-innovation/PETAce-Solo) and [PETAce-Network](https://github.com/tiktok-privacy-innovation/PETAce-Network).

For instructions on how to run examples of PETAce, please refer to the instructions in the example directory.

<!-- end-petace-getting-started -->

## Contribution

Please check [Contributing](CONTRIBUTING.md) for more details.

## Code of Conduct

Please check [Code of Conduct](CODE_OF_CONDUCT.md) for more details.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Citing PETAce

To cite PETAce in academic papers, please use the following BibTeX entries.

### Version 0.4.0

```tex
    @misc{petace,
        title = {PETAce (release 0.4.0)},
        howpublished = {\url{https://github.com/tiktok-privacy-innovation/PETAce}},
        month = Jan,
        year = 2025,
        note = {TikTok Pte. Ltd.},
        key = {PETAce}
    }
```
