Here we give a simple example to run protocols in PETAce.

# SetOps
To run python examples, execute the following in commands in separate terminal sessions.

```bash
python3 ./example/setops/ecdh_psi.py -p 0
python3 ./example/setops/ecdh_psi.py -p 1
```

# SecureNumpy

To run python examples, execute the following in commands in separate terminal sessions.

```bash
python3 ./example/securenumpy/linear_regression.py -p 0
python3 ./example/securenumpy/linear_regression.py -p 1
```

# Bigdata
When using the big data engine, you only need to update the part related to engine initialization, while the rest remains consistent. However, there are some points to note:
- The number of rows in the data cannot be less than the number of partition.
- Some functions, such as reshape, are not supported in big data mode and will result in errors. This is because big data inherently does not support these operations.

To run python examples, execute the following in commands in separate terminal sessions.

```bash
python3 ./example/securenumpy/bigdata.py -p 0
python3 ./example/securenumpy/bigdata.py -p 1
```
