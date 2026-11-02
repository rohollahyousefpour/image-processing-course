"""Make the uint8 overflow trap visible (Week 1)."""
import numpy as np

a = np.uint8(200)
b = np.uint8(100)
print("200 + 100 as uint8 =", a + b, "(wrapped! 300 mod 256 = 44)")
print("as int             =", int(a) + int(b))
