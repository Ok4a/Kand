import ctypes

fun  = ctypes.CDLL("C:/Users/owkaa/OneDrive/Dokumenter/SDU/Kand/python/runCfnInPy/mydll.dll")
fun.square.argtypes = [ctypes.c_int]

for i in "hello World!":
    print(fun.square(i))

#gcc -shared -o mydll.dll .\mydll.c
# .dll to win .so for linux