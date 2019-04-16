def say_hello_to(name):
    print("Hello %s!" % name)

cdef long long cal_test(int n):

    cdef long long s = 0
    cdef int i, j

    for i in range(n):
        for j in range(n):
            s += i * j
    
    return s

def test(n):
    return cal_test(n)