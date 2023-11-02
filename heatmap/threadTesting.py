import threading

def coder(number):
    print(f'the number we got is {number}')
    value = number+10
    return value

threads = []

for k in range(10):
    t = threading.Thread(target=coder, args=(k,))
    threads.append(t)
    t.start()
print('threads :', threads)

