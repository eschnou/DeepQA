import numpy as np

with open("D:\\Projects\\NLP\\GoogleNews-vectors-negative300.bin", "rb", 0) as f:
                header = f.readline()
                vocab_size, layer1_size = map(int, header.split())
                binary_len = np.dtype('float32').itemsize * layer1_size
                initW = np.random.uniform(-0.25,0.25,(1000, layer1_size))
                vocab = dict()
                for line in range(1000):
                    word = []
                    while True:
                        ch = f.read(1)
                        if ch == b' ':
                            word = b''.join(word)
                            break
                        if ch != b'\n':
                            word.append(ch)
                    vocab[word] = line
                    initW[line] = np.fromstring(f.read(binary_len), dtype='float32')

                print(type(initW)) # numpy.ndarray
                print(initW.shape) # (vocab_size, embedding_dim)
