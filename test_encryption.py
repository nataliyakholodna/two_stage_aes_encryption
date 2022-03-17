from utils import bytes_to_bits, pad, encrypt, decrypt

# ---------------------------------------------- Test Helper functions ----------------------------------------------

password = 'hello'
text = pad('text text text text text')

cipher_text = encrypt(text, password)

encrypted = decrypt(cipher_text, password)

print('Padded original text:')
print(text)

print('\nOriginal text in bytes:')
print(text.encode('cp437'))

print('\nInteger values of original text symbols:')
print([b for b in text.encode('cp437')])
print([ord(i) for i in text])

print('\nInteger values of encrypted text symbols:')
print([b for b in encrypted.encode('cp437')])

print('\nInteger values of cipher text symbols:')
print([b for b in cipher_text])

print('\nEncoded text in bytes:')
print(cipher_text)

print('\nEncoded text:')
print(cipher_text.decode('cp437', "ignore"))

print('\nDecoded text:')
print(encrypted)