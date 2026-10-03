### 4. Cryptanalysis
• Unit-II Conventional and Symmetric Cryptography 2.1 Introduction: 1. Plain text 2. cipher text 3. cryptanalysis 4. cryptanalysis 5. encryption 6. Decryption

### Symmetric key cryptography/Encryption
• Unit-II Conventional and Symmetric Cryptography
2.1 Introduction:
1.
• Symmetric key cryptography/Encryption

### 3. Cryptography or Cryptology
• Cryptography or Cryptology
-  Cryptography is a Greek word having the meaning of “Secret Writing”.

### 2. Cipher text
• Cipher text
-  Cipher text is encrypted text transformed from plaintext using an encryption
algorithm.
• For Encryption, C=E(P)=(P+K) mod 26
For Decryption, P=D(C)=(C-k)mod 26
Example:
Plain Text=HELLO      AND
Key=3
Encryption:
Cipher Text=KHOOR
Decryption:
Plain Text=HELLO
• There are various types of substitution ciphers which are as follows:
1.
• Means find CipherText letter in the row of key.
• Fill in the Encrypted Message: - Fill the cipher-text row wise.
• DES is an implementation of a Feistel Cipher.
• Cipher Text will be “QUIJH”
-
Example -2:
-
Plaintext : HOW ARE YOU
Onetime pad(Key) : NCB TZQ ARX
[For Decryption: Do (C.T.
• Plaintext is the input to a crypto system, with cipher text being the output. — In cryptography, algorithms transform plaintext into cipher text, and cipher text into

### 6. Decryption
• Keys and ciphers are the key to encrypting and decryption.

### 2.2 Substitution and Transposition Techniques
• Substitution and Transposition Techniques
-  Substitution Techniques
-  Substitution technique involves replacing letters with other letters and symbols.

### Playfair Cipher
• Playfair Cipher - Rules to make playfair matrix of KEY.
• Playfair Cipher
-  Rules to make playfair matrix of KEY.

### Playfair matrix (key matrix)
• Example:
Playfair matrix (key matrix):
Key=MONARCHY
• Steps: 1. Write a matrix. 2. Write a matrix. 3. Write a matrix. 4. Write a matrix. 5. Write a matrix. 6. Write a matrix. 7. Write a matrix. 8. Write a matrix. 9. Write a matrix. 10. Write a matrix.

### Example
• If a pair is a repeated letter, insert filler like 'X’
Example: TEXT→ TE  XT
HELLO →HE  LL O →HE LX LO
3.
• For example, the hidden message may be in invisible ink between the visible lines of a
private letter.
• Example: RSA algorithm, DSA

### 2. Polyalphabetic Cipher
• Vigenere Cipher - It is a polyalphabetic cipher. 2. Repeat your key until the length of plaintext. Find corresponding row and column from Vigenere Table. 3.
• Polyalphabetic Cipher
• It is a polyalphabetic cipher. — For Vigenere Cipher, use Vigenere Table.

### Key characteristics
• To encrypt a message using the Vigenere Cipher you first need to choose a key.
• Key characteristics: - Key must be as long as the plaintext, and not repeating - Key must be as long as the plaintext, and not repeating - Key must be used once. - Key must be as long as the plaintext, and not repeating - Key must be used once. - There should be two copies of the key: One for sender and other for receiver. - It is also known as One Time Pad (OTP). - It involves rearranging of letters in plain
• Detecting the pixel's color byte is a key step in the development of a symmetric-key block cipher.
• DES offers a lower level of security due to its 56-bit key, which can be feasibly broken by a brute-force attack.
• DES offers a lower level of security due to its 56-bit key, which can be feasibly broken
by a brute-force attack.
• The DES function f applies a 48-bit key to the rightmost 32 bits to produce a 32-bit output.
• Cryptography/Secret Key /Private Key
• It is also known as Public key cryptography.

### VIGNER CIPHER/ polyalphabetic cipher
• VIGNER CIPHER/ polyalphabetic cipher
-  It is a polyalphabetic cipher.

### 1. Plain text
• Vernam cipher is a stream cipher where the plain text is added with a random stream of data of the same length to generate the encrypted data.
• Completeness − Each bit of ciphertext depends on many bits of plaintext.
• Plain Text: - INFORMATION SECURITY
Number of Rails/Key: - 3
i.
• Avalanche effect − A small change in plaintext results in the very great change in the
• Means find CipherText letter in the row of key. And then consider corresponding — letter of column header for getting plain text back.
• Split the message (Plain Text) in pair of characters.

### 5. Encryption
• Plaintext= secret
keyword =bcd
ENCRYPTION:
-  First we must generate the keystream, by repeating the letters of the key until it is the
same length as the plaintext.
• Though, key length is 64-bit, DES has an effective key length of 56 bits, since 8 of the 64
bits of the key are not used by the encryption algorithm (function as check bits only).
• It is the breaking of “Secret Codes”. — It is the science of breaking Encryption.

### Examples of Steganography
• Examples of Steganography: 1)Character Marking

### Types of Steganography
• Image Steganography - A comprehensive breakdown of the types of steganography based on the carrier medium used.

### Methods of Steganography
• Invisible Ink
3)Pin punctures
-  Methods of Steganography
-
Digital Steganography:
We can insert date or we can hide data in the image by replacing bits of image.
• — Syntactic/Semantic Methods: Uses specific grammar structures or replaces words with

### Most common technique are
• Most common technique are:
1)LSB,
2)DCT and
3)Append type.

### Core Findings & Technical Analysis
• LSB: Least Significant Bit
-Replace LSB bit with a bit from hidden data.
• Least Significant Bit (LSB) Insertion: Alters the last bit of a pixel's color byte (e.g.,
changing a color value from 255 to 254).
• — Header Manipulation: Injects secret data bits into unused, optional, or flexible fields
• LSB  has smallest effect on the amount of color. — Replacing LSB to hidden data will have small effect on the picture.

### 3. Video Steganography
• Motion Vector Embedding: Hides data inside the motion vectors used during video
compression (e.g., MPEG, H.264) to predict pixel movement between frames.
• Video Steganography
• Video steganography treats a video as a continuous stream of images (frames) and audio. It

### DES Structure
• It uses 16 round Feistel structure.
• General Structure of DES is shown in figure:

### Initial and Final Permutation
• Since DES is based on the Feistel Cipher, all that is required to specify DES is −
-  Round function
-  Key schedule
-  Any additional processing − Initial and final permutation
Initial and Final Permutation
-  The initial and final permutations are straight Permutation boxes (P-boxes) that are
inverses of each other.

### Key Generation
• Key Generation
The round-key generator creates sixteen 48-bit keys out of a 56-bit cipher key.

### Advantage of DES
• Advantage of DES
-
DES uses the symmetric-key algorithm, thus, it is possible to perform encryption and
decryption by a single key with the same algorithm.

### Disadvantage of DES
• Disadvantage of DES
-
Because DES uses a smaller key, it is less secure.

### Data Encryption Standard (DES)
• Symmetric Cryptography: Data Encryption Standard- Structure, Advantages and
Disadvantages
-  Data Encryption Standard (DES):
The Data Encryption Standard (DES) is a symmetric-key block cipher published by the
National Institute of Standards and Technology (NIST).
• DES Structure — The Data Encryption Standard (DES) is a block cipher.

### Asymmetric key cryptography/Encryption
• Asymmetric key cryptography/Encryption

### Caesar Cipher
• would include the Caesar-shift cipher, where each letter is shifted based on a numeric key.
• Caesar Cipher — It is also known as shift cipher or additive cipher.

### Hill Cipher
• Hill Cipher — Hill cipher is a polygraphic substitution cipher based on linear algebra.

### 1. Monoalphabetic Cipher
• Monoalphabetic Cipher

### Steps of algorithm
• Steps of algorithm — 1. Make a playfair matrix.

### Rail fence cipher
• Rail fence cipher — (TRANSPOSITION TECHNIQUES)
• Decryption — Determine the Zigzag Pattern: - The number of columns in the rail fence cipher remains

### 2.3 Steganography
• STEGNOGRAPHY — Steganography means hiding a message within another message or image.
• This is the most popular form of steganography due to the massive amount of digital — image sharing online. Data is embedded by subtly altering the digital properties of the
• Physical & Digital Printed Steganography

### 2. Audio Steganography
• Audio Steganography
• — Echo Hiding: Injects a subtle, unnoticeable echo into the audio track. The delay between
• — Phase Coding: Replaces the phase of an initial audio segment with a reference phase that

### 4. Text Steganography
• Text steganography involves altering the formatting, arrangement, or structure of words — within a document. It requires the least amount of memory but offers a very small hiding

### 5. Network (Protocol) Steganography
• Network (Protocol) Steganography
• — Packet Timing: Alters the delay or arrival intervals between consecutive network

### Digital Steganography
• — Digital Watermarking / Printer Dots: Fine, yellow tracking dots printed invisibly by

### EXAMPLE-2
• EXAMPLE-2 — PT V I G E N E R E C I P H E R

### Rules
• Rules to make playfair matrix of KEY. — 1. Arrange in 5*5 matrix.

### 1. Image Steganography
• — Palette Modification: Hides data by altering the color palette list inside indexed images
• — Frame-by-Frame Embedding: Treats individual video frames as static images and
• — Low-Bit Encoding: Similar to LSB in images, this modifies the least significant bits of