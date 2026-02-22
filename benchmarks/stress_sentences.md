# Tokenizer Stress Test — 100 Sentences

## Basic
1. Hello
2. The quick brown fox jumps over the lazy dog.
3. I am a student.
4. She sells seashells by the seashore.
5. To be or not to be, that is the question.

## Short / Single Word
6. OK
7. a
8. I
9. Go!
10. No.

## Numbers and Math
11. 42 + 7 = 49
12. 3.14159265358979
13. The temperature was -40 degrees.
14. 1,000,000 dollars
15. 2^10 = 1024 and 2^20 = 1048576

## Punctuation Heavy
16. Wait... really?! No way!!!
17. "Hello," she said, "how are you?"
18. (a) first; (b) second; (c) third.
19. Mr. Dr. Prof. St. Ave. Blvd.
20. :-) ;-) :-( :-O :P <3

## Contractions and Apostrophes
21. I can't believe it's not butter!
22. They're going to their house over there.
23. We've been told that he'd already left.
24. It's a dog's life, isn't it?
25. Who's on first? What's on second? I don't know's on third.

## Accented Characters
26. Café crème brûlée
27. El niño está en la piñata.
28. Über die Straße gehen.
29. Résumé naïve façade coöperate
30. Björk sång en låt.

## Mixed Case
31. iPhone iPad macOS iOS
32. McDonald's KFC NASA UNESCO
33. camelCase PascalCase snake_case kebab-case
34. XML JSON HTML CSS HTTP HTTPS
35. LaTeX BibTeX PostgreSQL MySQL SQLite

## Hyphenated and Compound Words
36. state-of-the-art technology
37. well-known anti-inflammatory self-driving
38. mother-in-law brother-in-law sister-in-law
39. up-to-date out-of-pocket runner-up
40. re-enter pre-existing co-author non-trivial

## Repeated Characters
41. Nooooooo!
42. Hahahahahaha
43. ssssssssssnake
44. Booooooring
45. aaaaaaaaaa bbbbbbbbbb cccccccccc

## Long Sentences
46. The mitochondria is the powerhouse of the cell, and this is one of the most commonly cited facts in all of biology education across the entire world.
47. In order to understand recursion, one must first understand recursion, which requires understanding recursion, which in turn requires understanding recursion.
48. Supercalifragilisticexpialidocious is a word that was popularized by the 1964 Disney musical film Mary Poppins.
49. The longest word in the English language that does not repeat a letter is uncopyrightable.
50. Antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis floccinaucinihilipilification.

## Technical / Code-like
51. fn main() { println!("Hello, world!"); }
52. SELECT * FROM users WHERE id = 42;
53. git commit -m "fix: resolve tokenizer bug"
54. https://example.com/path?query=value&key=123#anchor
55. user@example.com sent an email to admin@test.org

## Special Characters and Symbols
56. Price: $19.99 or 17.49 GBP
57. 100% of 50% is 50%
58. H2O is water. CO2 is carbon dioxide.
59. The ratio is 3:1 or maybe 4:1.
60. Use * for multiplication and / for division.

## Multilingual Fragments
61. Bonjour le monde
62. Hola mundo
63. Guten Tag
64. Konnichiwa
65. Namaste dost

## Whitespace Edge Cases
66. word   word   word
67. 	tab	separated	values
68. trailing spaces
69.    leading spaces
70. mixed   	 	  whitespace

## Single Characters and Short Tokens
71. A B C D E F G
72. x y z
73. 0 1 2 3 4 5 6 7 8 9
74. ! @ # $ % ^ & * ( )
75. . , ; : ? /

## Possessives and Plurals
76. The children's toys were scattered.
77. James's car and the Joneses' house.
78. The dogs' owner called the cats' vet.
79. It's the workers' compensation.
80. Charles's and Diana's wedding.

## Scientific and Technical Text
81. The DNA sequence ATCGATCG contains 8 nucleotides.
82. Einstein's equation E=mc^2 changed physics forever.
83. The pH level was 7.4, indicating a neutral solution.
84. Latency: 12.5ms, throughput: 1.2GB/s, IOPS: 50000
85. TCP/IP, UDP, HTTP/2, gRPC, WebSocket

## Quotes and Nested Punctuation
86. He said, "She said, 'Hello!'"
87. The "so-called" experts disagreed.
88. Title: "War and Peace" by Leo Tolstoy
89. (see [1], [2], and [3] for details)
90. {key: "value", count: 42, flag: true}

## Edge Case Words
91. I'd've thought you'd've known.
92. Rock 'n' roll ain't noise pollution.
93. The # symbol is called an octothorpe.
94. C++ is not the same as C# or C.
95. AT&T vs T-Mobile vs Verizon

## Stress Patterns
96. a aa aaa aaaa aaaaa
97. The the the the the the the the.
98. .,.,.,.,.,.,.,.,
99. 123abc456def789ghi
100. One fish, two fish, red fish, blue fish.
