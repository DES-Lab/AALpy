# Theory of VPA
A VPA (Visible Pushdown Automata) is similar to a regular Pushdown Automata with the difference of a split-up alphabet. 

The alphabet of a VPA is a triple with:
- &Sigma;<sub>call</sub>: call set (set of all call letters --> letters that are used for push actions on the automata)
- &Sigma;<sub>ret</sub>: return set (set of all return letters --> letters that are used for pop actions on the automata)
- &Sigma;<sub>int</sub>: internal set (set of all internal letters --> letters that are used for internal transitions that don't alter the stack)

The unification of all three sets is the alphabet.  
The symmetric difference is the empty set.

### The call/return balance
The call/return balance is a function &beta;. The function maps a word of the language to an integer based on the composition of call and return letters. Here is how it works: 
- Call letters add "1" to the balance
- Return letters subtract "1" from the balance
- Internal letters have no impact on the balance

#### Example:  
Imagine a language with:
- &Sigma;<sub>call</sub> = {a, b} 
- &Sigma;<sub>ret</sub> = {c, d}
- &Sigma;<sub>int</sub> = {e, f}

We would have the following balances for the following words:
- aabbccdd = 0 (1+1+1+1-1-1-1-1)
- abab = 4 (1+1+1+1)
- cccc = -4 (-1-1-1-1)
- cdeabef = 0 (-1-1+0+1+1+0+0)

### Call-matched, return-matched, well-matched
By defining the call/return balance we can introduce the definition of call-matched, return-matched and well-matched words. The definition is as follows:
- **Return-matched** words have a &beta; >= 0 (They have more call-letters and therefore the balance is greater-equal than 0)  
The set of return matched words is called MR(&Sigma;)
- **Call-matched** words have a &beta; =< 0 (They have more return-letters and therefore the balance is greater-equal than 0)  
The set of call matched words is called MC(&Sigma;)
- **Well-matched** words have a &beta; = 0 (The composition of call and return letters in well-matched words equalizes to zero)  
The set of well matched words is called MW(&Sigma;)

### Context pairs (CP)
Furthermore we want to introduce the context pairs, the set of context pairs is CP(&Sigma;). Context pairs are well matched words of the form u*v.  
- u has the form of MR(&Sigma;) * &Sigma;<sub>call</sub> or is the empty word &epsilon;  
This means if u is of the form MR(&Sigma;) * &Sigma;<sub>call</sub> it has at least a &beta; >= 1
- v is of the form MC(&Sigma;)
- &beta;(u) = -&beta;(v).
- v is the matching word for u to be u*v &isin; WM(&Sigma;)
#### Example:  
If we get back at our last language with the following split:
- &Sigma;<sub>call</sub> = {a, b} 
- &Sigma;<sub>ret</sub> = {c, d}
- &Sigma;<sub>int</sub> = {e, f}

We have could have this types of context pairs:
- u = aa | v = cc
- u = aca | v = d