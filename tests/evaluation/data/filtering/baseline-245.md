[ Skip to main content ](https://www.linkedin.com/pulse/why-quantum-computers-can-break-rsa-lattice-timothy-hollebeek-vi8re#main-content)
For decades, online security has relied on math problems that are easy to do in one direction but extremely hard to undo. The RSA encryption system, for example, is based on multiplying two large prime numbers together. Multiplying is easy, but trying to split the result back into its original factors—factoring—is incredibly slow for normal computers when the numbers are large enough.
However, quantum computers threaten to change this, breaking RSA completely while leaving lattice-based cryptography secure. To understand why, we need to explore how quantum computers work and what makes factoring different from lattice problems.
###  How Quantum Computers Process Information
Normal computers store information in bits, which can be either 01. Quantum computers use qubits, which can be in a mix of 0 and 1 at the same time. This ability is called superposition.
A qubit’s state can be thought of as a point on a sphere. If the qubit is exactly at the north pole, it’s definitely 0. If it’s at the south pole, it’s definitely 1. But most of the time, it’s somewhere in between, meaning it has a probability of being either when measured.
The equator of the sphere represents states where there’s a perfect 50/50 chance of being 0 or 1. Quantum computers can manipulate qubits while they are in these mixed states, allowing them to explore many possibilities at once. And because of entanglement, the individual qubits remain in sync as their values are changed by mathematical operations. However, there’s a catch: the moment we measure a qubit, it “collapses” into a definite value, and all the extra information is lost.
This is why quantum computers can’t just magically try every possible answer at once. Instead, they need to use carefully designed quantum algorithms that guide them toward the right answer before measurement wipes out the information.
###  How Quantum Computers Break RSA
RSA encryption depends on the difficulty of factoring large numbers. Given a number like 851, a normal computer would have to test different pairs of numbers until it finds that 851 = 23 × 37. For small numbers, this is easy, but when the numbers are hundreds of digits long, factoring becomes painfully slow.
However, factoring has a hidden weakness: it has a periodic structure that quantum computers can exploit.
Here’s how it works:
## Recommended by LinkedIn
[ Quantum-Safe Encryption: Assessing Readiness and… ](https://www.linkedin.com/pulse/quantum-safe-encryption-assessing-readiness-charting-path-hans-beers-rv0oe)
[ Quantum Threat Tracker (QTT) Review Praising the Tool… Marin Ivezic  ](https://www.linkedin.com/pulse/quantum-threat-tracker-qtt-review-praising-tool-demo-marin-ivezic-sxnnc)
[ BREAKING THE CODE: HOW QUANTUM COMPUTING COULD SHATTER… Octaloop IT & NS  ](https://www.linkedin.com/pulse/breaking-code-how-quantum-computing-could-shatter-e3gcf)
  1. Instead of trying to directly factor the number, we turn the problem into finding the period of a repeating function related to the number.
  2. This period acts like a fingerprint that reveals the factors of the number.
  3. Shor’s algorithm, a quantum algorithm developed in 1994, is designed to find this period.
  4. The quantum computer sets up a superposition of many possible values and applies a series of operations that amplify the correct period while canceling out incorrect results.
  5. When the quantum computer measures the result, it collapses onto the right period, which can then be used to quickly find the factors.


Normal computers can’t do this efficiently, but quantum computers can find the period exponentially faster, making factoring—and breaking RSA—practical.
###  Why Lattice Cryptography is Safe
Lattice-based cryptography doesn’t rely on factoring numbers. Instead, it’s based on problems that involve high-dimensional grids of points (called lattices), where the challenge is to find the shortest vector between points.
To visualize this, imagine a sprawling 3D grid of dots in space. If you are given a random point, finding the shortest path to the closest dot in the grid is difficult. Now imagine this in 500 or 1000 dimensions—it becomes incredibly hard.
Unlike factoring, these problems don’t have a useful repeating structure for a quantum computer to exploit. While quantum computers are great at detecting periodicity, lattice problems appear random and messy, so quantum tricks like Shor’s algorithm don’t help.
Some quantum algorithms (like Grover’s algorithm) can still speed up brute-force searching slightly, but the improvement isn’t enough to break lattice cryptography. Security researchers can simply increase the size of the lattice to compensate.
###  Conclusion: RSA is Doomed, But Lattices Are Safe
Quantum computers completely change the game for encryption. They turn the factoring problem from “impossibly slow” to “instant” using Shor’s algorithm. Since RSA is based on factoring, it will eventually become obsolete.
Lattice cryptography, however, is built on problems that don’t have an exploitable structure. Quantum computers can’t amplify the correct answer the way they can with factoring. This makes lattice cryptography one of the best candidates for securing data in a post-quantum world.
That’s why governments and tech companies are moving toward lattice-based encryption—it’s one of the few defenses that quantum computers can’t crack.
Insightful


RSA is at risk in the quantum era because quantum algorithms like Shor’s can break it by factoring large numbers efficiently. Lattice-based cryptography is currently our best defense. It’s built on problems like LWE and SVP, which are much harder for both classical and quantum computers. So far, no efficient quantum attacks exist against them. While nothing is future proof, the best path forward is to adopt hybrid cryptographic mechanisms combining classical and post-quantum algorithms to ensure security during the transition. Cryptography must evolve, and so should our strategies. Now is the time for organizations, developers, and security teams to explore, test, and gradually implement quantum safe solutions. The future won’t wait neither should we.
Nice explanation, As there are no known algorithms to solve the lattice based hardness problems, hence the Lattice based cryptosystems are quantum resistant.
Great explanation [Timothy Hollebeek](https://www.linkedin.com/in/timothy-hollebeek-505b814?trk=article-ssr-frontend-pulse_x-social-details_comments-action_comment-text)
To be honest, I have heard this type of reasoning (lack of quantum exploitable structure of lattice based cryptography) a lot, but it is not a mathematically and computationally sound perspective. The general approach in cryptography can be simplified as follows: 1) Find a computationally hard problem (NP-complete or worse), 2) Use it to obtain keys by taking some parts/components of the calculation, 3) Distribute and manage the key cryptomaterial in some sort of data structures and infrastructure (X.509, TSL, CRL ...). In the case of lattice based cryptography, we exchange "Integer Factorization" with the SVP and/or LWE problems. The cracking of a cryptographic scheme means that we are in the domain of approximation algorithms for finding problem solutions. Especially, with SVP and LWE I cannot stop having the feeling that we don’t look at them the right way when applying quantum algorithms ...In addition, the problem hierarchy of quantum classes seems to require a lot more research. Hence, we really cannot make definite statements as of now.
To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=%2Fpulse%2Fwhy-quantum-computers-can-break-rsa-lattice-timothy-hollebeek-vi8re&trk=article-ssr-frontend-pulse_x-social-details_feed-cta-banner-cta)
##  More articles by Timothy Hollebeek 
  * [ The Insanity of Implicit Requirements  ](https://www.linkedin.com/pulse/insanity-implicit-requirements-timothy-hollebeek-d87wf)
###  The Insanity of Implicit Requirements 
The CA/Browser Forum has spent years crafting the Baseline Requirements, a Certificate Profile (CP) that tells… 
  * [ Why You Can’t Copy a Qubit (And Why That Breaks Everything You Expect About Computers)  ](https://www.linkedin.com/pulse/why-you-cant-copy-qubit-breaks-everything-expect-timothy-hollebeek-ijakc)
###  Why You Can’t Copy a Qubit (And Why That Breaks Everything You Expect About Computers) 
Imagine you’re in a top-secret research lab where a quantum computer is running the most advanced calculation ever… 
  * [ Why Electrons Get Entangled  ](https://www.linkedin.com/pulse/why-electrons-get-entangled-timothy-hollebeek-emfpc)
###  Why Electrons Get Entangled 
There is no spoon. There also is no electron. 
  * [ Slightly Less Than Awesome Crypto  ](https://www.linkedin.com/pulse/slightly-less-than-awesome-crypto-timothy-hollebeek)
###  Slightly Less Than Awesome Crypto 
Continuing the trend of short articles on Mondays, I want to talk about innovation in the PKI space, and some of the… 
  * [ Monday Musings  ](https://www.linkedin.com/pulse/monday-musings-timothy-hollebeek)
###  Monday Musings 
Going to try something new here .. 
  * [ ASC X9 is helping the financial services industry prepare for quantum computing  ](https://www.linkedin.com/pulse/asc-x9-helping-financial-services-industry-prepare-timothy-hollebeek)
###  ASC X9 is helping the financial services industry prepare for quantum computing 
Since I have a strong understanding of both quantum mechanics and financial key management, I am going to be heavily… 

Show more 
##  Others also viewed 
  * ### [ Quantum Computing, Encryption Standards, and the Future of Global Cybersecurity  Dr. Tan Kian Hua 陈建桦博士  ](https://www.linkedin.com/pulse/quantum-computing-encryption-standards-future-global-%E9%99%88%E5%BB%BA%E6%A1%A6%E5%8D%9A%E5%A3%AB-di4ac)
  * ### [ NISQ Era Challenges to Modern Cryptography  Matthias Kaiser  ](https://www.linkedin.com/pulse/nisq-era-challenges-modern-cryptography-matthias-kaiser-o3ohe)
  * ### [ What Can Quantum Computing Do?  ](https://www.linkedin.com/pulse/what-can-quantum-computing-do-roger-grimes-j6wwe)
  * ### [ Quantum computing is overhyped. It's also about to break your security. Both are true.  Rishi (Raju) Cherukuri  ](https://www.linkedin.com/pulse/quantum-computing-overhyped-its-also-break-your-both-true-cherukuri-j6gsf)
  * ### [ QKD Is True Quantum Security Today  ](https://www.linkedin.com/pulse/qkd-true-quantum-security-today-roger-grimes-mknje)
  * ### [ The Quantum Frontier: NTT's Quest for Unbreakable Cryptography  ](https://www.linkedin.com/pulse/quantum-frontier-ntts-quest-unbreakable-cryptography-ntt-pvlle)
  * ### [ The Transformative Impact of Quantum Computing on Cryptography  Mohammad Adnan Rabbani  ](https://www.linkedin.com/pulse/transformative-impact-quantum-computing-cryptography-rabbani)
  * ### [ Responsible Computing with Utility-Scale Quantum  Matthias Troyer  ](https://www.linkedin.com/pulse/responsible-computing-utility-scale-quantum-matthias-troyer-8sdwc)
  * ### [ Quantum Isn't the Future Anymore  ](https://www.linkedin.com/pulse/quantum-isnt-future-anymore-zyncgroup-7og0e)
  * ### [ Quantum Computing and the Silent Threat to Today’s Encryption  ](https://www.linkedin.com/pulse/quantum-computing-silent-threat-todays-encryption-jeff-hamm-e8efe)


##  Similar topics 
  * ### [ Quantum Code Breaking Techniques Explained  ](https://www.linkedin.com/top-content/technology/quantum-computing-applications/quantum-code-breaking-techniques-explained/)
  * ### [ How Quantum Computing Will Transform Cryptography  ](https://www.linkedin.com/top-content/innovation/quantum-computing-developments/how-quantum-computing-will-transform-cryptography/)
  * ### [ How Qubits Advance Scientific Computing  ](https://www.linkedin.com/top-content/technology/quantum-computing-applications/how-qubits-advance-scientific-computing/)
  * ### [ Quantum Threats to Encryption Scalability  ](https://www.linkedin.com/top-content/technology/quantum-computing-applications/quantum-threats-to-encryption-scalability/)
  * ### [ Quantum Entanglement in Next-Generation Encryption  ](https://www.linkedin.com/top-content/technology/quantum-computing-applications/quantum-entanglement-in-next-generation-encryption/)
  * ### [ Quantum Computing's Impact on AES Encryption Security  ](https://www.linkedin.com/top-content/innovation/quantum-computing-developments/quantum-computing-s-impact-on-aes-encryption-security/)
  * ### [ Quantum Superposition Principles  ](https://www.linkedin.com/top-content/design/qubit-design-basics/quantum-superposition-principles/)
  * ### [ Impact of Quantum States on Algorithm Performance  ](https://www.linkedin.com/top-content/technology/quantum-computing-applications/impact-of-quantum-states-on-algorithm-performance/)
  * ### [ Quantum Field Theory Applications in Lattice Interactions  ](https://www.linkedin.com/top-content/technology/quantum-computing-applications/quantum-field-theory-applications-in-lattice-interactions/)
  * ### [ How Quantum Computing Differs From Binary Processing  ](https://www.linkedin.com/top-content/artificial-intelligence/understanding-advanced-computing/how-quantum-computing-differs-from-binary-processing/)


##  Explore content categories 


##  Sign in to view more content 
Create your free account or sign in to continue your search
Sign in with Email
New to LinkedIn? [Join now](https://www.linkedin.com/signup/cold-join?session_redirect=%2Fpulse%2Fwhy-quantum-computers-can-break-rsa-lattice-timothy-hollebeek-vi8re&trk=pulse-article_contextual-sign-in-modal_join-link)
By clicking Continue to join or sign in, you agree to LinkedIn’s [User Agreement](https://www.linkedin.com/legal/user-agreement?trk=linkedin-tc_auth-button_user-agreement), [Privacy Policy](https://www.linkedin.com/legal/privacy-policy?trk=linkedin-tc_auth-button_privacy-policy), and [Cookie Policy](https://www.linkedin.com/legal/cookie-policy?trk=linkedin-tc_auth-button_cookie-policy). 
