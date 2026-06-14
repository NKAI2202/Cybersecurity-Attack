# Cybersecurity Attack Demonstration Project
GitHub: NKAI2202Location: Cambridge, UKStatus: Completed Project
Project Overview
his project is a complete practical demonstration of 14 common cybersecurity attacks, performed in a controlled, legal, and authorised environment. The goal was to understand how each attack works, how it is executed, what risks it poses, and how to defend against it. All tests were carried out on my own systems or dedicated lab environments — no unauthorised networks or devices were used.
For machine learning-based intrusion detection and analysis, I used the Edge‑IIoTset Cyber Security Dataset, available at:https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot This dataset includes real IoT/IIoT traffic data and samples of all 14 attack types, enabling me to build, train, and test ML models to detect and classify threats.
This work shows understanding of offensive security, vulnerabilities, exploitation techniques, defence strategies, and data‑driven threat detection — core skills in cybersecurity, ethical hacking, and network defence.
Attacks Demonstrated
Below is the full list of 14 attacks covered in this project, with clear explanations of what each one does:
1. Footprinting & Reconnaissance
Gathering public information about a target (IP addresses, domains, emails, network details)
Techniques: WHOIS lookup, DNS enumeration, search engine queries, network scanning
Purpose: First step to identify possible entry points
2. Network Scanning
Probing networks to find active devices, open ports, and running services
Tools used: Nmap, Ping, Traceroute
Purpose: Map the network and find which services may be vulnerable
3. Enumeration
Extracting detailed information from systems: user accounts, shared folders, network resources
Techniques: SMB enumeration, SNMP enumeration, LDAP enumeration
Purpose: Collect data needed to plan further attacks
4. Password Attacks
Testing or cracking passwords to gain unauthorised access
Types: Brute force, dictionary attack, rainbow tables, hash cracking
Tools used: Hydra, John the Ripper
Purpose: Bypass authentication
5. ARP Spoofing / Poisoning
Sending fake ARP messages to trick the network into linking my MAC address to the target’s IP
Result: All traffic from the target passes through my machine (Man‑in‑the‑Middle)
Purpose: Intercept or modify data in transit
6. DNS Spoofing / Poisoning
Corrupting DNS cache so that users are sent to fake websites instead of real ones
Purpose: Redirect traffic, steal credentials, or deliver malware
7. Man‑in‑the‑Middle (MITM) Attack
Positioning between two communicating parties to secretly monitor or alter data
Includes: Session hijacking, traffic manipulation
Purpose: Read or change information without either side knowing
8. Sniffing & Traffic Analysis
Capturing and analysing network packets to see what data is being sent
Tools used: Wireshark, Tcpdump
Purpose: Find sensitive data (passwords, emails, files) sent in plain text
9. Phishing
Creating fake login pages or emails that look real, to trick users into entering credentials
Demonstrated: How fake pages are made, how they collect data, and how to detect them
Purpose: Social engineering to steal information
10. SQL Injection
Inserting malicious code into input fields to manipulate or access databases
Example: Bypassing login without a password, viewing hidden data
Purpose: Access, modify, or delete backend database information
11. Cross‑Site Scripting (XSS)
Injecting malicious scripts into websites that other users then run
Types: Stored, reflected, DOM‑based
Purpose: Steal cookies, redirect users, or deface websites
12. Cross‑Site Request Forgery (CSRF)
Tricking a logged‑in user into performing actions they did not intend
Example: Changing account details or making transactions
Purpose: Perform actions on behalf of a victim
13. Denial of Service (DoS)
Flooding a target system or network with too much traffic, making it slow or unavailable
Techniques: ICMP flood, SYN flood, HTTP flood
Purpose: Disrupt service and stop legitimate users from accessing it
14. Session Hijacking
Stealing or taking over an active user session after they have logged in
Methods: Stealing session cookies, exploiting insecure connections
Purpose: Access the system as if I were the authorised user
For Each Attack I Included
Explanation: What it is and how it works
Step‑by‑Step Execution: Exactly how I performed it in the lab
Tools Used: Software, commands, or scripts
Risk Level: How dangerous it is and what damage it can cause
Defence & Mitigation: How to prevent, detect, or stop the attack
Evidence: Screenshots, logs, or results
Key Learning Outcomes
How attackers think and operate
Where common vulnerabilities exist in networks, websites, and systems
How to identify and fix security weaknesses
Legal and ethical boundaries of cybersecurity testing
How to configure firewalls, encryption, and secure authentication
Best practices for securing systems and data
Technologies & Tools Used
Network Tools: Nmap, Wireshark, Netcat, Traceroute
Attacks Frameworks: Metasploit, Ettercap, Social Engineering Toolkit
Password Tools: Hydra, John the Ripper
Web Testing: Burp Suite, OWASP ZAP
Operating Systems: Kali Linux, Ubuntu, Windows
Scripting: Basic Python & Bash for automation
Repository Structure

plaintext
/
├── README.md                # This file — overview of the project
├── results/            # Proof and evidence of each attack
└── Data_preprocessing_and_training.py  # FULL PROJECT: all files, docs, and resources
Note: All content here is for educational and ethical purposes only. No real external systems were targeted or harmed.
Skills Demonstrated
Ethical hacking & penetration testing
Network security & traffic analysis
Web application security
Social engineering awareness
Vulnerability assessment & risk analysis
Security defence & mitigation strategies
Technical documentation & reporting
About Me
NKAI2202 — Final‑year Artificial Intelligence student based in Cambridge, UK.Available immediately for roles in Cybersecurity, Ethical Hacking, Network Security, or IT Security.
