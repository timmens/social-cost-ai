---
paginate: true
marp: true
author: Aimee Van Wynsberghe, Tim Mensinger and Stefan Höse
description: "The Social Costs of AI: Updates"
theme: custom
---

<!-- ===================================================================================
# TITLE PAGE
==================================================================================== -->

<!-- paginate: false -->
## The Social Costs of AI: Updates

<br/>

- Aimee Van Wynsberghe
- Tim Mensinger
- Stefan Höse

<br/>

**University of Bonn**

---
<!-- paginate: true -->
### Today's agenda

- Current Status

- Results

- Lessons Learned

- Next Steps

---
<!-- paginate: false -->
<!-- _class: lead -->
# Current Status

---
### Project status (last presentation)

- Build test computing infrastructure <span style="color: green;">&#10004;</span>
    - Setup energy measurement architecture <span style="color: green;">&#10004;</span>
    - Setup computing environment <span style="color: green;">&#10004;</span>
    - Run BERT model <span style="color: orange;">&#9881;</span>
- Build real computing infrastructure <span style="color: red;">&#10060;</span>
    - Get new computing hardware <span style="color: red;">&#10060;</span>
- Analyze energy data <span style="color: red;">&#10060;</span>


---
### Run BERT model

- No proper hardware yet
    - Use existing infrastructure

- No proper software for model training
    - Find software to train BERT base
    - Realize our existing hardware is too old

- Will come back to this task <span style="color: orange;">&#9881;</span>

---
### Build computing infrastructure

- Buy professional hardware system (Jetson AGX Xavier)
    - Delivered: 30. March 2022
    - Unsuccessful tests the very same day

- Realize there are problems with the system: 31. March 2022
    - Try to reflash the device until 9 p.m. of March, 31st
        - "Waiting" more than six hours (release payment)
    - Contact customer support by ticket system: 6. April 2022

- Return device: 3. June 2022

---
### Build computing infrastructure (2)

- Return device: 3. June 2022

- Get repaired device back: 4. July 2022
    - Still broken: mouse and keyboard didn't work
    - Contact customer support again

- Return device: 25. August 2022
    * After giving a deadline: 31. August 2022

- Repaired device arrives: 2. September 2022
    - It kinda works! &#x1F389;

---
### Build computing infrastructure (Back-Up Plan)

- Existing hardware was too old
    - Specifically: GPU was too old
    - Decision to buy and use a RTX 3060

- Replace this component: 13. June 2022
    - Works perfectly! (since 17. June 2022)

- Build computing infrastructure <span style="color: green;">&#10004;</span>

- Run BERT model <span style="color: green;">&#10004;</span>

---
### Run BERT model training

- Wikipedia
    - $\approx$ 1 billion words and symbols
    - Training time: 1 day
- BookCorpus
    - $\approx$ 11,000 books (read 10 books per month for 100 years)
    - $\approx$ 10 billion words and symbols
    - Training time: 6 days

---
<!-- paginate: false -->
<!-- _class: lead -->
# Results

---
<!-- paginate: false -->
<!-- _class: lead -->
## What can our AI model do?

**Notebook example**

---
### Energy consumption

- Strubell et al. (2019)
    - 1,500 kWh
    - 2,500 km (driven by average car)
    
- AI Lab
    - 40 kWh
        - Much less energy consumption
        * Possibly because RTX 3060 was released at 2020-09-17
    - 70 km (driven by average car)

---
<!-- paginate: false -->
<!-- _class: lead -->
# Whats Next?

---
### Lessons Learned

- Always have a back-up plan (we did!)

- Be less patient in B2B contexts
    * Set deadlines after experiencing the _first_ problem
    * _Be loud_ until your problem has been solved
    * The bigger the company the faster and better their reaction

---
<!-- paginate: false -->
<!-- _class: lead -->
# Thank you! <br/><hr/>Questions?
