lines = None
with open('nn1.txt', 'r') as nn0_file:
    lines = nn0_file.readlines()
f_1 = lines[:15000]
f_2 = lines[15000:]
with open('nn1_1.txt','w') as nn1_file:
    for line in f_1:
        nn1_file.write(line)

with open('nn1_2.txt','w') as nn2_file:
    for line in f_2:
        nn2_file.write(line)

examples_str = [line.rstrip('\n').split('   ')[0] for line in open('nn1_2.txt')]
with open('nn1_3.txt','w') as f:
    for line in examples_str:
        f.write(line + '\n')
