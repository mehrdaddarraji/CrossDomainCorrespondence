# fix parsing
# x is what you printed out and sent to me 
nbb_a = open("img_a_nbbs.txt",'w')
nbb_b = open("img_b_nbbs.txt", 'w')

for arr in x:
    n_1 = arr[0]
    n_2 = arr[1]
    s1 = str(n_1[0]) + " " + str(n_1[1]) + "\n"
    s2 = str(n_2[0]) + " " + str(n_2[1]) + "\n"
    nbb_a.write(s1)
    nbb_b.write(s2)